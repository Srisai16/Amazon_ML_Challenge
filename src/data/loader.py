"""
Memory-safe dataset encoder.

The raw challenge files are ~2.5 GB of TSV holding ~11.7 M records per split.
Loading them with ``pandas.read_csv`` needs tens of gigabytes, which does not
fit on the target machine, so every split is instead *encoded once* into compact
NumPy arrays on disk and subsequently memory-mapped.

On-disk layout, per source file (one prefix per file)::

    <prefix>.id     S12    raw entity_id, truncated
    <prefix>.idh    i8     64-bit FNV hash of entity_id (join key)
    <prefix>.name   S88    normalized business name
    <prefix>.addr   S160   normalized business address
    <prefix>.ntok   u4     CSR values - hashed name tokens
    <prefix>.nptr   i8     CSR offsets  (n_rows + 1)
    <prefix>.atok   u4     CSR values - hashed address tokens
    <prefix>.aptr   i8     CSR offsets  (n_rows + 1)

The two token namespaces are kept in separate CSR arrays, and each token hash is
namespaced by an initial CRC seed so that a name token can never collide with an
address token.

Everything is streamed, so peak RSS stays in the tens of megabytes regardless of
file size.
"""

import csv
import os
import sys
import time
from typing import Dict, Optional, Tuple
from zlib import crc32

import numpy as np

from src.data import fastnorm as fn

# CSV fields can be long; raise the hard limit so csv never raises mid-stream.
csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

# ---------------------------------------------------------------------------
# On-disk dtypes / widths
# ---------------------------------------------------------------------------
ID_W = 12
NAME_W = 88
ADDR_W = 160

MAX_NAME_TOK = 24
MAX_ADDR_TOK = 24

# Namespace seeds for token hashing (distinct from any CRC-relevant value).
NS_NAME = 0x6E
NS_ADDR = 0x61

# Dense document-frequency table: 2**24 int32 slots = 64 MB, touched sparsely.
DF_BITS = 24
DF_SIZE = 1 << DF_BITS
DF_MASK = DF_SIZE - 1

# Buffered rows are flushed into the token chunk list (and folded into the
# document-frequency table) every FLUSH_ROWS rows; progress is reported on a
# coarser PROGRESS_ROWS cadence.
FLUSH_ROWS = 200_000
PROGRESS_ROWS = 1_000_000


def fld(prefix: str, name: str) -> str:
    """Canonical cache path for one field of one encoded file.

    Every cache file is named ``<prefix>.<field>.npy``. Spelling the suffix
    explicitly matters: ``np.save`` silently appends ``.npy`` when it is
    missing, which previously produced unreadable cache files.
    """
    return f"{prefix}.{name}.npy"


# ---------------------------------------------------------------------------
# Vectorized hashing of fixed-width byte arrays
# ---------------------------------------------------------------------------
def hash_fixed_bytes(arr: np.ndarray) -> np.ndarray:
    """64-bit FNV-1a over a ``S{n}`` array, computed without Python loops.

    FNV constants are chosen so that the *distinct* token hashes stay spread
    across the dense df table even for short ASCII payloads.
    """
    if arr.dtype.kind != "S":
        raise TypeError(f"expected a fixed-width bytes array, got {arr.dtype}")
    width = arr.dtype.itemsize
    raw = np.ascontiguousarray(arr).view(np.uint8).reshape(len(arr), width)

    h = np.full(len(arr), 0xCBF29CE484222325, dtype=np.uint64)
    prime = np.uint64(0x100000001B3)
    with np.errstate(over="ignore"):
        for j in range(width):
            h = (h ^ raw[:, j].astype(np.uint64)) * prime
    return h.view(np.int64)


def _hash_id_str(s: str) -> int:
    """Stable 64-bit hash of an entity_id string (matches ``hash_fixed_bytes``)."""
    b = s.encode("utf-8", "ignore")[:ID_W]
    raw = b.ljust(ID_W, b"\x00")
    h = 0xCBF29CE484222325
    for byte in raw:
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    # Fold to a signed 64-bit value so it round-trips through np.int64.
    return h - 0x10000000000000000 if h >= 0x8000000000000000 else h


# ---------------------------------------------------------------------------
# Row counting
# ---------------------------------------------------------------------------
def count_rows(path: str) -> int:
    """Number of data rows (header excluded) in a TSV file."""
    total = 0
    block = 1 << 24
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(block)
            if not chunk:
                break
            total += chunk.count(b"\n")
    # Drop the header, and tolerate a missing trailing newline.
    return max(0, total - 1)


# ---------------------------------------------------------------------------
# Source encoding
# ---------------------------------------------------------------------------
def encode_source(
    tsv_path: str,
    out_prefix: str,
    n_rows: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, int]:
    """Stream one ``*_source{1,2,3}.tsv`` into the on-disk array layout."""
    if n_rows is None:
        n_rows = count_rows(tsv_path)

    ids = np.lib.format.open_memmap(
        fld(out_prefix, "id"), mode="w+", dtype=f"S{ID_W}", shape=(n_rows,))
    idh = np.lib.format.open_memmap(
        fld(out_prefix, "idh"), mode="w+", dtype="i8", shape=(n_rows,))
    names = np.lib.format.open_memmap(
        fld(out_prefix, "name"), mode="w+", dtype=f"S{NAME_W}", shape=(n_rows,))
    addrs = np.lib.format.open_memmap(
        fld(out_prefix, "addr"), mode="w+", dtype=f"S{ADDR_W}", shape=(n_rows,))

    # CSR token arrays are appended in chunks, then concatenated at the end.
    ntok_chunks: list = []
    atok_chunks: list = []
    nptr = np.zeros(n_rows + 1, dtype=np.int64)

    df_table = np.zeros(DF_SIZE, dtype=np.int32)

    base_name = fn.normalize_name
    base_addr = fn.normalize_addr
    n_toks = fn.name_tokens
    a_toks = fn.addr_tokens
    seed_n, seed_a = NS_NAME, NS_ADDR

    t0 = time.time()
    row = 0
    n_tok = 0

    def flush() -> None:
        """Move buffered tokens into the chunk list and fold their df in.

        Document frequencies must be accumulated here, on the same schedule as
        the chunking itself, so that they can never be skipped for inputs
        smaller than one progress-reporting interval.
        """
        if not buf_n:
            return
        cn = np.concatenate(buf_n)
        ca = np.concatenate(buf_a)
        ntok_chunks.append(cn)
        atok_chunks.append(ca)
        buf_n.clear()
        buf_a.clear()
        _accumulate_df(df_table, cn)
        _accumulate_df(df_table, ca)

    with open(tsv_path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader, None)  # header

        buf_n: list = []
        buf_a: list = []
        for parts in reader:
            if row >= n_rows:
                break

            eid = parts[0] if parts else ""
            raw_name = parts[1] if len(parts) > 1 else ""
            raw_addr = parts[2] if len(parts) > 2 else ""

            name = base_name(raw_name)
            addr = base_addr(raw_addr)

            nt = n_toks(name)
            at = a_toks(addr)
            if len(nt) > MAX_NAME_TOK:
                nt = nt[:MAX_NAME_TOK]
            if len(at) > MAX_ADDR_TOK:
                at = at[:MAX_ADDR_TOK]

            ids[row] = eid.encode("utf-8", "ignore")[:ID_W]
            idh[row] = _hash_id_str(eid)
            names[row] = name.encode("utf-8", "ignore")[:NAME_W]
            addrs[row] = addr.encode("utf-8", "ignore")[:ADDR_W]

            hn = [crc32(t.encode("utf-8", "ignore"), seed_n) for t in nt]
            ha = [crc32(t.encode("utf-8", "ignore"), seed_a) for t in at]
            buf_n.append(np.array(hn, dtype=np.uint32) if hn else _EMPTY_U32)
            buf_a.append(np.array(ha, dtype=np.uint32) if ha else _EMPTY_U32)
            n_tok += len(hn) + len(ha)

            row += 1
            nptr[row] = n_tok

            if len(buf_n) >= FLUSH_ROWS:
                flush()
                if verbose and row % PROGRESS_ROWS < FLUSH_ROWS:
                    done = row / n_rows
                    el = time.time() - t0
                    eta = el / done - el if done > 0 else 0.0
                    print(f"    {os.path.basename(tsv_path)}: {row:,}/{n_rows:,} "
                          f"({done:6.1%}) elapsed {el:6.0f}s eta {eta:6.0f}s",
                          flush=True)

    flush()
    if not ntok_chunks:
        ntok_chunks = [_EMPTY_U32]
        atok_chunks = [_EMPTY_U32]

    np.save(fld(out_prefix, "ntok"), np.concatenate(ntok_chunks))
    np.save(fld(out_prefix, "nptr"), nptr)
    np.save(fld(out_prefix, "atok"), np.concatenate(atok_chunks))
    np.save(fld(out_prefix, "aptr"), nptr.copy())
    np.save(fld(out_prefix, "df"), df_table)

    for mm in (ids, idh, names, addrs):
        mm.flush()
    del ids, idh, names, addrs

    stats = {"rows": row, "tokens": n_tok, "seconds": time.time() - t0}
    if verbose:
        print(f"  encoded {os.path.basename(tsv_path)}: {row:,} rows, "
              f"{n_tok:,} tokens in {stats['seconds']:.0f}s", flush=True)
    return stats


_EMPTY_U32 = np.empty(0, dtype=np.uint32)


def _accumulate_df(df_table: np.ndarray, hashes: np.ndarray) -> None:
    """Add one posting to the dense document-frequency table."""
    if hashes.size == 0:
        return
    idx = (hashes & np.uint32(DF_MASK)).astype(np.intp)
    np.add.at(df_table, idx, 1)


# ---------------------------------------------------------------------------
# Ground-truth encoding
# ---------------------------------------------------------------------------
def encode_ground_truth(
    gt_path: str,
    s1_prefix: str,
    tgt_prefixes: Tuple[str, ...],
    out_prefix: str,
    verbose: bool = True,
) -> Dict[str, int]:
    """Encode ``train_ground_truth.tsv`` as a sorted array of pair keys.

    A positive pair ``(s1_row, tgt_row)`` is stored as the single int64
    ``s1_row * n_targets + tgt_row``. Because the array is sorted, all positives
    for one Source 1 row occupy a contiguous slice, and membership of an
    arbitrary candidate pair is a single ``searchsorted``. This keeps the whole
    label structure in ~60 MB instead of a Python dict of millions of entries.
    """
    s1_idh = np.asarray(np.load(fld(s1_prefix, "idh"), mmap_mode="r"))
    n_s1 = len(s1_idh)
    # searchsorted needs a sorted key array, and idh is stored in file order.
    s1_order = np.argsort(s1_idh, kind="stable")
    s1_sorted = s1_idh[s1_order]

    # Global target pool: S2 rows first, then S3 rows.
    tgt_hashes = [np.asarray(np.load(fld(p, "idh"), mmap_mode="r")) for p in tgt_prefixes]
    n_targets = int(sum(len(h) for h in tgt_hashes))
    tgt_all = np.concatenate(tgt_hashes)
    del tgt_hashes
    order = np.argsort(tgt_all, kind="stable")
    sorted_hashes = tgt_all[order]
    del tgt_all

    # ---- Pass 1: read the file into flat hash arrays -----------------------
    s1_h_chunks: list = []
    tgt_h_chunks: list = []
    cnt_chunks: list = []
    t0 = time.time()
    n_gt_rows = 0

    with open(gt_path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader, None)
        buf_s1: list = []
        buf_tgt: list = []
        buf_cnt: list = []
        for parts in reader:
            if len(parts) < 2:
                continue
            buf_s1.append(_hash_id_str(parts[0].strip()))
            raw = parts[1].strip()
            if raw:
                ids = [m.strip() for m in raw.split(",") if m.strip()]
                if ids:
                    buf_tgt.append(np.fromiter(
                        (_hash_id_str(m) for m in ids), dtype=np.int64, count=len(ids)))
                    buf_cnt.append(len(ids))
                else:
                    buf_tgt.append(_EMPTY_I64)
                    buf_cnt.append(0)
            else:
                buf_tgt.append(_EMPTY_I64)
                buf_cnt.append(0)
            n_gt_rows += 1

            if len(buf_s1) >= 1_000_000:
                s1_h_chunks.append(np.array(buf_s1, dtype=np.int64))
                tgt_h_chunks.append(np.concatenate(buf_tgt))
                cnt_chunks.append(np.array(buf_cnt, dtype=np.int64))
                buf_s1.clear()
                buf_tgt.clear()
                buf_cnt.clear()
                if verbose:
                    print(f"    ground truth: {n_gt_rows:,} rows read "
                          f"({time.time() - t0:.0f}s)", flush=True)

        if buf_s1:
            s1_h_chunks.append(np.array(buf_s1, dtype=np.int64))
            tgt_h_chunks.append(np.concatenate(buf_tgt))
            cnt_chunks.append(np.array(buf_cnt, dtype=np.int64))

    s1_h = np.concatenate(s1_h_chunks)
    tgt_h = np.concatenate(tgt_h_chunks)
    tgt_cnt = np.concatenate(cnt_chunks)
    del s1_h_chunks, tgt_h_chunks, cnt_chunks
    del buf_s1, buf_tgt, buf_cnt

    # ---- Pass 2: resolve both sides vectorized ----------------------------
    p1 = np.searchsorted(s1_sorted, s1_h)
    p1c = np.minimum(p1, n_s1 - 1)
    ok_s1 = (p1 < n_s1) & (s1_sorted[p1c] == s1_h)
    s1_row = np.where(ok_s1, s1_order[p1c], -1).astype(np.int64)

    p2 = np.searchsorted(sorted_hashes, tgt_h)
    p2c = np.minimum(p2, len(sorted_hashes) - 1)
    ok_tgt = (p2 < len(sorted_hashes)) & (sorted_hashes[p2c] == tgt_h)
    tgt_row = np.where(ok_tgt, order[p2c], -1).astype(np.int64)
    del sorted_hashes, order, p1, p1c, p2, p2c, s1_order, s1_sorted

    # Expand the per-row counts so every pair knows its owning S1 row.
    owner = np.repeat(s1_row, tgt_cnt)
    keep = ok_tgt & (owner >= 0)

    n_pairs = int(keep.sum())
    n_resolved = int(ok_s1.sum())
    pair_owner = owner[keep]
    pair_tgt = tgt_row[keep]
    del owner, keep, ok_tgt, ok_s1, s1_h, tgt_h, tgt_cnt, s1_row, tgt_row

    # Sort by (owner, target) so each S1 row's positives are contiguous.
    keys = pair_owner * n_targets + pair_tgt
    keys.sort()

    # Per-S1-row positive counts, indexed by row over the full S1 table.
    counts = np.bincount(pair_owner, minlength=n_s1).astype(np.int64)
    np.save(fld(out_prefix, "pos_keys"), keys)
    np.save(fld(out_prefix, "pos_counts"), counts)

    n_singletons = int((counts == 0).sum())
    if verbose:
        print(f"  encoded ground truth: {n_gt_rows:,} rows -> {n_pairs:,} pairs, "
              f"{n_singletons:,} singletons, "
              f"{n_resolved:,} S1 ids resolved "
              f"({time.time() - t0:.0f}s)", flush=True)

    return {"pairs": n_pairs, "singletons": n_singletons,
            "unresolved": int(n_gt_rows - n_resolved)}


_EMPTY_I64 = np.empty(0, dtype=np.int64)
