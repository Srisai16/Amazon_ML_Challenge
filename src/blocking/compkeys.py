"""
Composite-key blocking index.

Why this exists
---------------
A pure rare-token index is *sparse*: a record whose name is "Prime Money" has
both tokens far above any usable document-frequency cap, so it contributes no
posting at all and can never be retrieved. Measured on the training split that
degenerate case drove pair completeness to 0.00.

Composite keys fix this by hashing *combinations* of fields that are naturally
highly selective - an exact business name, a name plus its house number, a
house number plus a postal code. Such keys have short posting lists by
construction, so they can be indexed with a generous document-frequency cap
without exploding the candidate volume, while still catching the word-order
transpositions and address noise that defeat naive exact matching.

The same :func:`doc_keys` function produces keys for both the target pool and
the queries, which is what guarantees the two sides agree.
"""

import os
from typing import Iterator, List, Optional, Sequence, Tuple
from zlib import crc32

import numpy as np

from src.data import fastnorm as fn
from src.data.loader import DF_MASK, DF_SIZE, fld

# One CRC seed per key family, so a key from one family can never alias another.
SEED_FULL_NAME = 0x11
SEED_SORTED_NAME = 0x12
SEED_NAME_ADDRNO = 0x13
SEED_NAME_POSTAL = 0x14
SEED_ADDRNO_POSTAL = 0x15
SEED_ADDRNO_STREET = 0x16
SEED_PARTIAL_NAME = 0x17
SEED_POSTAL_NAME1 = 0x18
SEED_NAME_FIRST2 = 0x19
SEED_NAME1_ADDRNO = 0x20
SEED_NAME1_STREET = 0x21
SEED_NAME1_POSTAL = 0x22
SEED_ADDR_TOK_PAIR = 0x23
SEED_PREFIX4_POSTAL = 0x24
SEED_PREFIX4_ADDRNO = 0x25

# Street-type and noise words skipped when extracting distinctive address tokens
_STREET_TYPES = {
    "st", "rd", "ave", "blvd", "dr", "ln", "hwy", "pkwy", "ct", "cir", "pl",
    "plz", "sq", "ter", "trl", "way", "aly", "apt", "ste", "fl", "bldg",
    "blk", "twr", "rue", "allee", "chemin", "imp", "quai", "cours", "rond",
    "esp", "fbg", "opp", "near", "beside", "behind", "n", "s", "e", "w",
    "ne", "nw", "se", "sw", "h", "no", "plot", "shop", "flat", "gali", "room",
    "floor", "sector", "sec", "flr", "unit", "in", "at", "to", "by", "of",
    "and", "the", "dl", "ma", "tx", "ca", "ny", "il", "pa", "oh", "ga", "nc",
}

MAX_KEYS_PER_DOC = 14

# Score contribution of each key family
FAMILY_WEIGHT = {
    SEED_FULL_NAME: 16.0,
    SEED_SORTED_NAME: 14.0,
    SEED_NAME_FIRST2: 13.5,
    SEED_NAME_ADDRNO: 13.0,
    SEED_NAME1_ADDRNO: 12.5,
    SEED_NAME_POSTAL: 12.0,
    SEED_NAME1_POSTAL: 11.5,
    SEED_NAME1_STREET: 11.0,
    SEED_PARTIAL_NAME: 11.0,
    SEED_PREFIX4_POSTAL: 10.5,
    SEED_PREFIX4_ADDRNO: 10.5,
    SEED_ADDRNO_STREET: 10.5,
    SEED_ADDRNO_POSTAL: 10.0,
    SEED_POSTAL_NAME1: 10.0,
    SEED_ADDR_TOK_PAIR: 9.5,
}


def doc_keys(name_norm: str, addr_norm: str) -> List[Tuple[int, float]]:
    """Composite blocking keys for one record, each with its family weight.

    Shared verbatim by the target-index build and the query path. Returns a list
    of ``(key_hash, weight)`` pairs, possibly empty.
    """
    if not name_norm and not addr_norm:
        return []

    core = [t for t in name_norm.split() if t and t not in fn.LEGAL_DROP]

    addr_toks = addr_norm.split() if addr_norm else []
    digits = [t for t in addr_toks if t.isdigit()]

    # House number: the leading short numeric token of the address.
    addr_no = ""
    for t in addr_toks:
        if t.isdigit() and 1 <= len(t) <= 5:
            addr_no = t
            break
    if not addr_no and digits:
        addr_no = digits[0]

    # Postal code: a 6-digit PIN (India) or 5-digit code (US / France).
    postal = ""
    for t in reversed(digits):
        if len(t) in (5, 6):
            postal = t
            break

    # Distinctive street/address words
    distinctive_addr = [t for t in addr_toks if not t.isdigit() and t not in _STREET_TYPES and len(t) >= 3]

    keys: List[Tuple[int, float]] = []

    def add(seed: int, text: str) -> None:
        keys.append((crc32(text.encode("utf-8"), seed), FAMILY_WEIGHT.get(seed, 10.0)))

    if core:
        full = " ".join(core)
        add(SEED_FULL_NAME, full)
        add(SEED_SORTED_NAME, " ".join(sorted(core)))
        if len(core) >= 2:
            add(SEED_NAME_FIRST2, core[0] + " " + core[1])
        if len(core) >= 3:
            add(SEED_PARTIAL_NAME, " ".join(sorted(core)[:3]))

        name1 = core[0]
        if addr_no:
            add(SEED_NAME_ADDRNO, full + "|" + addr_no)
            add(SEED_NAME1_ADDRNO, name1 + "|" + addr_no)
            if len(name1) >= 4:
                add(SEED_PREFIX4_ADDRNO, name1[:4] + "|" + addr_no)
        if postal:
            add(SEED_NAME_POSTAL, full + "|" + postal)
            add(SEED_POSTAL_NAME1, postal + "|" + name1)
            add(SEED_NAME1_POSTAL, name1 + "|" + postal)
            if len(name1) >= 4:
                add(SEED_PREFIX4_POSTAL, name1[:4] + "|" + postal)
        if distinctive_addr:
            for st in distinctive_addr[:2]:
                add(SEED_NAME1_STREET, name1 + "|" + st)

    if addr_no and postal:
        add(SEED_ADDRNO_POSTAL, addr_no + "|" + postal)

    if addr_no and distinctive_addr:
        for st in distinctive_addr[:3]:
            add(SEED_ADDRNO_STREET, addr_no + "|" + st)

    if len(distinctive_addr) >= 2:
        for i in range(min(len(distinctive_addr) - 1, 2)):
            add(SEED_ADDR_TOK_PAIR, distinctive_addr[i] + "|" + distinctive_addr[i + 1])

    # De-duplicate while preserving order, then cap. When two families collide
    # on the same hash the stronger evidence wins.
    best: dict = {}
    order: List[int] = []
    for k, w in keys:
        if k not in best:
            best[k] = w
            order.append(k)
        elif w > best[k]:
            best[k] = w
    return [(k, best[k]) for k in order][:MAX_KEYS_PER_DOC]


def _iter_keys(name_mm, addr_mm, lo: int, hi: int, row_offset: int
               ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield ``(key_array, row_array)`` for rows ``[lo, hi)`` of a memmap pair."""
    keys: List[int] = []
    rows: List[int] = []
    for r in range(lo, hi):
        n = name_mm[r]
        a = addr_mm[r]
        ks = doc_keys(n.decode("utf-8", "ignore"),
                      a.decode("utf-8", "ignore"))
        if ks:
            keys.extend(k for k, _w in ks)
            rows.extend([r + row_offset] * len(ks))
    if keys:
        yield (np.array(keys, dtype=np.uint32),
               np.array(rows, dtype=np.int64))
    else:
        yield (np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.int64))


class CompositeIndex:
    """Slot-grouped postings over composite keys, for one target source."""

    def __init__(self, tok: np.ndarray, row: np.ndarray,
                 offset: np.ndarray):
        self.tok = tok
        self.row = row
        self.offset = offset

    def lookup(self, key: int) -> Tuple[int, int]:
        slot = int(key) & DF_MASK
        return int(self.offset[slot]), int(self.offset[slot + 1])

    def get(self, key: int) -> np.ndarray:
        start, stop = self.lookup(key)
        if stop <= start:
            return np.empty(0, dtype=np.int32)
        toks = np.asarray(self.tok[start:stop])
        mask = toks == np.uint32(key)
        return np.asarray(self.row[start:stop])[mask]


def build_composite_index(target_prefix: str, out_prefix: str,
                          row_offset: int, df_cap: int,
                          verbose: bool = True) -> CompositeIndex:
    """Build one composite-key postings table via a disk-backed counting sort."""
    name_mm = np.load(fld(target_prefix, "name"), mmap_mode="r")
    addr_mm = np.load(fld(target_prefix, "addr"), mmap_mode="r")
    n = len(name_mm)

    BLOCK = 250_000
    # ---- Pass 1: count keys per slot -----------------------------------
    counts = np.zeros(DF_SIZE, dtype=np.int64)
    total = 0
    for lo in range(0, n, BLOCK):
        hi = min(lo + BLOCK, n)
        keys, _ = next(_iter_keys(name_mm, addr_mm, lo, hi, row_offset))
        if keys.size == 0:
            continue
        slots = (keys & np.uint32(DF_MASK)).astype(np.intp)
        np.add.at(counts, slots, 1)
        total += keys.size
    offset = np.zeros(DF_SIZE + 1, dtype=np.int64)
    np.cumsum(counts, out=offset[1:])
    if verbose:
        print(f"    {os.path.basename(target_prefix)}: {total:,} composite keys "
              f"over {int((counts > 0).sum()):,} slots", flush=True)

    # ---- Pass 2: place --------------------------------------------------
    tok_out = np.lib.format.open_memmap(
        fld(out_prefix, "ck_tok"), mode="w+", dtype=np.uint32, shape=(total,))
    row_out = np.lib.format.open_memmap(
        fld(out_prefix, "ck_row"), mode="w+", dtype=np.int32, shape=(total,))

    cursor = offset[:-1].copy()
    placed = 0
    for lo in range(0, n, BLOCK):
        hi = min(lo + BLOCK, n)
        keys, rows = next(_iter_keys(name_mm, addr_mm, lo, hi, row_offset))
        if keys.size == 0:
            continue
        slots = (keys & np.uint32(DF_MASK)).astype(np.intp)
        order = np.argsort(slots, kind="stable")
        for s, t, r in zip(slots[order].tolist(), keys[order].tolist(),
                           rows[order].tolist()):
            pos = cursor[s]
            tok_out[pos] = t
            row_out[pos] = r
            cursor[s] = pos + 1
        placed += keys.size

    tok_out.flush()
    row_out.flush()
    if placed != total:
        raise RuntimeError(
            f"composite key mismatch: counted {total:,} placed {placed:,}")
    np.save(fld(out_prefix, "ck_offset"), offset)

    del tok_out, row_out, cursor, counts, offset
    return CompositeIndex(
        np.load(fld(out_prefix, "ck_tok"), mmap_mode="r"),
        np.load(fld(out_prefix, "ck_row"), mmap_mode="r"),
        np.load(fld(out_prefix, "ck_offset"), mmap_mode="r"),
    )


def query_keys(s1_prefix: str, lo: int, hi: int) -> List[List[Tuple[int, float]]]:
    """Composite keys for a contiguous block of Source 1 rows."""
    name_mm = np.load(fld(s1_prefix, "name"), mmap_mode="r")
    addr_mm = np.load(fld(s1_prefix, "addr"), mmap_mode="r")
    out: List[List[Tuple[int, float]]] = []
    for r in range(lo, hi):
        out.append(doc_keys(name_mm[r].decode("utf-8", "ignore"),
                            addr_mm[r].decode("utf-8", "ignore")))
    return out
