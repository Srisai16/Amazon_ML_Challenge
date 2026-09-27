"""
Memory-bounded rare-token inverted index.

Why this shape
--------------
The target pool holds ~10.3 M records and Source 1 holds ~1.7-2.2 M queries. A
classic ``rank_bm25`` / dense ``numpy.dot`` approach needs either an
``N_queries x N_targets`` score matrix (terabytes) or a per-query linear scan
over every target (trillions of Python operations). Both are hopeless here.

Instead we exploit the fact that matching records almost always *share rare
tokens* - a distinctive surname in the business name, a house number, a postal
code, a street name. Tokens that are too common to be discriminative are
dropped using a dense document-frequency table, which turns the problem into a
sparse incidence join.

Memory strategy
---------------
Postings are grouped by token via a **counting sort into a disk-backed memmap**,
so the only in-RAM structures are three 2**24-entry tables (192 MB total). The
alternative - materialising and globally sorting ~80 M (token, row) pairs -
would need several gigabytes of peak RAM for the argsort alone.

Postings for one slot are stored contiguously. Because 32-bit token hashes
collide occasionally, a query must filter its slot by exact token value; the
resulting false candidates are harmless because the GBDT reranker re-scores
every candidate anyway.
"""

import os
from typing import Optional, Tuple

import numpy as np

from src.data import loader
from src.data.loader import DF_BITS, DF_MASK, DF_SIZE, fld

# Tunables. These trade recall against candidate volume.
MAX_TOK_PER_DOC = 4        # name tokens kept per target document
MAX_TOK_PER_DOC_ADDR = 4   # address tokens kept per target document
QUERY_TOK_NAME = 5         # name tokens used per query
QUERY_TOK_ADDR = 5         # address tokens used per query


class TokenPostings:
    """A grouped, disk-backed postings table over one token namespace."""

    def __init__(self, tok: np.ndarray, row: np.ndarray,
                 offset: np.ndarray, df: np.ndarray):
        self.tok = tok
        self.row = row
        self.offset = offset
        self.df = df

    @property
    def n_postings(self) -> int:
        return len(self.tok)

    def lookup(self, token: int) -> Tuple[slice, np.ndarray, np.ndarray]:
        """Return the (start, stop) range of ``token`` inside its slot."""
        slot = int(token) & DF_MASK
        return self.offset[slot], self.offset[slot + 1]


def _select_doc_indices(
    toks: np.ndarray,
    df: np.ndarray,
    max_per_doc: int,
    df_cap: int,
) -> np.ndarray:
    """Indices of the tokens of one document that should enter the index.

    A token qualifies when its document frequency is non-zero (so it can be
    joined on) and at most ``df_cap`` (so its posting list stays short). When
    more than ``max_per_doc`` qualify, the *rarest* ones win.

    Both the counting pass and the placement pass of :func:`build_postings` call
    this exact function, which is what guarantees the two passes agree on the
    total number of postings.
    """
    slots = (toks & np.uint32(DF_MASK)).astype(np.intp)
    freqs = df[slots]
    idx = np.flatnonzero((freqs > 0) & (freqs <= df_cap))
    if idx.size == 0:
        return idx
    # A document can repeat a token ("main st main st"). Emitting the same
    # (token, row) posting twice would double-count it in the counting pass and
    # inflate the IDF vote, so keep only the first occurrence.
    _, first_pos = np.unique(toks[idx], return_index=True)
    idx = idx[np.sort(first_pos)]
    if idx.size > max_per_doc:
        # Rank by frequency, keep the rarest, then restore positional order so
        # postings are written in a deterministic sequence.
        keep = np.argsort(freqs[idx], kind="stable")[:max_per_doc]
        idx = idx[np.sort(keep)]
    return idx


def build_postings(
    tok_path: str,
    ptr_path: str,
    df: np.ndarray,
    out_prefix: str,
    max_per_doc: int,
    df_cap: int,
    verbose: bool = True,
) -> TokenPostings:
    """Counting-sort a token CSR into a disk-backed, slot-grouped postings table.

    ``tok_path``/``ptr_path`` are the cached CSR arrays. The two passes re-read
    the memmap instead of materialising the full selection in RAM.
    """
    tok_mm = np.load(tok_path, mmap_mode="r")
    ptr_mm = np.load(ptr_path, mmap_mode="r")
    n_docs = len(ptr_mm) - 1

    # ---- Pass 1: count postings per slot --------------------------------
    counts = np.zeros(DF_SIZE, dtype=np.int64)
    total = 0
    for d in range(n_docs):
        lo, hi = int(ptr_mm[d]), int(ptr_mm[d + 1])
        if lo == hi:
            continue
        idx = _select_doc_indices(np.asarray(tok_mm[lo:hi]), df,
                                  max_per_doc, df_cap)
        if idx.size == 0:
            continue
        toks = np.asarray(tok_mm[lo:hi])
        slots = (toks[idx] & np.uint32(DF_MASK)).astype(np.intp)
        # One posting per selected token, so count each selected slot once.
        # Using a de-duplicated slot list here would under-count whenever one
        # document's selected tokens span several different slots.
        np.add.at(counts, slots, 1)
        total += idx.size

    offset = np.zeros(DF_SIZE + 1, dtype=np.int64)
    np.cumsum(counts, out=offset[1:])
    if verbose:
        print(f"    postings: {total:,} entries over "
              f"{int((counts > 0).sum()):,} slots", flush=True)

    # ---- Pass 2: place postings ----------------------------------------
    tok_out = np.lib.format.open_memmap(
        fld(out_prefix, "post_tok"), mode="w+", dtype=np.uint32, shape=(total,))
    row_out = np.lib.format.open_memmap(
        fld(out_prefix, "post_row"), mode="w+", dtype=np.int32, shape=(total,))

    cursor = offset[:-1].copy()
    placed = 0
    for d in range(n_docs):
        lo, hi = int(ptr_mm[d]), int(ptr_mm[d + 1])
        if lo == hi:
            continue
        toks = np.asarray(tok_mm[lo:hi])
        idx = _select_doc_indices(toks, df, max_per_doc, df_cap)
        if idx.size == 0:
            continue
        sel_t = toks[idx]
        sel_s = (sel_t & np.uint32(DF_MASK)).astype(np.intp)
        # A document's tokens are unique, so its selected slots are unique and
        # a sequential write per slot is safe.
        for t, s in zip(sel_t.tolist(), sel_s.tolist()):
            pos = cursor[s]
            tok_out[pos] = t
            row_out[pos] = d
            cursor[s] = pos + 1
        placed += idx.size

    tok_out.flush()
    row_out.flush()
    if placed != total:
        raise RuntimeError(
            f"posting count mismatch: counted {total:,} but placed {placed:,}. "
            f"The selection helper is not deterministic across passes.")

    np.save(fld(out_prefix, "post_offset"), offset)
    np.save(fld(out_prefix, "post_df"), df)

    del tok_out, row_out, cursor, counts, offset
    return TokenPostings(
        np.load(fld(out_prefix, "post_tok"), mmap_mode="r"),
        np.load(fld(out_prefix, "post_row"), mmap_mode="r"),
        np.load(fld(out_prefix, "post_offset"), mmap_mode="r"),
        df,
    )


def query_tokens(
    tok_path: str,
    ptr_path: str,
    lo_row: int,
    hi_row: int,
    df: np.ndarray,
    df_cap: int,
    max_per_query: int,
) -> list:
    """Select the discriminative query tokens for a block of queries.

    Returns a list (one entry per query) of ``(token, idf)`` tuples.
    """
    tok_mm = np.load(tok_path, mmap_mode="r")
    ptr_mm = np.load(ptr_path, mmap_mode="r")
    log_df = np.log1p(df.astype(np.float64))

    out: list = []
    for d in range(lo_row, hi_row):
        lo, hi = int(ptr_mm[d]), int(ptr_mm[d + 1])
        if lo == hi:
            out.append(())
            continue
        toks = np.asarray(tok_mm[lo:hi])
        slots = (toks & np.uint32(DF_MASK)).astype(np.intp)
        freqs = df[slots]
        mask = (freqs > 0) & (freqs <= df_cap)
        if not mask.any():
            out.append(())
            continue
        idx = np.flatnonzero(mask)
        if idx.size > max_per_query:
            # Rarest first: maximise the chance that at least one posting list
            # is short enough to keep the candidate volume bounded.
            idx = idx[np.argsort(freqs[idx], kind="stable")[:max_per_query]]
        pairs = [(int(toks[i]), float(log_df[slots[i]]))
                 for i in idx.tolist()]
        out.append(tuple(pairs))
    return out


def combined_df(prefixes, verbose: bool = True) -> np.ndarray:
    """Sum the per-file document-frequency tables of a target pool."""
    total = np.zeros(DF_SIZE, dtype=np.int64)
    for pref in prefixes:
        path = fld(pref, "df")
        arr = np.load(path, mmap_mode="r")
        if verbose:
            print(f"    df <- {os.path.basename(pref)} "
                  f"(max {int(arr.max()):,})", flush=True)
        total += arr
    return total
