"""
Rare-token candidate generation.

Given the query tokens selected by :mod:`src.blocking.index`, this module turns
them into a ranked candidate list per Source 1 entity. Candidates are scored by
the IDF-weighted number of shared rare tokens, which behaves like a tf-idf
cosine on token sets, and the top ``top_k`` survive.

Everything is streamed in query blocks so peak memory stays flat regardless of
whether the split holds 1.7 M or 2.2 M queries.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.blocking import compkeys
from src.blocking import index as bindex
from src.data.loader import fld

# Weight of an address-token hit relative to a name-token hit. Address evidence
# is slightly discounted: two businesses on one street share a postal code, so
# address agreement alone is weaker than a distinctive business-name token.
ADDR_TOKEN_WEIGHT = 0.8

QUERY_BLOCK = 5_000


class RareTokenBlocker:
    """Builds the postings index and answers block-wise candidate queries."""

    def __init__(
        self,
        cache_root: str,
        split: str,
        df_cap_name: int = 250,
        df_cap_addr: int = 250,
        top_k: int = 30,
        max_tok_per_doc: int = bindex.MAX_TOK_PER_DOC,
        max_tok_per_doc_addr: int = bindex.MAX_TOK_PER_DOC_ADDR,
        query_tok_name: int = bindex.QUERY_TOK_NAME,
        query_tok_addr: int = bindex.QUERY_TOK_ADDR,
        comp_df_cap: int = 3000,
    ):
        self.cache_root = cache_root
        self.split = split
        self.df_cap_name = df_cap_name
        self.df_cap_addr = df_cap_addr
        self.top_k = top_k
        self.max_tok_per_doc = max_tok_per_doc
        self.max_tok_per_doc_addr = max_tok_per_doc_addr
        self.query_tok_name = query_tok_name
        self.query_tok_addr = query_tok_addr
        self.comp_df_cap = comp_df_cap

        self.s2 = os.path.join(cache_root, split, "s2")
        self.s3 = os.path.join(cache_root, split, "s3")
        self.s1 = os.path.join(cache_root, split, "s1")
        self.idx_dir = os.path.join(cache_root, split, "idx")
        self.n2 = 0
        self.n3 = 0
        self.df_name: Optional[np.ndarray] = None
        self.df_addr: Optional[np.ndarray] = None
        self.p_name: List[bindex.TokenPostings] = []
        self.p_addr: List[bindex.TokenPostings] = []
        self.c_name: List[compkeys.CompositeIndex] = []
        self.c_addr: List[compkeys.CompositeIndex] = []

    # -- construction ----------------------------------------------------
    def build(self, force: bool = False, verbose: bool = True) -> "RareTokenBlocker":
        """Build (or load) the four postings tables for the target pool."""
        os.makedirs(self.idx_dir, exist_ok=True)
        self.n2 = len(np.load(fld(self.s2, "idh"), mmap_mode="r"))
        self.n3 = len(np.load(fld(self.s3, "idh"), mmap_mode="r"))

        if verbose:
            print(f"  [{self.split}] target pool: S2={self.n2:,} S3={self.n3:,} "
                  f"(total {self.n2 + self.n3:,})", flush=True)

        # Document frequencies must be global over S2 + S3, otherwise a token
        # common in one source looks rare when the other source is added later.
        df_n = bindex.combined_df([self.s2, self.s3], verbose=verbose)
        df_a = bindex.combined_df([self.s2, self.s3], verbose=False)
        self.df_name, self.df_addr = df_n, df_a

        self.p_name = []
        self.p_addr = []
        for tag, src in (("s2", self.s2), ("s3", self.s3)):
            self.p_name.append(self._build_one(
                f"{tag}_name", fld(src, "ntok"), fld(src, "nptr"), df_n,
                self.max_tok_per_doc, self.df_cap_name, force, verbose))
            self.p_addr.append(self._build_one(
                f"{tag}_addr", fld(src, "atok"), fld(src, "aptr"), df_a,
                self.max_tok_per_doc_addr, self.df_cap_addr, force, verbose))

        # The composite-key channel carries most of the recall; the rare-token
        # channel only adds typo tolerance on top of it.
        self.c_name = [self._build_comp(tag, src, force, verbose)
                       for tag, src in (("s2", self.s2), ("s3", self.s3))]
        self.c_addr = self.c_name
        return self

    def _build_comp(self, tag: str, src: str, force: bool,
                    verbose: bool) -> compkeys.CompositeIndex:
        prefix = os.path.join(self.idx_dir, f"{tag}_comp")
        done = fld(prefix, "ck_offset")
        if not force and os.path.exists(done):
            if verbose:
                print(f"    [{tag}_comp] loaded from cache", flush=True)
            return compkeys.CompositeIndex(
                np.load(fld(prefix, "ck_tok"), mmap_mode="r"),
                np.load(fld(prefix, "ck_row"), mmap_mode="r"),
                np.load(done, mmap_mode="r"))
        if verbose:
            print(f"    [{tag}_comp] building composite-key index", flush=True)
        return compkeys.build_composite_index(
            src, prefix, 0 if tag == "s2" else self.n2,
            self.comp_df_cap, verbose)

    def _build_one(self, name: str, tok_path: str, ptr_path: str,
                   df: np.ndarray, max_per_doc: int, df_cap: int,
                   force: bool, verbose: bool) -> bindex.TokenPostings:
        prefix = os.path.join(self.idx_dir, name)
        done = fld(prefix, "post_offset")
        if not force and os.path.exists(done):
            if verbose:
                print(f"    [{name}] loaded from cache", flush=True)
            return bindex.TokenPostings(
                np.load(fld(prefix, "post_tok"), mmap_mode="r"),
                np.load(fld(prefix, "post_row"), mmap_mode="r"),
                np.load(done, mmap_mode="r"), df)
        if verbose:
            print(f"    [{name}] building postings "
                  f"(df_cap={df_cap}, max/doc={max_per_doc})", flush=True)
        return bindex.build_postings(
            tok_path, ptr_path, df, prefix, max_per_doc, df_cap, verbose)

    # -- querying --------------------------------------------------------
    def block_range(self, lo: int, hi: int, verbose: bool = False
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Candidates for queries ``[lo, hi)``.

        Returns ``(q_local, tgt_global_row, score)`` sorted by descending score
        within each query and truncated to ``top_k``.
        """
        qn = bindex.query_tokens(
            fld(self.s1, "ntok"), fld(self.s1, "nptr"), lo, hi,
            self.df_name, self.df_cap_name, self.query_tok_name)
        qa = bindex.query_tokens(
            fld(self.s1, "atok"), fld(self.s1, "aptr"), lo, hi,
            self.df_addr, self.df_cap_addr, self.query_tok_addr)
        qc = compkeys.query_keys(self.s1, lo, hi)

        rows_l: list = []
        cols_l: list = []
        val_l: list = []

        # (per-query key lists, matching posting tables, idf multiplier). Keeping
        # the namespace explicit avoids inferring it from a weight value.
        sources = (
            (qn, self.p_name, 1.0),
            (qa, self.p_addr, ADDR_TOKEN_WEIGHT),
            (qc, self.c_name, 1.0),
        )

        for i in range(hi - lo):
            for q_keys, postings, weight in sources:
                for entry in q_keys[i]:
                    for src_i, post in enumerate(postings):
                        off = 0 if src_i == 0 else self.n2
                        if isinstance(post, compkeys.CompositeIndex):
                            hits = post.get(entry[0])
                            if hits.size:
                                rows_l.append(np.full(hits.size, i, np.int32))
                                cols_l.append(hits.astype(np.int32))
                                val_l.append(np.full(
                                    hits.size, np.float32(entry[1]),
                                    np.float32))
                            continue
                        token, idf = entry
                        w = np.float32(idf * weight)
                        start, stop = post.lookup(token)
                        if stop <= start:
                            continue
                        toks = post.tok[start:stop]
                        mask = toks == np.uint32(token)
                        if not mask.any():
                            continue
                        hits = np.asarray(post.row[start:stop])[mask]
                        rows_l.append(np.full(hits.size, i, dtype=np.int32))
                        cols_l.append((hits + off).astype(np.int32))
                        val_l.append(np.full(hits.size, w, dtype=np.float32))

        if not rows_l:
            return (np.empty(0, np.int32), np.empty(0, np.int32),
                    np.empty(0, np.float32))

        rows = np.concatenate(rows_l)
        cols = np.concatenate(cols_l)
        vals = np.concatenate(val_l)
        del rows_l, cols_l, val_l

        rows, cols, vals = _top_k_per_query(
            rows, cols, vals, self.n2 + self.n3, self.top_k)
        if verbose:
            print(f"    queries {lo:,}-{hi:,}: {len(rows):,} candidate pairs, "
                  f"{len(rows) / max(1, hi - lo):.1f} per query", flush=True)
        return rows, cols, vals

    def block_all(self, n_queries: int, progress_every: int = 20,
                  verbose: bool = True):
        """Yield ``(q_global, tgt_row, score)`` for every query in the split."""
        t0_blocks = 0
        for lo in range(0, n_queries, QUERY_BLOCK):
            hi = min(lo + QUERY_BLOCK, n_queries)
            r, c, v = self.block_range(lo, hi)
            if len(r):
                yield r.astype(np.int32) + lo, c, v
            t0_blocks += 1
            if verbose and t0_blocks % progress_every == 0:
                print(f"    blocked {hi:,}/{n_queries:,} queries "
                  f"({hi / n_queries:.1%})", flush=True)


def _top_k_per_query(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray,
                     n_targets: int, k: int
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse duplicate (query, target) pairs and keep the best ``k`` per query.

    Fully vectorised: duplicates are merged through a single ``bincount`` over a
    factored integer key, then one ``lexsort`` groups everything by query with
    the highest score first, and a running rank mask truncates to ``k``.
    """
    key = rows.astype(np.int64) * n_targets + cols.astype(np.int64)
    uniq, inverse = np.unique(key, return_inverse=True)
    summed = np.bincount(inverse, weights=vals.astype(np.float64),
                         minlength=len(uniq)).astype(np.float32)
    del key, inverse

    u_rows = (uniq // n_targets).astype(np.int32)
    u_cols = (uniq % n_targets).astype(np.int32)
    del uniq

    order = np.lexsort((-summed, u_rows))
    s_rows = u_rows[order]
    s_cols = u_cols[order]
    s_val = summed[order]
    del order, u_rows, u_cols, summed

    # Rank of each entry inside its query group.
    new_group = np.empty(len(s_rows), dtype=bool)
    new_group[0] = True
    np.not_equal(s_rows[1:], s_rows[:-1], out=new_group[1:])
    group_id = np.cumsum(new_group) - 1
    group_start = np.flatnonzero(new_group)
    rank = np.arange(len(s_rows), dtype=np.int64) - group_start[group_id]

    keep = rank < k
    return s_rows[keep], s_cols[keep], s_val[keep]
