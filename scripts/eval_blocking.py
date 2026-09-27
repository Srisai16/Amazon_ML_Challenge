"""
Measure blocking recall (pair completeness) and candidate volume on train.

This is the ceiling diagnostic: no amount of reranking can recover a true match
that blocking never proposed. Runs on a random sample of Source 1 queries
against the *full* training target pool, so the recall estimate reflects
production conditions.

Usage::

    python scripts/eval_blocking.py --n 20000 --top-k 30
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.blocking.blocker import RareTokenBlocker
from src.data.loader import fld

CACHE_ROOT = r"S:\Amazon_ML_Challenge\cache"
SPLIT = "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000,
                    help="number of Source 1 queries to sample")
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--df-cap-name", type=int, default=250)
    ap.add_argument("--df-cap-addr", type=int, default=250)
    ap.add_argument("--max-tok-per-doc", type=int, default=4)
    ap.add_argument("--max-tok-per-doc-addr", type=int, default=4)
    ap.add_argument("--query-tok-name", type=int, default=5)
    ap.add_argument("--query-tok-addr", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-build", action="store_true",
                    help="reuse an existing postings index")
    args = ap.parse_args()

    n_s1 = len(np.load(fld(os.path.join(CACHE_ROOT, SPLIT, "s1"), "idh"),
                       mmap_mode="r"))
    pos_keys = np.load(fld(os.path.join(CACHE_ROOT, SPLIT, "gt"), "pos_keys"))
    n_targets = (len(np.load(fld(os.path.join(CACHE_ROOT, SPLIT, "s2"), "idh"),
                            mmap_mode="r"))
                 + len(np.load(fld(os.path.join(CACHE_ROOT, SPLIT, "s3"), "idh"),
                               mmap_mode="r")))

    rng = np.random.default_rng(args.seed)
    n_sample = min(args.n, n_s1)
    # Take a contiguous block of queries for fast sequential cache reading
    start_idx = int(rng.integers(0, max(1, n_s1 - n_sample)))
    sample = np.arange(start_idx, start_idx + n_sample)
    print(f"Source 1 rows: {n_s1:,} | target pool: {n_targets:,} | "
          f"sampled queries: {len(sample):,} (span [{start_idx:,}, {start_idx + n_sample:,}))")

    # --- recall of the positive pairs belonging to the sampled queries ----
    lo_key = int(sample[0]) * n_targets
    hi_key = int(sample[-1]) * n_targets
    a = np.searchsorted(pos_keys, lo_key, "left")
    b = np.searchsorted(pos_keys, hi_key, "right")
    sub = pos_keys[a:b]
    sub_rows = (sub // n_targets).astype(np.int64)
    keep = np.isin(sub_rows, sample)
    sub = sub[keep]
    sub_rows = sub_rows[keep]
    sub_cols = (sub % n_targets).astype(np.int64)
    truth = {}
    for r, c in zip(sub_rows.tolist(), sub_cols.tolist()):
        truth.setdefault(r, set()).add(c)
    total_true = sum(len(v) for v in truth.values())
    n_singletons = len(sample) - len(truth)
    print(f"true pairs for sampled queries: {total_true:,} "
          f"(from {len(truth):,} non-singleton queries, "
          f"{n_singletons:,} singletons)")

    # --- build / load the index -------------------------------------------
    t0 = time.time()
    blocker = RareTokenBlocker(
        CACHE_ROOT, SPLIT,
        df_cap_name=args.df_cap_name, df_cap_addr=args.df_cap_addr,
        top_k=args.top_k, max_tok_per_doc=args.max_tok_per_doc,
        max_tok_per_doc_addr=args.max_tok_per_doc_addr,
        query_tok_name=args.query_tok_name,
        query_tok_addr=args.query_tok_addr,
    ).build(force=not args.no_build)
    print(f"index ready in {time.time() - t0:.0f}s  "
          f"(target pool {blocker.n2 + blocker.n3:,})")

    # --- block queries in contiguous blocks ---
    t0 = time.time()
    found: dict = {}
    n_cand = 0
    step = 2500
    
    # Process the sampled contiguous slice
    for lo in range(int(sample[0]), int(sample[-1]) + 1, step):
        hi = min(lo + step, int(sample[-1]) + 1)
        r, c, v = blocker.block_range(lo, hi)
        if len(r) == 0:
            continue
        r_global = r + lo
        sel = np.isin(r_global, sample)
        r_sel = r_global[sel]
        c_sel = c[sel]
        n_cand += len(r_sel)
        for rr, cc in zip(r_sel.tolist(), c_sel.tolist()):
            found.setdefault(rr, set()).add(cc)
    elapsed = time.time() - t0
    print(f"blocked {len(sample):,} queries in {elapsed:.0f}s "
          f"({len(sample) / max(elapsed, 1e-9):,.0f} q/s)")

    # --- score -------------------------------------------------------------
    recalled = 0
    per_q_recall = []
    for r, true_cols in truth.items():
        hit = len(found.get(r, set()) & true_cols)
        recalled += hit
        per_q_recall.append(hit / len(true_cols))
    per_q_recall = np.array(per_q_recall) if per_q_recall else np.zeros(1)

    recall = recalled / total_true if total_true else 0.0
    avg_cand = n_cand / len(sample)
    print()
    print("=" * 62)
    print(f"  PAIR COMPLETENESS (recall ceiling) : {recall:.4f}")
    print(f"  mean per-query recall              : {per_q_recall.mean():.4f}")
    print(f"  queries with 0 of their matches    : "
          f"{int((per_q_recall == 0).sum()):,} / {len(truth):,}")
    print(f"  queries with full recall           : "
          f"{int((per_q_recall == 1.0).sum()):,} / {len(truth):,} "
          f"({(per_q_recall == 1.0).mean():.1%})")
    print(f"  avg candidates per query           : {avg_cand:.2f}")
    print(f"  total candidate pairs              : {n_cand:,}")
    print(f"  reduction ratio vs full cross-prod : "
          f"{1.0 - n_cand / (len(sample) * n_targets):.8f}")
    print("=" * 62)


if __name__ == "__main__":
    main()
