"""
Fast sanity check of the composite-key design.

Builds an in-memory key -> rows map over a *subsample* of the target pool and
measures how many of a query's true matches that are present in the subsample
are retrieved. Much quicker than a full index build, so key design can be
iterated on before paying for the real thing.
"""

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.blocking.compkeys import doc_keys
from src.data.loader import fld

CACHE = r"S:\Amazon_ML_Challenge\cache"
SPLIT = "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-target", type=int, default=120_000)
    ap.add_argument("--n-query", type=int, default=4_000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    s2 = os.path.join(CACHE, SPLIT, "s2")
    s3 = os.path.join(CACHE, SPLIT, "s3")
    s1 = os.path.join(CACHE, SPLIT, "s1")
    gt = os.path.join(CACHE, SPLIT, "gt")

    n2 = len(np.load(fld(s2, "idh"), mmap_mode="r"))
    n3 = len(np.load(fld(s3, "idh"), mmap_mode="r"))
    n_targets = n2 + n3
    n_s1 = len(np.load(fld(s1, "idh"), mmap_mode="r"))
    pos_keys = np.load(fld(gt, "pos_keys"))

    rng = np.random.default_rng(args.seed)

    # --- target subsample: a contiguous head keeps the memmap reads cheap ---
    per = args.n_target // 2
    tgt_rows = np.concatenate([
        np.arange(0, per),
        n2 + np.arange(0, per),
    ])
    in_subsample = np.zeros(n_targets, dtype=bool)
    in_subsample[tgt_rows] = True

    keymap = defaultdict(list)
    names = {"s2": np.load(fld(s2, "name"), mmap_mode="r"),
             "s3": np.load(fld(s3, "name"), mmap_mode="r")}
    addrs = {"s2": np.load(fld(s2, "addr"), mmap_mode="r"),
             "s3": np.load(fld(s3, "addr"), mmap_mode="r")}
    n_nonzero = 0
    for tag, base, off in (("s2", s2, 0), ("s3", s3, n2)):
        nm = names[tag]
        ad = addrs[tag]
        local = tgt_rows[tgt_rows >= off][:per] - off
        for r in local.tolist():
            ks = doc_keys(nm[r].decode("utf-8", "ignore"),
                          ad[r].decode("utf-8", "ignore"))
            if ks:
                n_nonzero += 1
            for k in ks:
                keymap[k].append(r + off)
    print(f"target subsample: {len(tgt_rows):,} rows, "
          f"{n_nonzero:,} produced >=1 key, {len(keymap):,} distinct keys")
    avg_post = np.mean([len(v) for v in keymap.values()])
    print(f"avg posting list length: {avg_post:.1f}, "
          f"max: {max(len(v) for v in keymap.values())}")

    # --- queries ----------------------------------------------------------
    query_rows = np.sort(rng.choice(n_s1, size=args.n_query, replace=False))
    lo_k, hi_k = int(query_rows[0]) * n_targets, int(query_rows[-1]) * n_targets
    sl = pos_keys[np.searchsorted(pos_keys, lo_k, "left"):
                  np.searchsorted(pos_keys, hi_k, "right")]
    sr = (sl // n_targets).astype(np.int64)
    sel = np.isin(sr, query_rows)
    sl = sl[sel]
    sr = sr[sel]
    sc = (sl % n_targets).astype(np.int64)

    truth = defaultdict(set)
    for r, c in zip(sr.tolist(), sc.tolist()):
        if in_subsample[c]:
            truth[r].add(c)
    total_in = sum(len(v) for v in truth.values())
    print(f"queries: {len(query_rows):,}; true pairs present in subsample: "
          f"{total_in:,} (from {len(truth):,} queries)")

    nm1 = np.load(fld(s1, "name"), mmap_mode="r")
    ad1 = np.load(fld(s1, "addr"), mmap_mode="r")
    hit = 0
    n_cand = 0
    per_q = []
    for r in query_rows.tolist():
        ks = doc_keys(nm1[r].decode("utf-8", "ignore"),
                      ad1[r].decode("utf-8", "ignore"))
        cands = set()
        for k in ks:
            cands.update(keymap.get(k, ()))
        n_cand += len(cands)
        t = truth.get(r)
        if t:
            h = len(cands & t)
            hit += h
            per_q.append(h / len(t))
    per_q = np.array(per_q) if per_q else np.zeros(1)
    print()
    print("=" * 60)
    print(f"  COMPOSITE-KEY RECALL (subsample) : {hit / max(1, total_in):.4f}")
    print(f"  mean per-query recall             : {per_q.mean():.4f}")
    print(f"  full-recall queries               : "
          f"{(per_q == 1.0).mean():.1%}")
    print(f"  avg candidates per query          : "
          f"{n_cand / len(query_rows):.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
