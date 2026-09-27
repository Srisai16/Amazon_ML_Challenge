"""
Build the on-disk NumPy cache for one or both dataset splits.

Usage::

    python scripts/build_cache.py --split train
    python scripts/build_cache.py --split test
    python scripts/build_cache.py --split all --workers 3

The heavy source files are encoded in parallel worker processes. Each worker
holds only a 64 MB document-frequency table plus small row buffers, so a handful
of workers fit comfortably in the available RAM.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data import loader

DATA_ROOT = r"S:\Amazon_ML_Challenge\dataset\student_resource\dataset"
CACHE_ROOT = r"S:\Amazon_ML_Challenge\cache"

SOURCE_FILES = {
    "train": ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"],
    "test": ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"],
}
TAGS = {"source1.tsv": "s1", "source2.tsv": "s2", "source3.tsv": "s3"}


def tag_for(fname: str) -> str:
    """'train_source2.tsv' -> 's2'. Matches on the suffix, not the full name."""
    for suffix, tag in TAGS.items():
        if fname.endswith(suffix):
            return tag
    raise KeyError(f"cannot derive a source tag from {fname!r}")


def prefix_for(split: str, tag: str, cache_root: str) -> str:
    return os.path.join(cache_root, split, tag)


def is_encoded(prefix: str, n_rows: int) -> bool:
    """True when a complete cache already exists for this prefix."""
    for field in ("id", "idh", "name", "addr", "ntok", "nptr", "atok", "aptr", "df"):
        path = loader.fld(prefix, field)
        if not os.path.exists(path):
            return False
    try:
        return len(np.load(loader.fld(prefix, "idh"), mmap_mode="r")) == n_rows
    except Exception:
        return False


def encode_one(args):
    """Worker entry point: encode a single source file."""
    split, fname, data_root, cache_root = args
    tag = tag_for(fname)
    prefix = prefix_for(split, tag, cache_root)
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    src = os.path.join(data_root, split, fname)

    n_rows = loader.count_rows(src)
    if is_encoded(prefix, n_rows):
        return f"SKIP {split}/{fname} ({n_rows:,} rows already cached)"

    t0 = time.time()
    stats = loader.encode_source(src, prefix, n_rows=n_rows, verbose=False)
    return (f"DONE {split}/{fname}: {stats['rows']:,} rows, "
            f"{stats['tokens']:,} tokens in {time.time() - t0:.0f}s")


def build_ground_truth(data_root: str, cache_root: str, split: str = "train"):
    """Encode the training labels. Requires source1/2/3 of that split."""
    gt_src = os.path.join(data_root, "train", "train_ground_truth.tsv")
    gt_prefix = os.path.join(cache_root, "train", "gt")
    if not os.path.exists(loader.fld(gt_prefix, "pos_keys")):
        return loader.encode_ground_truth(
            gt_src,
            prefix_for("train", "s1", cache_root),
            (prefix_for("train", "s2", cache_root),
             prefix_for("train", "s3", cache_root)),
            gt_prefix,
        )
    keys = np.load(loader.fld(gt_prefix, "pos_keys"), mmap_mode="r")
    counts = np.load(loader.fld(gt_prefix, "pos_counts"), mmap_mode="r")
    return {"pairs": len(keys), "singletons": int((counts == 0).sum()),
            "unresolved": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", choices=["train", "test", "all"])
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--cache-root", default=CACHE_ROOT)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    splits = ["train", "test"] if args.split == "all" else [args.split]
    t_start = time.time()

    for split in splits:
        jobs = [(split, f, args.data_root, args.cache_root)
                for f in SOURCE_FILES[split]]
        # Source 1 first: the ground-truth encoder joins against it.
        jobs.sort(key=lambda j: 0 if tag_for(j[1]) == "s1" else 1)
        print(f"[{split}] encoding {len(jobs)} source files "
              f"with {args.workers} workers", flush=True)
        if args.workers <= 1:
            for j in jobs:
                print("  " + encode_one(j), flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                for msg in ex.map(encode_one, jobs):
                    print("  " + msg, flush=True)

    if "train" in splits:
        print("[train] encoding ground truth", flush=True)
        stats = build_ground_truth(args.data_root, args.cache_root)
        print(f"  ground truth: {stats}", flush=True)

    print(f"\nALL CACHE BUILT in {time.time() - t_start:.0f}s -> {args.cache_root}")


if __name__ == "__main__":
    main()
