"""Encode the country column for every source file (background-friendly)."""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import countries

DATA_ROOT = r"S:\Amazon_ML_Challenge\dataset\student_resource\dataset"
CACHE_ROOT = r"S:\Amazon_ML_Challenge\cache"

FILES = [
    ("train", "train_source1.tsv", "s1"),
    ("train", "train_source2.tsv", "s2"),
    ("train", "train_source3.tsv", "s3"),
    ("test", "test_source1.tsv", "s1"),
    ("test", "test_source2.tsv", "s2"),
    ("test", "test_source3.tsv", "s3"),
]


def job(args):
    split, fname, tag, data_root, cache_root = args
    prefix = os.path.join(cache_root, split, tag)
    t0 = time.time()
    codes = countries.encode_split(
        os.path.join(data_root, split, fname), prefix, verbose=True)
    return f"DONE {split}/{fname}: {len(codes)} labels in {time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--cache-root", default=CACHE_ROOT)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    jobs = [(s, f, t, args.data_root, args.cache_root) for s, f, t in FILES]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for msg in ex.map(job, jobs):
            print(msg, flush=True)
    print("COUNTRIES DONE")


if __name__ == "__main__":
    main()
