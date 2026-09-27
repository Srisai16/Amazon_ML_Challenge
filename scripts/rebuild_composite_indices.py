"""Rebuild composite keys index for train and test splits to activate 93.9% recall key families."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.blocking import compkeys
from src.data.loader import fld
import numpy as np

CACHE_ROOT = r"S:\Amazon_ML_Challenge\cache"

def rebuild_comp(split: str):
    s2 = os.path.join(CACHE_ROOT, split, "s2")
    s3 = os.path.join(CACHE_ROOT, split, "s3")
    idx_dir = os.path.join(CACHE_ROOT, split, "idx")
    os.makedirs(idx_dir, exist_ok=True)

    n2 = len(np.load(fld(s2, "idh"), mmap_mode="r"))

    for tag, src, off in (("s2", s2, 0), ("s3", s3, n2)):
        prefix = os.path.join(idx_dir, f"{tag}_comp")
        print(f"[{split}] Building composite key index for {tag} (offset={off:,})...")
        t0 = time.time()
        c_idx = compkeys.build_composite_index(src, prefix, off, df_cap=3000, verbose=True)
        print(f"[{split}] {tag} composite index built in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    rebuild_comp("train")
    rebuild_comp("test")
