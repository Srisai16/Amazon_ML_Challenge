import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.blocking.blocker import RareTokenBlocker
from src.data.loader import fld

CACHE = r"S:\Amazon_ML_Challenge\cache"
SPLIT = "train"

def main():
    s2 = os.path.join(CACHE, SPLIT, "s2")
    s3 = os.path.join(CACHE, SPLIT, "s3")
    s1 = os.path.join(CACHE, SPLIT, "s1")
    gt = os.path.join(CACHE, SPLIT, "gt")

    n2 = len(np.load(fld(s2, "idh"), mmap_mode="r"))
    n3 = len(np.load(fld(s3, "idh"), mmap_mode="r"))
    n_targets = n2 + n3
    n_s1 = len(np.load(fld(s1, "idh"), mmap_mode="r"))
    pos_keys = np.load(fld(gt, "pos_keys"))

    lo, hi = 10000, 11000
    lo_key, hi_key = lo * n_targets, hi * n_targets
    sub = pos_keys[np.searchsorted(pos_keys, lo_key, "left"):np.searchsorted(pos_keys, hi_key, "right")]
    sub_r = (sub // n_targets).astype(np.int64)
    sub_c = (sub % n_targets).astype(np.int64)
    truth = {}
    for r, c in zip(sub_r.tolist(), sub_c.tolist()):
        truth.setdefault(r, set()).add(c)

    blocker = RareTokenBlocker(CACHE, SPLIT).build(force=False, verbose=False)
    r, c, v = blocker.block_range(lo, hi)
    r_global = r + lo
    found = {}
    for rr, cc in zip(r_global.tolist(), c.tolist()):
        found.setdefault(rr, set()).add(cc)

    nm1 = np.load(fld(s1, "name"), mmap_mode="r")
    ad1 = np.load(fld(s1, "addr"), mmap_mode="r")
    nm2 = np.load(fld(s2, "name"), mmap_mode="r")
    ad2 = np.load(fld(s2, "addr"), mmap_mode="r")
    nm3 = np.load(fld(s3, "name"), mmap_mode="r")
    ad3 = np.load(fld(s3, "addr"), mmap_mode="r")

    def get_tgt(tgt_idx):
        if tgt_idx < n2:
            return "S2", nm2[tgt_idx].decode("utf-8", "ignore"), ad2[tgt_idx].decode("utf-8", "ignore")
        else:
            idx = tgt_idx - n2
            return "S3", nm3[idx].decode("utf-8", "ignore"), ad3[idx].decode("utf-8", "ignore")

    missed_count = 0
    print("=== MISSED TRUE PAIRS SAMPLE ===")
    for q_idx, true_set in truth.items():
        retrieved = found.get(q_idx, set())
        missed = true_set - retrieved
        if missed:
            s1_name = nm1[q_idx].decode("utf-8", "ignore")
            s1_addr = ad1[q_idx].decode("utf-8", "ignore")
            for m in list(missed)[:2]:
                src, t_name, t_addr = get_tgt(m)
                print(f"Query [{q_idx}]: {s1_name!r} | {s1_addr!r}")
                print(f"  Missed {src} [{m}]: {t_name!r} | {t_addr!r}")
                print("-" * 60)
                missed_count += 1
                if missed_count >= 15:
                    return

if __name__ == "__main__":
    main()
