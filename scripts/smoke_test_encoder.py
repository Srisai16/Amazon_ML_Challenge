"""Smoke test: encode a small slice of the real data and measure throughput."""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import loader, fastnorm as fn

B = r"S:\Amazon_ML_Challenge\dataset\student_resource\dataset"
TMP = r"C:\Users\srisai\AppData\Local\Temp\opencode\enc_test"
os.makedirs(TMP, exist_ok=True)

SAMPLE = 60_000


def make_sample(src, dst, n):
    """Write the first n data rows of a source file to a temp TSV."""
    with open(os.path.join(B, src), encoding="utf-8", errors="replace", newline="") as fh:
        header = fh.readline()
        with open(dst, "w", encoding="utf-8", newline="") as out:
            out.write(header)
            for i, line in enumerate(fh):
                if i >= n:
                    break
                out.write(line)
    return dst


print("=== 1. normalizer sanity ===")
for raw in ["Marina Ecole France Sarl", "  ORELEE'S BARBERSHOP, Inc. ",
            "Société Générale", "Zephay Labs Inc"]:
    print(f"  {raw!r}")
    print(f"     name      -> {fn.normalize_name(raw)!r}")
    print(f"     name_core -> {fn.normalize_name_core(raw)!r}")
    print(f"     addr      -> {fn.normalize_addr('63 R. DE DIEPPE, LILLE')!r}")

print("\n=== 2. hash consistency (scalar vs vectorized) ===")
ids = ["S1-925783039", "S2-166376419", "S3-202863386", "S1-00001"]
arr = np.array([i.encode() for i in ids], dtype="S12")
vec = loader.hash_fixed_bytes(arr)
sca = np.array([loader._hash_id_str(i) for i in ids], dtype=np.int64)
print("  vectorized:", vec)
print("  scalar    :", sca)
assert np.array_equal(vec, sca), "HASH MISMATCH between vectorized and scalar paths"
print("  OK - both paths agree")

print("\n=== 3. encode a 60k-row sample ===")
s1 = make_sample("train\\train_source1.tsv", os.path.join(TMP, "s1.tsv"), SAMPLE)
s2 = make_sample("train\\train_source2.tsv", os.path.join(TMP, "s2.tsv"), SAMPLE)
p1 = os.path.join(TMP, "e_s1")
p2 = os.path.join(TMP, "e_s2")

st1 = loader.encode_source(s1, p1)
st2 = loader.encode_source(s2, p2)
rate = (st1["rows"] + st2["rows"]) / (st1["seconds"] + st2["seconds"])
print(f"\n  throughput: {rate:,.0f} rows/sec  -> 11.7M rows would take "
      f"{11_700_000 / rate / 60:.1f} min")

print("\n=== 4. inspect the encoded output ===")
nm = np.load(loader.fld(p1, "name"), mmap_mode="r")
ad = np.load(loader.fld(p1, "addr"), mmap_mode="r")
ntok = np.load(loader.fld(p1, "ntok"))
nptr = np.load(loader.fld(p1, "nptr"))
atok = np.load(loader.fld(p1, "atok"))
df = np.load(loader.fld(p1, "df"))
for i in range(5):
    print(f"  row {i}: name={nm[i]!r} addr={ad[i]!r}")
    print(f"          ntok={list(ntok[nptr[i]:nptr[i + 1]])} "
          f"atok={list(atok[nptr[i]:nptr[i + 1]])}")
print(f"  distinct df slots hit: {int((df > 0).sum()):,}")
print(f"  max df: {int(df.max()):,}   median nonzero df: "
      f"{int(np.median(df[df > 0])) if (df > 0).any() else 0:,}")

print("\n=== 5. ground truth encode on the sample ===")
# Build a tiny GT restricted to sampled S1 ids.
# NOTE: str() on a numpy bytes_ scalar yields "b'...'", so decode explicitly.
_id_arr = np.load(loader.fld(p1, "id"), mmap_mode="r")
sampled = {_id_arr[i].decode("utf-8") for i in range(SAMPLE)}
gt_src = os.path.join(B, "train\\train_ground_truth.tsv")
gt_small = os.path.join(TMP, "gt.tsv")
kept = 0
# The GT file is not in the same order as source1, so the whole file has to be
# scanned to find the rows belonging to the sampled S1 entities.
with open(gt_src, encoding="utf-8", errors="replace", newline="") as fh, \
     open(gt_small, "w", encoding="utf-8", newline="") as out:
    out.write(fh.readline())
    for line in fh:
        s1id = line.split("\t", 1)[0].strip()
        if s1id in sampled:
            out.write(line)
            kept += 1
        if kept >= 20000:
            break
print(f"  kept {kept} ground-truth rows for the sample")
gst = loader.encode_ground_truth(gt_small, p1, (p2,), os.path.join(TMP, "e_gt"))
print(f"  result: {gst}")

keys = np.load(loader.fld(os.path.join(TMP, "e_gt"), "pos_keys"))
counts = np.load(loader.fld(os.path.join(TMP, "e_gt"), "pos_counts"))
n_t = st2["rows"]
print(f"  pos_keys sample: {keys[:8]}")
print(f"  decoded (s1_row, tgt_row): "
      f"{[(int(k) // n_t, int(k) % n_t) for k in keys[:8]]}")
print(f"  counts sum = {counts.sum():,}  pos_keys len = {len(keys):,}  "
      f"match = {counts.sum() == len(keys)}")
assert counts.sum() == len(keys)
assert len(counts) == st1["rows"]

# Guards against silently losing the join: nearly every S1 id and nearly every
# referenced target id must resolve against the encoded tables.
n_gt = 20000
assert gst["unresolved"] < n_gt * 0.01, (
    f"{gst['unresolved']:,}/{n_gt:,} S1 ids failed to resolve - the id hash "
    f"join is broken (unsorted searchsorted array?)")
assert gst["pairs"] > 0, "no positive pairs resolved - the target id join is broken"
names_t = np.load(loader.fld(p1, "name"), mmap_mode="r")
addrs_t = np.load(loader.fld(p1, "addr"), mmap_mode="r")
nm2 = np.load(loader.fld(p2, "name"), mmap_mode="r")
ad2 = np.load(loader.fld(p2, "addr"), mmap_mode="r")
print(f"  S1 join: {n_gt - gst['unresolved']:,}/{n_gt:,} resolved  |  "
      f"pairs: {gst['pairs']:,}  non-singleton S1: {int((counts > 0).sum()):,}")

# Spot-check one resolved pair against the raw source rows it points at.
k = int(keys[0])
r1, rt = k // n_t, k % n_t
print(f"  spot check pair: S1 row {r1} -> target row {rt}")
print(f"    S1     : {names_t[r1]!r} @ {addrs_t[r1]!r}")
print(f"    target : {nm2[rt]!r} @ {ad2[rt]!r}")
print("\nALL SMOKE TESTS PASSED")
