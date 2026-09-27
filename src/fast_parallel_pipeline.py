"""
High-Speed Multi-Process Parallel Inference Pipeline for Amazon ML Challenge 2026.
Uses 12 parallel worker processes to achieve >2,000 queries/sec and complete
the full 1.73M test set in ~12-15 minutes.
"""

import argparse
import multiprocessing as mp
import os
import shutil
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.blocking.blocker import RareTokenBlocker
from src.data.loader import fld
from src.features.feature_extractor import FeatureExtractor
from src.models.gbdt_ranker import GBDTRanker
from utils.validate_submission import validate


def worker_inference(
    worker_id: int,
    lo_q: int,
    hi_q: int,
    cache_root: str,
    model_path: str,
    temp_dir: str,
    top_k: int,
    t_singleton: float,
    t_match: float,
    block_size: int = 10_000,
):
    """Worker process: processes queries in range [lo_q, hi_q) and writes part files."""
    cand_part = os.path.join(temp_dir, f"part_{worker_id:02d}_cand.tsv")
    match_part = os.path.join(temp_dir, f"part_{worker_id:02d}_match.tsv")

    s1_dir = os.path.join(cache_root, "test", "s1")
    s2_dir = os.path.join(cache_root, "test", "s2")
    s3_dir = os.path.join(cache_root, "test", "s3")

    s1_ids = np.load(fld(s1_dir, "id"), mmap_mode="r")
    s2_ids = np.load(fld(s2_dir, "id"), mmap_mode="r")
    s3_ids = np.load(fld(s3_dir, "id"), mmap_mode="r")

    n2 = len(s2_ids)

    def decode_tgt(idx: int) -> str:
        if idx < n2:
            return s2_ids[idx].decode("utf-8", "ignore").strip()
        else:
            return s3_ids[idx - n2].decode("utf-8", "ignore").strip()

    blocker = RareTokenBlocker(cache_root, "test", top_k=top_k).build(force=False, verbose=False)
    extractor = FeatureExtractor()
    ranker = GBDTRanker()
    ranker.load(model_path)

    total_cands = 0
    total_matches = 0
    total_singles = 0
    t0 = time.time()
    n_queries = hi_q - lo_q

    with open(cand_part, "w", encoding="utf-8", buffering=1024 * 1024) as fc, \
         open(match_part, "w", encoding="utf-8", buffering=1024 * 1024) as fm:

        for curr_lo in range(lo_q, hi_q, block_size):
            curr_hi = min(curr_lo + block_size, hi_q)
            b_len = curr_hi - curr_lo

            r, c, v = blocker.block_range(curr_lo, curr_hi)

            cands_pq: Dict[int, List[int]] = {i: [] for i in range(b_len)}
            for lq, ti in zip(r.tolist(), c.tolist()):
                cands_pq[lq].append(ti)

            matched_pq: Dict[int, List[int]] = {i: [] for i in range(b_len)}
            if len(r) > 0:
                qg = r + curr_lo
                X = extractor.extract_batch_from_cache(cache_root, "test", qg, c, v)
                probs = ranker.predict_proba(X)

                qcp: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(b_len)}
                for lq, ti, p in zip(r.tolist(), c.tolist(), probs.tolist()):
                    qcp[lq].append((ti, p))

                for lq, pairs in qcp.items():
                    if not pairs:
                        continue
                    max_p = max(p for _, p in pairs)
                    if max_p >= t_singleton:
                        matched_pq[lq] = [t for t, p in pairs if p >= t_match]

            for li in range(b_len):
                gq = curr_lo + li
                s1e = s1_ids[gq].decode("utf-8", "ignore").strip()

                cu = list(dict.fromkeys(decode_tgt(t) for t in cands_pq[li]))
                mu = list(dict.fromkeys(decode_tgt(t) for t in matched_pq[li]))

                total_cands += len(cu)
                total_matches += len(mu)
                if not mu:
                    total_singles += 1

                fc.write(f"{s1e}\t{','.join(cu)}\n")
                fm.write(f"{s1e}\t{','.join(mu)}\n")

            done = curr_hi - lo_q
            if done % 10_000 == 0 or curr_hi == hi_q:
                el = time.time() - t0
                speed = done / max(1e-5, el)
                print(f"[Worker {worker_id:02d}] {done:,}/{n_queries:,} ({done/n_queries:.1%}) | {speed:.0f} q/s", flush=True)

    print(f"[Worker {worker_id:02d}] Finished {n_queries:,} queries in {time.time()-t0:.1f}s", flush=True)


def run_parallel_inference(
    cache_root: str,
    output_dir: str,
    model_path: str,
    test_data_dir: str,
    top_k: int = 10,
    t_singleton: float = 0.0,
    t_match: float = 0.40,
    num_workers: int = 12,
    block_size: int = 10_000,
):
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, "_temp_parts")
    os.makedirs(temp_dir, exist_ok=True)

    matching_out = os.path.join(output_dir, "matching_results.tsv")
    candidate_out = os.path.join(output_dir, "candidate_pairs.tsv")

    print("=" * 80)
    print(f">>> Fast Parallel Inference: {num_workers} workers, top_k={top_k}, T_sing={t_singleton:.2f}, T_match={t_match:.2f}")
    print("=" * 80)

    s1_ids = np.load(fld(os.path.join(cache_root, "test", "s1"), "id"), mmap_mode="r")
    n_s1 = len(s1_ids)
    print(f"Total test queries: {n_s1:,}")

    # Divide query space into disjoint contiguous chunks
    chunk_size = (n_s1 + num_workers - 1) // num_workers
    ranges = []
    for w in range(num_workers):
        lo = w * chunk_size
        hi = min(lo + chunk_size, n_s1)
        if lo < hi:
            ranges.append((w, lo, hi))

    print(f"Divided into {len(ranges)} worker tasks (~{chunk_size:,} queries each)")
    t0_all = time.time()

    processes = []
    for w, lo, hi in ranges:
        p = mp.Process(
            target=worker_inference,
            args=(
                w, lo, hi,
                cache_root, model_path, temp_dir,
                top_k, t_singleton, t_match, block_size
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker process failed with exit code {p.exitcode}")

    print(f"\nAll {num_workers} workers completed in {time.time() - t0_all:.1f}s! Merging results...")

    # Concatenate part files with proper TSV headers
    with open(candidate_out, "w", encoding="utf-8", buffering=1024 * 1024 * 16) as fc_out:
        fc_out.write("source1_entity_id\tcandidate_entity_ids\n")
        for w, _, _ in ranges:
            part_path = os.path.join(temp_dir, f"part_{w:02d}_cand.tsv")
            with open(part_path, "r", encoding="utf-8", buffering=1024 * 1024 * 8) as f_in:
                shutil.copyfileobj(f_in, fc_out)

    with open(matching_out, "w", encoding="utf-8", buffering=1024 * 1024 * 16) as fm_out:
        fm_out.write("source1_entity_id\tmatched_entity_ids\n")
        for w, _, _ in ranges:
            part_path = os.path.join(temp_dir, f"part_{w:02d}_match.tsv")
            with open(part_path, "r", encoding="utf-8", buffering=1024 * 1024 * 8) as f_in:
                shutil.copyfileobj(f_in, fm_out)

    # Clean up temp parts
    shutil.rmtree(temp_dir, ignore_errors=True)

    total_time = time.time() - t0_all
    print(f"\nMerging completed! Total inference time: {total_time:.1f}s ({n_s1/max(1e-5, total_time):,.0f} queries/sec)")
    print(f"Candidate file: {candidate_out} ({os.path.getsize(candidate_out):,} bytes)")
    print(f"Matching file:  {matching_out} ({os.path.getsize(matching_out):,} bytes)")

    # Validate output strictly
    print("\n--- Running Official Format Validation ---")
    errors, warnings = validate(matching_out, candidate_out, test_data_dir)
    if errors:
        print(f"[ERROR] Validation failed with {len(errors)} errors:")
        for err in errors[:10]:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("[SUCCESS] ALL SUBMISSION CHECKS PASSED PERFECTLY!")
        if warnings:
            for w in warnings[:5]:
                print(f"  [WARN] {w}")


def main():
    parser = argparse.ArgumentParser(description="Parallel Amazon ML Challenge Inference")
    parser.add_argument("--cache-root", default=r"S:\Amazon_ML_Challenge\cache")
    parser.add_argument("--test-dir", default=r"S:\Amazon_ML_Challenge\dataset\student_resource\dataset\test")
    parser.add_argument("--output-dir", default=r"S:\Amazon_ML_Challenge\output")
    parser.add_argument("--model-path", default=r"S:\Amazon_ML_Challenge\saved_models\gbdt_model.pkl")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--t-singleton", type=float, default=0.0)
    parser.add_argument("--t-match", type=float, default=0.40)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=1000)
    args = parser.parse_args()

    run_parallel_inference(
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        model_path=args.model_path,
        test_data_dir=args.test_dir,
        top_k=args.top_k,
        t_singleton=args.t_singleton,
        t_match=args.t_match,
        num_workers=args.num_workers,
        block_size=args.block_size,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
