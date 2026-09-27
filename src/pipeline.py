"""
Master End-to-End Execution Pipeline for Amazon ML Challenge 2026.
Executes High-Recall Candidate Generation -> Vectorized Feature Extraction -> GBDT Matching -> Format Validation.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.blocking.blocker import RareTokenBlocker
from src.config import cfg
from src.data.loader import fld
from src.features.feature_extractor import FeatureExtractor
from src.models.gbdt_ranker import GBDTRanker
from utils.validate_submission import validate


def run_test_inference(
    cache_root: str,
    output_dir: str,
    model_path: str,
    test_data_dir: str,
    top_k: int = 30,
    block_size: int = 10_000,
    match_threshold: float = 0.65,
    force_build_index: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    matching_out = os.path.join(output_dir, "matching_results.tsv")
    candidate_out = os.path.join(output_dir, "candidate_pairs.tsv")

    print("=" * 80)
    print(">>> Starting Amazon ML Challenge 2026 End-to-End Inference Pipeline")
    print("=" * 80)

    # 1. Load Model
    ranker = GBDTRanker()
    if os.path.exists(model_path):
        ranker.load(model_path)
    else:
        print(f"Warning: Model not found at {model_path}. Using default threshold {match_threshold}")
        ranker.optimal_threshold = match_threshold

    threshold = match_threshold or ranker.optimal_threshold
    print(f"Using match decision threshold: {threshold:.3f}")

    # 2. Setup Test Cache Paths & Entity IDs
    s1_dir = os.path.join(cache_root, "test", "s1")
    s2_dir = os.path.join(cache_root, "test", "s2")
    s3_dir = os.path.join(cache_root, "test", "s3")

    s1_ids = np.load(fld(s1_dir, "id"), mmap_mode="r")
    s2_ids = np.load(fld(s2_dir, "id"), mmap_mode="r")
    s3_ids = np.load(fld(s3_dir, "id"), mmap_mode="r")

    n_s1 = len(s1_ids)
    n2 = len(s2_ids)
    n3 = len(s3_ids)
    n_targets = n2 + n3

    print(f"Test Set Summary: S1={n_s1:,} queries | S2={n2:,} | S3={n3:,} | Total Targets={n_targets:,}")

    # 3. Build/Load Test Blocker Index
    print("\n--- STAGE 1: Candidate Generation (Blocking Index Build) ---")
    t0_idx = time.time()
    blocker = RareTokenBlocker(cache_root, "test", top_k=top_k).build(force=force_build_index, verbose=True)
    print(f"Test blocking index ready in {time.time() - t0_idx:.1f}s")

    # 4. Stream Inference across Test Queries
    print(f"\n--- STAGE 2: Streaming Candidate Generation & Match Prediction ({n_s1:,} queries) ---")
    extractor = FeatureExtractor()

    t0_inf = time.time()
    total_cand_pairs = 0
    total_matches = 0
    total_singletons = 0

    with open(candidate_out, "w", encoding="utf-8") as f_cand, \
         open(matching_out, "w", encoding="utf-8") as f_match:

        # Write TSV Headers
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

        def decode_tgt_id(tgt_idx: int) -> str:
            if tgt_idx < n2:
                return s2_ids[tgt_idx].decode("utf-8", "ignore").strip()
            else:
                return s3_ids[tgt_idx - n2].decode("utf-8", "ignore").strip()

        for lo in range(0, n_s1, block_size):
            hi = min(lo + block_size, n_s1)
            b_len = hi - lo

            # Block current chunk
            r, c, v = blocker.block_range(lo, hi)

            # Map candidates per local query in chunk
            cands_per_query: Dict[int, List[int]] = {i: [] for i in range(b_len)}
            for local_q, tgt_idx in zip(r.tolist(), c.tolist()):
                cands_per_query[local_q].append(tgt_idx)

            # Extract features & predict for candidates in chunk
            matched_per_query: Dict[int, List[int]] = {i: [] for i in range(b_len)}
            if len(r) > 0:
                q_global = r + lo
                X_batch = extractor.extract_batch_from_cache(cache_root, "test", q_global, c, v)
                probs = ranker.predict_proba(X_batch)

                for local_q, tgt_idx, p in zip(r.tolist(), c.tolist(), probs.tolist()):
                    if p >= threshold:
                        matched_per_query[local_q].append(tgt_idx)

            # Write results for each query in this block
            for local_i in range(b_len):
                global_q = lo + local_i
                s1_eid = s1_ids[global_q].decode("utf-8", "ignore").strip()

                cand_tgt_ids = [decode_tgt_id(t) for t in cands_per_query[local_i]]
                match_tgt_ids = [decode_tgt_id(t) for t in matched_per_query[local_i]]

                # Ensure candidate uniqueness and match subset constraint
                cand_unique = list(dict.fromkeys(cand_tgt_ids))
                match_unique = list(dict.fromkeys(match_tgt_ids))

                total_cand_pairs += len(cand_unique)
                total_matches += len(match_unique)
                if not match_unique:
                    total_singletons += 1

                f_cand.write(f"{s1_eid}\t{','.join(cand_unique)}\n")
                f_match.write(f"{s1_eid}\t{','.join(match_unique)}\n")

            if (hi % 100_000 == 0) or (hi == n_s1):
                el = time.time() - t0_inf
                q_per_sec = hi / max(1e-5, el)
                print(f"  Processed {hi:,}/{n_s1:,} queries ({hi/n_s1:.1%}) | "
                      f"Speed: {q_per_sec:,.0f} q/s | "
                      f"Cand/q: {total_cand_pairs/hi:.1f} | "
                      f"Matches/q: {total_matches/hi:.2f} | "
                      f"Singletons: {total_singletons/hi:.1%}", flush=True)

    elapsed = time.time() - t0_inf
    print(f"\nInference completed in {elapsed:.1f}s ({n_s1/max(1e-5, elapsed):,.0f} queries/sec)")
    print(f"Candidate output: {candidate_out} ({total_cand_pairs:,} total candidate pairs)")
    print(f"Matching output:  {matching_out} ({total_matches:,} total matches, {total_singletons:,} singletons)")

    # 5. Strict Competition Format Validation
    print("\n--- STAGE 3: Strict Submission Format Validation ---")
    errors, warnings = validate(matching_out, candidate_out, test_data_dir)
    if errors:
        print(f"[ERROR] Submission validation failed with {len(errors)} errors:")
        for err in errors[:10]:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("[SUCCESS] PASS: All competition format, singleton, subset, and schema constraints PASSED!")
        if warnings:
            print(f"[INFO] {len(warnings)} non-fatal warnings reported:")
            for w in warnings[:5]:
                print(f"  - {w}")


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 Pipeline Runner")
    parser.add_argument("--cache-root", default=r"S:\Amazon_ML_Challenge\cache", help="Path to cache directory")
    parser.add_argument("--test-dir", default=r"S:\Amazon_ML_Challenge\dataset\student_resource\dataset\test", help="Path to test data directory")
    parser.add_argument("--output-dir", default=r"S:\Amazon_ML_Challenge\output", help="Path to output directory")
    parser.add_argument("--model-path", default=r"S:\Amazon_ML_Challenge\saved_models\gbdt_model.pkl", help="Path to saved GBDT model")
    parser.add_argument("--top-k", type=int, default=30, help="Max candidates per query from blocker")
    parser.add_argument("--match-threshold", type=float, default=0.65, help="Decision threshold for match prediction")
    parser.add_argument("--block-size", type=int, default=10_000, help="Query batch size for streaming")
    parser.add_argument("--force-build-index", action="store_true", help="Force rebuild candidate index")
    args = parser.parse_args()

    run_test_inference(
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        model_path=args.model_path,
        test_data_dir=args.test_dir,
        top_k=args.top_k,
        block_size=args.block_size,
        match_threshold=args.match_threshold,
        force_build_index=args.force_build_index,
    )


if __name__ == "__main__":
    main()
