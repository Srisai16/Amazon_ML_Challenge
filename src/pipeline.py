"""
Master Pipeline Runner for Amazon ML Challenge 2026.
Executes End-to-End Data Ingestion -> Blocking -> Feature Extraction -> Modeling -> Validation.
"""

import argparse
import os
import sys
from typing import Dict, List, Set, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.blocking.hybrid_blocker import HybridBlocker
from src.config import cfg
from src.evaluation.metrics import compute_blocking_metrics, compute_macro_f05
from src.features.feature_extractor import FeatureExtractor
from src.models.gbdt_ranker import GBDTRanker
from utils.validate_submission import validate


def load_dataset(data_dir: str, is_train: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[Dict[str, Set[str]]]]:
    """Loads source1, source2, source3, and ground_truth TSV files with explicit tab separator."""
    prefix = "train" if is_train else "test"
    s1_path = os.path.join(data_dir, f"{prefix}_source1.tsv")
    s2_path = os.path.join(data_dir, f"{prefix}_source2.tsv")
    s3_path = os.path.join(data_dir, f"{prefix}_source3.tsv")

    print(f"Loading data from {data_dir} (is_train={is_train})...")
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str).fillna("")
    df_s2 = pd.read_csv(s2_path, sep="\t", dtype=str).fillna("")
    df_s3 = pd.read_csv(s3_path, sep="\t", dtype=str).fillna("")

    ground_truth = None
    if is_train:
        gt_path = os.path.join(data_dir, "train_ground_truth.tsv")
        df_gt = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
        ground_truth = {}
        for _, row in df_gt.iterrows():
            s1_id = row["source1_entity_id"]
            raw_m = str(row["matched_entity_ids"]).strip()
            matches = {m.strip() for m in raw_m.split(",") if m.strip()} if raw_m else set()
            ground_truth[s1_id] = matches

    return df_s1, df_s2, df_s3, ground_truth


def run_pipeline(train_dir: str, test_dir: str, output_dir: str, max_candidates: int = 25):
    os.makedirs(output_dir, exist_ok=True)
    matching_out = os.path.join(output_dir, "matching_results.tsv")
    candidate_out = os.path.join(output_dir, "candidate_pairs.tsv")

    print("=" * 75)
    print("🚀 Starting Amazon ML Challenge 2026 End-to-End Pipeline")
    print("=" * 75)

    # 1. Load Training Data
    train_s1, train_s2, train_s3, train_gt = load_dataset(train_dir, is_train=True)
    train_targets = pd.concat([train_s2, train_s3], ignore_index=True)
    print(f"Train Source 1 records: {len(train_s1)} | Target pool (S2+S3): {len(train_targets)}")

    # 2. Train/Val Split for Local CV
    s1_train_ids, s1_val_ids = train_test_split(
        train_s1["entity_id"].tolist(),
        test_size=0.2,
        random_state=cfg.SEED
    )
    df_s1_train = train_s1[train_s1["entity_id"].isin(s1_train_ids)].copy()
    df_s1_val = train_s1[train_s1["entity_id"].isin(s1_val_ids)].copy()
    val_gt = {s1_id: train_gt.get(s1_id, set()) for s1_id in s1_val_ids}

    # 3. Blocking / Candidate Generation
    print("\n--- STAGE 1: Candidate Generation (Blocking) ---")
    blocker = HybridBlocker(max_candidates_per_s1=max_candidates)
    blocker.fit(train_targets)

    train_cands = blocker.block_all(df_s1_train)
    val_cands = blocker.block_all(df_s1_val)

    # Evaluate validation blocking metrics
    block_metrics = compute_blocking_metrics(val_gt, val_cands, len(train_targets))
    print(f"Validation Blocking Recall Ceiling: {block_metrics['pair_completeness_recall']:.4f}")
    print(f"Validation Average Candidates / S1: {block_metrics['avg_candidates_per_s1']:.2f}")
    print(f"Validation Reduction Ratio:         {block_metrics['reduction_ratio']:.6f}")

    # 4. Feature Extraction
    print("\n--- STAGE 2: Multi-Modal Feature Extraction ---")
    extractor = FeatureExtractor()
    s1_dict = train_s1.set_index("entity_id").to_dict("index")
    target_dict = train_targets.set_index("entity_id").to_dict("index")

    X_train, y_train, train_pairs = extractor.extract_candidate_dataset_features(
        s1_dict, target_dict, train_cands, train_gt
    )
    X_val, y_val, val_pairs = extractor.extract_candidate_dataset_features(
        s1_dict, target_dict, val_cands, val_gt
    )
    print(f"Training Features Matrix:   {X_train.shape} (Positives: {sum(y_train)})")
    print(f"Validation Features Matrix: {X_val.shape} (Positives: {sum(y_val)})")

    # 5. GBDT Ranker Training & Threshold Optimization for Macro F0.5
    print("\n--- STAGE 3: GBDT Model Training & F0.5 Threshold Tuning ---")
    ranker = GBDTRanker()
    ranker.train(X_train, y_train, X_val, y_val)

    val_probs = ranker.predict_proba(X_val)
    best_t = ranker.optimize_threshold(val_probs, val_pairs, val_gt, s1_val_ids)

    # 6. Test Inference & Final Packaging
    print("\n--- STAGE 4: Test Ingestion, Candidate Generation & Prediction ---")
    test_s1, test_s2, test_s3, _ = load_dataset(test_dir, is_train=False)
    test_targets = pd.concat([test_s2, test_s3], ignore_index=True)

    test_blocker = HybridBlocker(max_candidates_per_s1=max_candidates)
    test_blocker.fit(test_targets)
    test_candidates = test_blocker.block_all(test_s1)

    # Save candidate_pairs.tsv
    test_blocker.save_candidate_pairs_tsv(test_candidates, candidate_out)

    # Test Feature Extraction & Prediction
    test_s1_dict = test_s1.set_index("entity_id").to_dict("index")
    test_target_dict = test_targets.set_index("entity_id").to_dict("index")

    X_test, _, test_pairs = extractor.extract_candidate_dataset_features(
        test_s1_dict, test_target_dict, test_candidates
    )
    test_probs = ranker.predict_proba(X_test)
    test_predictions = ranker.generate_predictions(
        test_probs, test_pairs, test_s1["entity_id"].tolist(), threshold=best_t
    )

    # Save matching_results.tsv
    ranker.save_matching_results_tsv(test_predictions, matching_out)

    # 7. Self-Validation
    print("\n--- STAGE 5: Strict Submission Format Validation ---")
    errors, warnings = validate(matching_out, candidate_out, test_dir)
    if errors:
        print(f"❌ Pipeline finished with validation errors! Please review.")
        sys.exit(1)
    else:
        print("🎉 SUCCESS! Submission files generated, validated, and ready for upload!")


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 Pipeline")
    parser.add_argument("--train-dir", default=cfg.paths.TRAIN_DIR, help="Path to train data directory")
    parser.add_argument("--test-dir", default=cfg.paths.TEST_DIR, help="Path to test data directory")
    parser.add_argument("--output-dir", default=cfg.paths.OUTPUT_DIR, help="Path to output directory")
    parser.add_argument("--max-candidates", type=int, default=cfg.blocking.MAX_CANDIDATES_PER_SOURCE1)
    args = parser.parse_args()

    run_pipeline(args.train_dir, args.test_dir, args.output_dir, args.max_candidates)


if __name__ == "__main__":
    main()
