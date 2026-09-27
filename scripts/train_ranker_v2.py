"""
High-Performance Model Training & Two-Stage Threshold Optimizer for Team Technocrats.
Maximizes per-entity Macro F0.5 via Two-Stage Thresholding (Singleton Safeguard + Match Selection).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import roc_auc_score

from src.blocking.blocker import RareTokenBlocker
from src.data.loader import fld
from src.evaluation.metrics import calculate_entity_f_beta, compute_macro_f05
from src.features.feature_extractor import FeatureExtractor
from src.models.gbdt_ranker import GBDTRanker

CACHE_ROOT = r"S:\Amazon_ML_Challenge\cache"
SPLIT = "train"


def optimize_two_stage_thresholds(
    q_val: np.ndarray,
    t_val: np.ndarray,
    val_probs: np.ndarray,
    val_gt: dict,
    val_queries: set,
):
    """
    Grid search for optimal two-stage decision policy:
    1. Singleton Safeguard: If max prob for query < T_sing -> predict empty list.
    2. Match Selection: Otherwise include all target candidates with prob >= T_match.
    """
    print("\n--- STAGE 4: Two-Stage Threshold Grid Search for Macro F0.5 ---")
    t0 = time.time()

    q_to_pairs = {q: [] for q in val_queries}
    for q, t, p in zip(q_val.tolist(), t_val.tolist(), val_probs.tolist()):
        if q in q_to_pairs:
            q_to_pairs[q].append((t, p))

    best_f05 = -1.0
    best_t_sing = 0.48
    best_t_match = 0.52

    t_sing_grid = np.linspace(0.40, 0.65, 11)
    t_match_grid = np.linspace(0.45, 0.75, 13)

    for t_sing in t_sing_grid:
        for t_match in t_match_grid:
            scores = []
            for q in sorted(val_queries):
                true_s = val_gt[q]
                pairs = q_to_pairs.get(q, [])
                if not pairs:
                    pred_s = set()
                else:
                    max_p = max(p for _, p in pairs)
                    if max_p < t_sing:
                        pred_s = set()
                    else:
                        pred_s = {tgt for tgt, p in pairs if p >= t_match}

                sc = calculate_entity_f_beta(true_s, pred_s, beta=0.5)
                scores.append(sc)

            macro_f05 = np.mean(scores)
            if macro_f05 > best_f05:
                best_f05 = macro_f05
                best_t_sing = float(t_sing)
                best_t_match = float(t_match)

    print(f"Grid search complete in {time.time()-t0:.1f}s")
    print(f"Optimal Two-Stage Policy: T_singleton={best_t_sing:.3f}, T_match={best_t_match:.3f} | Best Validation Macro F0.5 = {best_f05:.4f}")
    return best_t_sing, best_t_match, best_f05


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-queries", type=int, default=150_000, help="Number of training queries to block")
    parser.add_argument("--top-k", type=int, default=35)
    parser.add_argument("--model-out", type=str, default=r"S:\Amazon_ML_Challenge\saved_models\gbdt_model.pkl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    s1_path = os.path.join(CACHE_ROOT, SPLIT, "s1")
    s2_path = os.path.join(CACHE_ROOT, SPLIT, "s2")
    s3_path = os.path.join(CACHE_ROOT, SPLIT, "s3")
    gt_path = os.path.join(CACHE_ROOT, SPLIT, "gt")

    n_s1 = len(np.load(fld(s1_path, "idh"), mmap_mode="r"))
    n2 = len(np.load(fld(s2_path, "idh"), mmap_mode="r"))
    n3 = len(np.load(fld(s3_path, "idh"), mmap_mode="r"))
    n_targets = n2 + n3
    pos_keys = np.load(fld(gt_path, "pos_keys"))

    print(f"Loaded dataset: S1={n_s1:,} | S2={n2:,} | S3={n3:,} | TargetPool={n_targets:,}")

    rng = np.random.default_rng(args.seed)
    n_queries = min(args.n_queries, n_s1)
    start_q = 0
    query_span = np.arange(start_q, start_q + n_queries)

    print(f"\n--- STAGE 1: Blocking Candidate Generation ({n_queries:,} queries, top_k={args.top_k}) ---")
    t0 = time.time()
    blocker = RareTokenBlocker(CACHE_ROOT, SPLIT, top_k=args.top_k).build(force=False, verbose=False)

    q_rows_l = []
    t_rows_l = []
    scores_l = []
    step = 5000

    for lo in range(start_q, start_q + n_queries, step):
        hi = min(lo + step, start_q + n_queries)
        r, c, v = blocker.block_range(lo, hi)
        if len(r) > 0:
            q_rows_l.append(r + lo)
            t_rows_l.append(c)
            scores_l.append(v)

    all_q = np.concatenate(q_rows_l)
    all_t = np.concatenate(t_rows_l)
    all_scores = np.concatenate(scores_l)
    print(f"Generated {len(all_q):,} candidate pairs ({len(all_q)/n_queries:.1f} per query) in {time.time()-t0:.1f}s")

    pair_keys = all_q.astype(np.int64) * n_targets + all_t.astype(np.int64)
    p = np.searchsorted(pos_keys, pair_keys)
    p_clipped = np.minimum(p, len(pos_keys) - 1)
    labels = (p < len(pos_keys)) & (pos_keys[p_clipped] == pair_keys)
    y = labels.astype(np.int32)
    n_pos = int(y.sum())
    print(f"Candidate labels: {n_pos:,} true positives, {len(y)-n_pos:,} negatives (Positive rate: {n_pos/len(y):.2%})")

    print(f"\n--- STAGE 2: Multi-Modal Feature Extraction ---")
    t0 = time.time()
    extractor = FeatureExtractor()
    X = extractor.extract_batch_from_cache(CACHE_ROOT, SPLIT, all_q, all_t, all_scores)
    print(f"Extracted {X.shape[1]} features for {X.shape[0]:,} pairs in {time.time()-t0:.1f}s")

    val_ratio = 0.2
    n_train_q = int(n_queries * (1.0 - val_ratio))
    train_queries = set(range(start_q, start_q + n_train_q))
    val_queries = set(range(start_q + n_train_q, start_q + n_queries))

    train_mask = np.isin(all_q, list(train_queries))
    val_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    q_val, t_val = all_q[val_mask], all_t[val_mask]

    print(f"Train split: {len(X_train):,} pairs ({int(y_train.sum()):,} pos)")
    print(f"Val split:   {len(X_val):,} pairs ({int(y_val.sum()):,} pos)")

    print(f"\n--- STAGE 3: High-Capacity LightGBM Model Training ---")
    ranker = GBDTRanker(model_params={
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 750,
        "learning_rate": 0.03,
        "max_depth": 9,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    })
    ranker.train(X_train, y_train, X_val, y_val, feature_names=extractor.feature_names)

    val_probs = ranker.predict_proba(X_val)
    auc = roc_auc_score(y_val, val_probs)
    print(f"\nValidation Pairwise ROC-AUC: {auc:.5f}")

    lo_k = (start_q + n_train_q) * n_targets
    hi_k = (start_q + n_queries) * n_targets
    sub = pos_keys[np.searchsorted(pos_keys, lo_k, "left"):np.searchsorted(pos_keys, hi_k, "right")]
    sr = (sub // n_targets).astype(np.int64)
    sc = (sub % n_targets).astype(np.int64)
    val_gt = {q: set() for q in val_queries}
    for r, c in zip(sr.tolist(), sc.tolist()):
        if r in val_gt:
            val_gt[r].add(c)

    t_sing, t_match, best_f05 = optimize_two_stage_thresholds(
        q_val, t_val, val_probs, val_gt, val_queries
    )

    ranker.optimal_threshold = t_match
    ranker.save(args.model_out)

    # Write parameters out to json
    param_out = os.path.join(os.path.dirname(args.model_out), "threshold_policy.json")
    import json
    with open(param_out, "w") as f:
        json.dump({"t_singleton": t_sing, "t_match": t_match, "val_f05": best_f05}, f, indent=2)
    print(f"Saved threshold policy to {param_out}")


if __name__ == "__main__":
    main()
