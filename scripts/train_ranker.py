"""
End-to-end training and local validation script for LightGBM Matcher.
Measures exact per-entity Macro F0.5 score and optimizes precision threshold.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-queries", type=int, default=30_000, help="Number of training queries to block")
    parser.add_argument("--top-k", type=int, default=30)
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

    # 1. Select query span
    rng = np.random.default_rng(args.seed)
    n_queries = min(args.n_queries, n_s1)
    start_q = 0
    query_span = np.arange(start_q, start_q + n_queries)

    print(f"\n--- 1. Candidate Generation (Blocking {n_queries:,} queries) ---")
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

    # 2. Label candidate pairs using binary search on pos_keys
    pair_keys = all_q.astype(np.int64) * n_targets + all_t.astype(np.int64)
    p = np.searchsorted(pos_keys, pair_keys)
    p_clipped = np.minimum(p, len(pos_keys) - 1)
    labels = (p < len(pos_keys)) & (pos_keys[p_clipped] == pair_keys)
    y = labels.astype(np.int32)
    n_pos = int(y.sum())
    print(f"Candidate labels: {n_pos:,} true positives, {len(y)-n_pos:,} negatives (Positive rate: {n_pos/len(y):.2%})")

    # 3. Extract multi-modal features
    print(f"\n--- 2. Multi-Modal Feature Extraction ---")
    t0 = time.time()
    extractor = FeatureExtractor()
    X = extractor.extract_batch_from_cache(CACHE_ROOT, SPLIT, all_q, all_t, all_scores)
    print(f"Extracted {X.shape[1]} features for {X.shape[0]:,} pairs in {time.time()-t0:.1f}s")

    # 4. Train/Val Split (by query entity, avoiding data leakage)
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

    # 5. Train LightGBM
    print(f"\n--- 3. GBDT Model Training ---")
    ranker = GBDTRanker()
    ranker.train(X_train, y_train, X_val, y_val, feature_names=extractor.feature_names)

    val_probs = ranker.predict_proba(X_val)
    auc = roc_auc_score(y_val, val_probs)
    print(f"\nValidation Pairwise ROC-AUC: {auc:.4f}")

    # 6. Build ground truth dictionary for validation queries
    lo_k = (start_q + n_train_q) * n_targets
    hi_k = (start_q + n_queries) * n_targets
    sub = pos_keys[np.searchsorted(pos_keys, lo_k, "left"):np.searchsorted(pos_keys, hi_k, "right")]
    sr = (sub // n_targets).astype(np.int64)
    sc = (sub % n_targets).astype(np.int64)
    val_gt = {q: set() for q in val_queries}
    for r, c in zip(sr.tolist(), sc.tolist()):
        if r in val_gt:
            val_gt[r].add(c)

    # 7. Optimize threshold for exact Macro F0.5
    print(f"\n--- 4. Threshold Tuning for Macro F0.5 ---")
    best_t = ranker.optimize_threshold(
        val_probs, q_val, t_val, val_gt, sorted(val_queries)
    )

    # Detailed threshold ablation
    print("\nThreshold Sweep Performance:")
    print(f"{'Threshold':>10} | {'Macro F0.5':>12} | {'Precision':>10} | {'Recall':>10} | {'Singletons':>12}")
    print("-" * 65)

    q_to_pairs = {q: [] for q in val_queries}
    for q, t, p in zip(q_val.tolist(), t_val.tolist(), val_probs.tolist()):
        q_to_pairs[q].append((t, p))

    for t in [0.50, 0.60, 0.65, 0.70, 0.72, 0.75, 0.78, 0.80, 0.85]:
        scores = []
        tot_tp = 0
        tot_pred = 0
        tot_true = 0
        correct_singletons = 0
        total_singletons = 0

        for q in sorted(val_queries):
            true_s = val_gt[q]
            pred_s = {tgt for tgt, pr in q_to_pairs.get(q, []) if pr >= t}
            sc_val = calculate_entity_f_beta(true_s, pred_s, beta=0.5)
            scores.append(sc_val)

            tp = len(true_s.intersection(pred_s))
            tot_tp += tp
            tot_pred += len(pred_s)
            tot_true += len(true_s)

            if len(true_s) == 0:
                total_singletons += 1
                if len(pred_s) == 0:
                    correct_singletons += 1

        macro_f05 = np.mean(scores)
        prec = tot_tp / max(1, tot_pred)
        rec = tot_tp / max(1, tot_true)
        sing_acc = correct_singletons / max(1, total_singletons)
        flag = " <--- OPTIMAL" if abs(t - best_t) < 0.02 else ""
        print(f"{t:10.2f} | {macro_f05:12.4f} | {prec:10.4f} | {rec:10.4f} | {sing_acc:12.1%}{flag}")

    # Feature importances
    print(f"\n--- 5. Top 15 Feature Importances ---")
    importances = ranker.model.feature_importances_
    order = np.argsort(importances)[::-1]
    for rank, idx in enumerate(order[:15], 1):
        print(f"  {rank:2d}. {extractor.feature_names[idx]:<28} : {importances[idx]:6d}")

    # Save model
    ranker.save(args.model_out)
    print(f"\nTraining pipeline finished successfully!")


if __name__ == "__main__":
    main()
