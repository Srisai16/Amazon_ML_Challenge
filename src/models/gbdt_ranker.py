"""
Gradient Boosted Decision Tree (LightGBM) Pairwise Matcher & Ranker.
Optimized for high-precision entity resolution and Macro F0.5.
"""

import os
from typing import Dict, List, Optional, Set, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import cfg
from src.evaluation.metrics import compute_macro_f05


class GBDTRanker:
    def __init__(self, model_params: Optional[Dict] = None):
        self.params = model_params or {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "n_estimators": cfg.model.GBDT_N_ESTIMATORS,
            "learning_rate": cfg.model.GBDT_LEARNING_RATE,
            "max_depth": cfg.model.GBDT_MAX_DEPTH,
            "num_leaves": cfg.model.GBDT_NUM_LEAVES,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": cfg.SEED,
            "verbose": -1,
            "scale_pos_weight": 1.0,  # Balanced to avoid over-predicting false positives
        }
        self.model = lgb.LGBMClassifier(**self.params)
        self.feature_names: List[str] = []
        self.optimal_threshold: float = cfg.model.MATCH_THRESHOLD

    def train(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None):
        """Fits LightGBM classifier on extracted entity pair features."""
        self.feature_names = list(X_train.columns)
        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None

        print(f"[GBDTRanker] Training LightGBM on {len(X_train)} candidate pairs ({sum(y_train)} positives)...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)] if eval_set else None
        )
        print("[GBDTRanker] Training complete.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns match probabilities for each pair."""
        return self.model.predict_proba(X)[:, 1]

    def optimize_threshold(
        self,
        val_probs: np.ndarray,
        val_pair_ids: List[str],
        ground_truth: Dict[str, Set[str]],
        s1_ids: List[str],
        threshold_range: np.ndarray = np.linspace(0.4, 0.9, 51)
    ) -> float:
        """
        Grid search on validation set to find the threshold that maximizes Macro F0.5.
        """
        best_threshold = self.optimal_threshold
        best_f05 = -1.0

        # Pre-group pair probs by S1 ID
        s1_to_pair_probs = {s1: [] for s1 in s1_ids}
        for prob, pair_id in zip(val_probs, val_pair_ids):
            s1_id, t_id = pair_id.split("::")
            if s1_id in s1_to_pair_probs:
                s1_to_pair_probs[s1_id].append((t_id, prob))

        for t in threshold_range:
            preds = {}
            for s1_id in s1_ids:
                matches = {t_id for t_id, p in s1_to_pair_probs.get(s1_id, []) if p >= t}
                preds[s1_id] = matches

            f05, _ = compute_macro_f05(ground_truth, preds)
            if f05 > best_f05:
                best_f05 = f05
                best_threshold = t

        print(f"[GBDTRanker] Optimized Threshold: {best_threshold:.3f} | Best Validation Macro F0.5: {best_f05:.4f}")
        self.optimal_threshold = best_threshold
        return best_threshold

    def generate_predictions(
        self,
        probs: np.ndarray,
        pair_ids: List[str],
        all_s1_ids: List[str],
        threshold: Optional[float] = None
    ) -> Dict[str, List[str]]:
        """
        Generates final matched IDs for every Source 1 entity.
        Enforces:
        - Exactly one row per Source 1 entity
        - Only predicted target IDs >= threshold
        - Singletons mapped to empty list
        """
        t = threshold if threshold is not None else self.optimal_threshold
        predictions = {s1: [] for s1 in all_s1_ids}

        for prob, pair_id in zip(probs, pair_ids):
            s1_id, t_id = pair_id.split("::")
            if prob >= t:
                if s1_id in predictions:
                    if t_id not in predictions[s1_id]:
                        predictions[s1_id].append(t_id)

        return predictions

    @staticmethod
    def save_matching_results_tsv(predictions: Dict[str, List[str]], output_path: str):
        """
        Saves predictions to matching_results.tsv:
        source1_entity_id<TAB>matched_entity_ids (comma separated)
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for s1_id, matches in predictions.items():
                match_str = ",".join(matches)
                f.write(f"{s1_id}\t{match_str}\n")
        print(f"[GBDTRanker] Saved matching results to {output_path} ({len(predictions)} entities)")
