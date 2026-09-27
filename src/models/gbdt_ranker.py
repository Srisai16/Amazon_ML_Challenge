"""
Gradient Boosted Decision Tree (LightGBM) Pairwise Matcher & Ranker.
Optimized for high-precision entity resolution and Macro F0.5.
"""

import os
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import joblib
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
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 7,
            "num_leaves": 45,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": cfg.SEED,
            "n_jobs": -1,
            "verbose": -1,
            "scale_pos_weight": 1.0,
        }
        self.model = lgb.LGBMClassifier(**self.params)
        self.feature_names: List[str] = []
        self.optimal_threshold: float = 0.72
        self.country_thresholds: Dict[str, float] = {
            "US": 0.72,
            "India": 0.70,
            "France": 0.73,
            "default": 0.72,
        }

    def train(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: np.ndarray,
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ):
        """Fits LightGBM classifier on extracted entity pair features."""
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)

        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        n_pos = int(y_train.sum())

        print(f"[GBDTRanker] Training LightGBM on {len(X_train):,} pairs ({n_pos:,} positives, {n_pos/max(1, len(X_train)):.1%})...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)] if eval_set else None,
        )
        print("[GBDTRanker] Training complete.")

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Returns match probabilities for each pair."""
        return self.model.predict_proba(X)[:, 1]

    def optimize_threshold(
        self,
        val_probs: np.ndarray,
        val_query_rows: np.ndarray,
        val_target_rows: np.ndarray,
        ground_truth: Dict[int, Set[int]],
        all_val_queries: Sequence[int],
        threshold_range: np.ndarray = np.linspace(0.45, 0.88, 44),
    ) -> float:
        """
        Grid search on validation set to find the threshold that maximizes Macro F0.5.
        """
        best_threshold = self.optimal_threshold
        best_f05 = -1.0

        # Pre-group pair probs by query index
        q_to_pairs = {q: [] for q in all_val_queries}
        for q, t, p in zip(val_query_rows.tolist(), val_target_rows.tolist(), val_probs.tolist()):
            if q in q_to_pairs:
                q_to_pairs[q].append((t, p))

        for t in threshold_range:
            preds = {}
            for q in all_val_queries:
                matches = {tgt for tgt, p in q_to_pairs.get(q, []) if p >= t}
                preds[q] = matches

            f05, _ = compute_macro_f05(ground_truth, preds)
            if f05 > best_f05:
                best_f05 = f05
                best_threshold = float(t)

        print(f"[GBDTRanker] Optimized Threshold: {best_threshold:.3f} | Best Validation Macro F0.5: {best_f05:.4f}")
        self.optimal_threshold = best_threshold
        return best_threshold

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({"model": self.model, "threshold": self.optimal_threshold, "names": self.feature_names}, path)
        print(f"[GBDTRanker] Model saved to {path}")

    def load(self, path: str):
        data = joblib.load(path)
        self.model = data["model"]
        self.optimal_threshold = data.get("threshold", 0.72)
        self.feature_names = data.get("names", [])
        print(f"[GBDTRanker] Loaded model from {path} (threshold={self.optimal_threshold:.3f})")
