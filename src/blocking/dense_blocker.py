"""
Dense Embedding Candidate Generation using SentenceTransformers and FAISS / PyTorch.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.preprocessing.normalizer import normalizer


class DenseBlocker:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", top_k: int = 15):
        self.model_name = model_name
        self.top_k = top_k
        self.model = None
        self.target_ids: List[str] = []
        self.target_embeddings: Optional[np.ndarray] = None

    def _lazy_load_model(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"[DenseBlocker] Warning: SentenceTransformer not available ({e}). Using TF-IDF fallback.")
                self.model = None

    def fit(self, target_df: pd.DataFrame, batch_size: int = 64):
        """
        Computes dense embeddings for the target pool (S2 + S3).
        """
        self._lazy_load_model()
        self.target_ids = target_df["entity_id"].tolist()

        texts = []
        for _, row in target_df.iterrows():
            composite = normalizer.create_composite_representation(
                str(row.get("business_name", "")),
                str(row.get("business_address", "")),
                str(row.get("country", ""))
            )
            texts.append(composite)

        if self.model is not None:
            self.target_embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True
            )
        else:
            # Fallback to TF-IDF vectorizer
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(max_features=5000)
            self.target_embeddings = self.vectorizer.fit_transform(texts).toarray()

    def block_all(self, source1_df: pd.DataFrame, top_k: int = None, batch_size: int = 64) -> Dict[str, List[str]]:
        """
        Retrieves top-K dense semantic candidates for each Source 1 entity.
        """
        k = top_k or self.top_k
        s1_ids = source1_df["entity_id"].tolist()

        texts = []
        for _, row in source1_df.iterrows():
            composite = normalizer.create_composite_representation(
                str(row.get("business_name", "")),
                str(row.get("business_address", "")),
                str(row.get("country", ""))
            )
            texts.append(composite)

        if self.model is not None:
            query_embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True
            )
        else:
            query_embeddings = self.vectorizer.transform(texts).toarray()

        # Cosine similarity (since embeddings are normalized)
        # sim_matrix: [N_s1, N_targets]
        sim_matrix = np.dot(query_embeddings, self.target_embeddings.T)

        candidates = {}
        for i, s1_id in enumerate(s1_ids):
            top_indices = np.argsort(sim_matrix[i])[::-1][:k]
            candidates[s1_id] = [self.target_ids[idx] for idx in top_indices]

        return candidates
