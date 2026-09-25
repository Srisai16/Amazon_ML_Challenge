"""
Lexical / Inverted-Index / BM25 Candidate Generation (Blocking).
Fast, scalable candidate retrieval using character n-grams, token overlap, and BM25.
"""

from collections import defaultdict
import re
from typing import Dict, List, Set, Tuple
import pandas as pd
from rank_bm25 import BM25Okapi

from src.preprocessing.normalizer import normalizer


class LexicalBlocker:
    def __init__(self, top_k: int = 20):
        self.top_k = top_k
        self.target_records: List[Dict] = []
        self.target_ids: List[str] = []
        self.bm25_name_index = None
        self.bm25_address_index = None
        self.country_to_indices = defaultdict(list)

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        # Return word tokens and character 3-grams for typo robustness
        words = re.findall(r"\b\w+\b", text.lower())
        char_3grams = [text[i:i+3] for i in range(max(0, len(text) - 2))]
        return words + char_3grams

    def fit(self, target_df: pd.DataFrame):
        """
        Fits BM25 and country indices on target dataset (S2 + S3 combined).
        """
        self.target_records = target_df.to_dict("records")
        self.target_ids = target_df["entity_id"].tolist()

        name_corpus = []
        address_corpus = []

        for idx, row in enumerate(self.target_records):
            clean_name = normalizer.normalize_business_name(str(row.get("business_name", "")))
            clean_addr = normalizer.normalize_address(str(row.get("business_address", "")))
            country = str(row.get("country", "")).strip().lower()

            name_corpus.append(self._tokenize(clean_name))
            address_corpus.append(self._tokenize(clean_addr))
            self.country_to_indices[country].append(idx)

        self.bm25_name_index = BM25Okapi(name_corpus)
        self.bm25_address_index = BM25Okapi(address_corpus)

    def retrieve_candidates(self, query_name: str, query_address: str, query_country: str, top_k: int = None) -> List[str]:
        """
        Retrieves top candidate entity_ids for a single Source 1 query.
        """
        k = top_k or self.top_k
        clean_name = normalizer.normalize_business_name(query_name)
        clean_addr = normalizer.normalize_address(query_address)
        clean_country = (query_country or "").strip().lower()

        name_tokens = self._tokenize(clean_name)
        addr_tokens = self._tokenize(clean_addr)

        name_scores = self.bm25_name_index.get_scores(name_tokens) if name_tokens else [0.0] * len(self.target_ids)
        addr_scores = self.bm25_address_index.get_scores(addr_tokens) if addr_tokens else [0.0] * len(self.target_ids)

        # Country filter: prioritize candidates from the same country
        allowed_indices = self.country_to_indices.get(clean_country, list(range(len(self.target_ids))))
        if not allowed_indices:
            allowed_indices = list(range(len(self.target_ids)))

        combined_scores = []
        for idx in allowed_indices:
            # Weighted combination of name score and address score
            score = (2.0 * name_scores[idx]) + (1.0 * addr_scores[idx])
            if score > 0:
                combined_scores.append((idx, score))

        combined_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in combined_scores[:k]]
        return [self.target_ids[idx] for idx in top_indices]

    def block_all(self, source1_df: pd.DataFrame, top_k: int = None) -> Dict[str, List[str]]:
        """
        Generates candidate mapping for all records in Source 1 DataFrame.
        """
        candidates = {}
        for _, row in source1_df.iterrows():
            s1_id = row["entity_id"]
            name = str(row.get("business_name", ""))
            addr = str(row.get("business_address", ""))
            country = str(row.get("country", ""))
            cands = self.retrieve_candidates(name, addr, country, top_k=top_k)
            candidates[s1_id] = cands
        return candidates
