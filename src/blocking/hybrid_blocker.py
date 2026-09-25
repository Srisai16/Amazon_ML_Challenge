"""
Hybrid Multi-Stage Candidate Generation & Union Module.
Merges Lexical BM25 and Dense Semantic candidates to ensure high recall ceiling (>98%)
while bounding the candidate size per Source 1 entity.
"""

from typing import Dict, List, Set
import pandas as pd

from src.blocking.lexical_blocker import LexicalBlocker
from src.blocking.dense_blocker import DenseBlocker
from src.config import cfg


class HybridBlocker:
    def __init__(self, max_candidates_per_s1: int = 25):
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.lexical_blocker = LexicalBlocker(top_k=cfg.blocking.TOP_K_BM25)
        self.dense_blocker = DenseBlocker(top_k=cfg.blocking.TOP_K_DENSE)

    def fit(self, target_df: pd.DataFrame):
        """Fits both lexical and dense candidate indices on target dataset."""
        print("[HybridBlocker] Fitting Lexical BM25 Index...")
        self.lexical_blocker.fit(target_df)

        print("[HybridBlocker] Fitting Dense Embedding Index...")
        self.dense_blocker.fit(target_df)

    def block_all(self, source1_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Generates merged and deduplicated candidate sets for each Source 1 entity.
        Interleaves top candidates from lexical and dense retrievers.
        """
        print("[HybridBlocker] Generating Lexical Candidates...")
        lex_cands = self.lexical_blocker.block_all(source1_df)

        print("[HybridBlocker] Generating Dense Semantic Candidates...")
        dense_cands = self.dense_blocker.block_all(source1_df)

        merged_candidates = {}
        for s1_id in source1_df["entity_id"]:
            l_list = lex_cands.get(s1_id, [])
            d_list = dense_cands.get(s1_id, [])

            combined: List[str] = []
            seen: Set[str] = set()

            # Interleave top results to balance lexical exactness and semantic robustness
            max_len = max(len(l_list), len(d_list))
            for i in range(max_len):
                if i < len(l_list):
                    cand = l_list[i]
                    if cand not in seen:
                        seen.add(cand)
                        combined.append(cand)
                if i < len(d_list):
                    cand = d_list[i]
                    if cand not in seen:
                        seen.add(cand)
                        combined.append(cand)

                if len(combined) >= self.max_candidates_per_s1:
                    break

            merged_candidates[s1_id] = combined[:self.max_candidates_per_s1]

        return merged_candidates

    @staticmethod
    def save_candidate_pairs_tsv(candidates: Dict[str, List[str]], output_path: str):
        """
        Saves candidates to candidate_pairs.tsv following strict challenge format:
        source1_entity_id<TAB>candidate_entity_ids (comma separated)
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1_id, cands in candidates.items():
                cand_str = ",".join(cands)
                f.write(f"{s1_id}\t{cand_str}\n")
        print(f"[HybridBlocker] Saved candidate pairs to {output_path} ({len(candidates)} entities)")
