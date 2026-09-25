"""
High-Precision Multi-Modal Feature Extraction Module for Entity Pairs.
Computes 40+ tabular features comparing business name, address, numbers, and country.
"""

import math
import re
from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd

from rapidfuzz import fuzz, distance
import jellyfish

from src.preprocessing.normalizer import normalizer


class FeatureExtractor:
    def __init__(self):
        pass

    def _get_jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def _get_overlap_coefficient(self, set1: Set[str], set2: Set[str]) -> float:
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        min_len = min(len(set1), len(set2))
        return intersection / min_len if min_len > 0 else 0.0

    def _get_char_ngrams(self, text: str, n: int = 3) -> Set[str]:
        if not text or len(text) < n:
            return set()
        return {text[i:i+n] for i in range(len(text) - n + 1)}

    def extract_pair_features(self, s1_rec: Dict, target_rec: Dict) -> Dict[str, float]:
        """
        Computes comprehensive similarity features for a single (S1, Target) entity pair.
        """
        feats = {}

        # 1. Clean & Normalized strings
        raw_name1 = str(s1_rec.get("business_name", ""))
        raw_name2 = str(target_rec.get("business_name", ""))
        name1 = normalizer.normalize_business_name(raw_name1)
        name2 = normalizer.normalize_business_name(raw_name2)
        name1_no_legal = normalizer.normalize_business_name(raw_name1, remove_legal=True)
        name2_no_legal = normalizer.normalize_business_name(raw_name2, remove_legal=True)

        raw_addr1 = str(s1_rec.get("business_address", ""))
        raw_addr2 = str(target_rec.get("business_address", ""))
        addr1 = normalizer.normalize_address(raw_addr1)
        addr2 = normalizer.normalize_address(raw_addr2)

        country1 = str(s1_rec.get("country", "")).strip().lower()
        country2 = str(target_rec.get("country", "")).strip().lower()

        target_id = str(target_rec.get("entity_id", ""))

        # 2. Basic Metadata & Source Indicators
        feats["is_same_country"] = 1.0 if country1 and country2 and (country1 == country2) else 0.0
        feats["is_source2"] = 1.0 if target_id.startswith("S2-") else 0.0
        feats["is_source3"] = 1.0 if target_id.startswith("S3-") else 0.0

        # Length features
        len_name1, len_name2 = len(name1), len(name2)
        feats["name_len_diff"] = abs(len_name1 - len_name2)
        feats["name_len_ratio"] = min(len_name1, len_name2) / max(len_name1, len_name2, 1)

        len_addr1, len_addr2 = len(addr1), len(addr2)
        feats["addr_len_diff"] = abs(len_addr1 - len_addr2)
        feats["addr_len_ratio"] = min(len_addr1, len_addr2) / max(len_addr1, len_addr2, 1)

        # 3. Business Name String Similarities (RapidFuzz / Levenshtein / Jaro-Winkler)
        feats["name_fuzz_ratio"] = fuzz.ratio(name1, name2) / 100.0
        feats["name_partial_ratio"] = fuzz.partial_ratio(name1, name2) / 100.0
        feats["name_token_sort_ratio"] = fuzz.token_sort_ratio(name1, name2) / 100.0
        feats["name_token_set_ratio"] = fuzz.token_set_ratio(name1, name2) / 100.0
        feats["name_wratio"] = fuzz.WRatio(name1, name2) / 100.0
        feats["name_jaro_winkler"] = distance.JaroWinkler.similarity(name1, name2)

        # Without legal suffix
        feats["name_no_legal_fuzz_ratio"] = fuzz.ratio(name1_no_legal, name2_no_legal) / 100.0
        feats["name_no_legal_token_sort"] = fuzz.token_sort_ratio(name1_no_legal, name2_no_legal) / 100.0

        # 4. Business Address String Similarities
        feats["addr_fuzz_ratio"] = fuzz.ratio(addr1, addr2) / 100.0
        feats["addr_partial_ratio"] = fuzz.partial_ratio(addr1, addr2) / 100.0
        feats["addr_token_sort_ratio"] = fuzz.token_sort_ratio(addr1, addr2) / 100.0
        feats["addr_token_set_ratio"] = fuzz.token_set_ratio(addr1, addr2) / 100.0
        feats["addr_jaro_winkler"] = distance.JaroWinkler.similarity(addr1, addr2)

        # 5. Token Overlap & Set Metrics
        name1_tokens = set(name1.split())
        name2_tokens = set(name2.split())
        feats["name_token_jaccard"] = self._get_jaccard_similarity(name1_tokens, name2_tokens)
        feats["name_token_overlap"] = self._get_overlap_coefficient(name1_tokens, name2_tokens)

        addr1_tokens = set(addr1.split())
        addr2_tokens = set(addr2.split())
        feats["addr_token_jaccard"] = self._get_jaccard_similarity(addr1_tokens, addr2_tokens)
        feats["addr_token_overlap"] = self._get_overlap_coefficient(addr1_tokens, addr2_tokens)

        # 6. Character N-Gram Similarities (3-gram and 4-gram)
        name1_3g = self._get_char_ngrams(name1, 3)
        name2_3g = self._get_char_ngrams(name2, 3)
        feats["name_3gram_jaccard"] = self._get_jaccard_similarity(name1_3g, name2_3g)

        name1_4g = self._get_char_ngrams(name1, 4)
        name2_4g = self._get_char_ngrams(name2, 4)
        feats["name_4gram_jaccard"] = self._get_jaccard_similarity(name1_4g, name2_4g)

        # 7. Exact Number & House / Suite Match (CRITICAL for Entity Resolution)
        nums1 = normalizer.extract_numbers(raw_addr1)
        nums2 = normalizer.extract_numbers(raw_addr2)
        common_nums = nums1.intersection(nums2)

        feats["num_shared_numbers"] = float(len(common_nums))
        feats["has_conflicting_numbers"] = 1.0 if nums1 and nums2 and not common_nums else 0.0
        feats["number_jaccard"] = self._get_jaccard_similarity(nums1, nums2)

        # 8. Postal Code Match
        pc1 = normalizer.extract_postal_code(raw_addr1, country1)
        pc2 = normalizer.extract_postal_code(raw_addr2, country2)
        feats["postal_code_exact_match"] = 1.0 if pc1 and pc2 and (pc1 == pc2) else 0.0
        feats["postal_code_mismatch"] = 1.0 if pc1 and pc2 and (pc1 != pc2) else 0.0

        # 9. Phonetic Soundex & Metaphone Similarities
        try:
            tok1 = name1.split()[0] if name1.split() else ""
            tok2 = name2.split()[0] if name2.split() else ""
            feats["first_token_soundex_match"] = 1.0 if tok1 and tok2 and (jellyfish.soundex(tok1) == jellyfish.soundex(tok2)) else 0.0
            feats["first_token_metaphone_match"] = 1.0 if tok1 and tok2 and (jellyfish.metaphone(tok1) == jellyfish.metaphone(tok2)) else 0.0
        except Exception:
            feats["first_token_soundex_match"] = 0.0
            feats["first_token_metaphone_match"] = 0.0

        # 10. Multi-Field Compound Interaction Scores
        feats["name_addr_fuzz_prod"] = feats["name_token_set_ratio"] * feats["addr_token_set_ratio"]
        feats["name_addr_jw_prod"] = feats["name_jaro_winkler"] * feats["addr_jaro_winkler"]

        return feats

    def extract_candidate_dataset_features(
        self,
        s1_records: Dict[str, Dict],
        target_records: Dict[str, Dict],
        candidate_mapping: Dict[str, List[str]],
        ground_truth: Optional[Dict[str, Set[str]]] = None
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray], List[str]]:
        """
        Builds feature matrix X and label vector y for all candidate pairs.
        """
        rows = []
        labels = []
        pair_ids = []

        for s1_id, cand_ids in candidate_mapping.items():
            s1_rec = s1_records.get(s1_id, {"entity_id": s1_id})
            true_matches = ground_truth.get(s1_id, set()) if ground_truth else None

            for t_id in cand_ids:
                t_rec = target_records.get(t_id, {"entity_id": t_id})
                feats = self.extract_pair_features(s1_rec, t_rec)
                rows.append(feats)
                pair_ids.append(f"{s1_id}::{t_id}")

                if true_matches is not None:
                    labels.append(1 if t_id in true_matches else 0)

        df_feats = pd.DataFrame(rows)
        y = np.array(labels) if labels else None
        return df_feats, y, pair_ids
