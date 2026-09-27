"""
High-Precision Multi-Modal Feature Extraction Module for Entity Pairs.
Optimized for memory-mapped arrays and ultra-fast C-speed RapidFuzz execution.
"""

import os
import re
from typing import Dict, List, Optional, Set, Tuple
import jellyfish
import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz

from src.data import fastnorm as fn
from src.data.loader import fld

FEATURE_NAMES = [
    "blocking_score",
    "is_same_country",
    "is_source2",
    "name_len_diff",
    "name_len_ratio",
    "addr_len_diff",
    "addr_len_ratio",
    "name_fuzz_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_wratio",
    "name_jaro_winkler",
    "addr_fuzz_ratio",
    "addr_partial_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_jaro_winkler",
    "name_token_jaccard",
    "name_token_overlap",
    "addr_token_jaccard",
    "addr_token_overlap",
    "name_3gram_jaccard",
    "num_shared_numbers",
    "has_conflicting_numbers",
    "number_jaccard",
    "postal_match",
    "postal_mismatch",
    "first_word_match",
    "first_word_jw",
    "first_word_soundex",
    "name_addr_fuzz_prod",
    "name_addr_jw_prod",
]


class FeatureExtractor:
    def __init__(self):
        self.feature_names = FEATURE_NAMES

    @staticmethod
    def _jaccard(s1: Set[str], s2: Set[str]) -> float:
        if not s1 or not s2:
            return 0.0
        inter = len(s1.intersection(s2))
        union = len(s1.union(s2))
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _overlap(s1: Set[str], s2: Set[str]) -> float:
        if not s1 or not s2:
            return 0.0
        inter = len(s1.intersection(s2))
        m = min(len(s1), len(s2))
        return inter / m if m > 0 else 0.0

    @staticmethod
    def _char_3grams(text: str) -> Set[str]:
        if len(text) < 3:
            return {text} if text else set()
        return {text[i:i + 3] for i in range(len(text) - 2)}

    def extract_pair_vector(
        self,
        name1: str,
        addr1: str,
        country1: int,
        name2: str,
        addr2: str,
        country2: int,
        is_s2: float,
        block_score: float = 0.0,
    ) -> List[float]:
        """Computes similarity feature vector for a single pair of normalized strings."""
        len_n1, len_n2 = len(name1), len(name2)
        len_a1, len_a2 = len(addr1), len(addr2)

        # Country match (0 is UNKNOWN)
        if country1 != 0 and country2 != 0:
            is_same_country = 1.0 if country1 == country2 else 0.0
        else:
            is_same_country = 1.0

        # RapidFuzz string similarities
        name_fuzz = fuzz.ratio(name1, name2) / 100.0
        name_partial = fuzz.partial_ratio(name1, name2) / 100.0
        name_token_sort = fuzz.token_sort_ratio(name1, name2) / 100.0
        name_token_set = fuzz.token_set_ratio(name1, name2) / 100.0
        name_wratio = fuzz.WRatio(name1, name2) / 100.0
        name_jw = distance.JaroWinkler.similarity(name1, name2)

        addr_fuzz = fuzz.ratio(addr1, addr2) / 100.0
        addr_partial = fuzz.partial_ratio(addr1, addr2) / 100.0
        addr_token_sort = fuzz.token_sort_ratio(addr1, addr2) / 100.0
        addr_token_set = fuzz.token_set_ratio(addr1, addr2) / 100.0
        addr_jw = distance.JaroWinkler.similarity(addr1, addr2)

        # Token set metrics
        n_toks1 = set(name1.split())
        n_toks2 = set(name2.split())
        name_jacc = self._jaccard(n_toks1, n_toks2)
        name_over = self._overlap(n_toks1, n_toks2)

        a_toks1 = set(addr1.split())
        a_toks2 = set(addr2.split())
        addr_jacc = self._jaccard(a_toks1, a_toks2)
        addr_over = self._overlap(a_toks1, a_toks2)

        # 3-gram jaccard
        g1 = self._char_3grams(name1)
        g2 = self._char_3grams(name2)
        name_3g = self._jaccard(g1, g2)

        # Numbers & postal
        nums1 = {t for t in a_toks1 if t.isdigit()}
        nums2 = {t for t in a_toks2 if t.isdigit()}
        common_nums = nums1.intersection(nums2)
        num_shared = float(len(common_nums))
        has_conflict = 1.0 if (nums1 and nums2 and not common_nums) else 0.0
        num_jacc = self._jaccard(nums1, nums2)

        p1 = {t for t in nums1 if len(t) in (5, 6)}
        p2 = {t for t in nums2 if len(t) in (5, 6)}
        if p1 and p2:
            p_match = 1.0 if p1.intersection(p2) else 0.0
            p_mismatch = 1.0 - p_match
        else:
            p_match = 0.0
            p_mismatch = 0.0

        # First word
        w1 = name1.split()[0] if name1 else ""
        w2 = name2.split()[0] if name2 else ""
        first_match = 1.0 if (w1 and w2 and w1 == w2) else 0.0
        first_jw = distance.JaroWinkler.similarity(w1, w2) if (w1 and w2) else 0.0

        try:
            first_snd = 1.0 if (w1 and w2 and jellyfish.soundex(w1) == jellyfish.soundex(w2)) else 0.0
        except Exception:
            first_snd = 0.0

        # Interaction scores
        name_addr_fuzz = name_token_set * addr_token_set
        name_addr_jw = name_jw * addr_jw

        return [
            block_score,
            is_same_country,
            is_s2,
            float(abs(len_n1 - len_n2)),
            min(len_n1, len_n2) / max(len_n1, len_n2, 1),
            float(abs(len_a1 - len_a2)),
            min(len_a1, len_a2) / max(len_a1, len_a2, 1),
            name_fuzz,
            name_partial,
            name_token_sort,
            name_token_set,
            name_wratio,
            name_jw,
            addr_fuzz,
            addr_partial,
            addr_token_sort,
            addr_token_set,
            addr_jw,
            name_jacc,
            name_over,
            addr_jacc,
            addr_over,
            name_3g,
            num_shared,
            has_conflict,
            num_jacc,
            p_match,
            p_mismatch,
            first_match,
            first_jw,
            first_snd,
            name_addr_fuzz,
            name_addr_jw,
        ]

    def extract_batch_from_cache(
        self,
        cache_root: str,
        split: str,
        query_rows: np.ndarray,
        target_rows: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        """
        Fast batch feature extraction from memory-mapped cache arrays.
        """
        s1_dir = os.path.join(cache_root, split, "s1")
        s2_dir = os.path.join(cache_root, split, "s2")
        s3_dir = os.path.join(cache_root, split, "s3")

        nm1 = np.load(fld(s1_dir, "name"), mmap_mode="r")
        ad1 = np.load(fld(s1_dir, "addr"), mmap_mode="r")
        c1 = np.load(fld(s1_dir, "country"), mmap_mode="r")

        nm2 = np.load(fld(s2_dir, "name"), mmap_mode="r")
        ad2 = np.load(fld(s2_dir, "addr"), mmap_mode="r")
        c2 = np.load(fld(s2_dir, "country"), mmap_mode="r")
        n2 = len(nm2)

        nm3 = np.load(fld(s3_dir, "name"), mmap_mode="r")
        ad3 = np.load(fld(s3_dir, "addr"), mmap_mode="r")
        c3 = np.load(fld(s3_dir, "country"), mmap_mode="r")

        n_pairs = len(query_rows)
        X = np.zeros((n_pairs, len(self.feature_names)), dtype=np.float32)

        for i in range(n_pairs):
            qr = int(query_rows[i])
            tr = int(target_rows[i])
            sc = float(scores[i])

            q_name = nm1[qr].decode("utf-8", "ignore")
            q_addr = ad1[qr].decode("utf-8", "ignore")
            q_cnt = int(c1[qr])

            if tr < n2:
                is_s2 = 1.0
                t_name = nm2[tr].decode("utf-8", "ignore")
                t_addr = ad2[tr].decode("utf-8", "ignore")
                t_cnt = int(c2[tr])
            else:
                is_s2 = 0.0
                idx = tr - n2
                t_name = nm3[idx].decode("utf-8", "ignore")
                t_addr = ad3[idx].decode("utf-8", "ignore")
                t_cnt = int(c3[idx])

            X[i] = self.extract_pair_vector(
                q_name, q_addr, q_cnt,
                t_name, t_addr, t_cnt,
                is_s2, sc
            )

        return X
