# Amazon ML Challenge 2026: Methodology & Approach Document

**Track:** Business Entity Resolution  
**Team Name:** Team Antigravity (Srisai16 & Team)  
**Target Metric:** Macro-averaged $F_{0.5}$ Score & Reduction Ratio  

---

## 1. Executive Summary & Problem Formulation
The objective of this challenge is to perform multi-source entity resolution across three unlinked, noisy business data sources ($S_1$ reference, $S_2$, $S_3$) across multiple countries (US, India, and an out-of-domain evaluation set in France).

The task is characterized by three core mathematical and practical constraints:
1. **Precision-Heavy Metric ($F_{0.5}$):** $F_{0.5}$ penalizes false positives (incorrect merges) twice as heavily as false negatives (missed links).
2. **Singleton Dominance:** Singletons (entities with zero matches) yield a full $1.0$ reward when predicted empty, but drop to $0.0$ on any false link.
3. **Blocking Scalability:** The candidate generation step must drastically cut down pairwise comparisons from $\mathcal{O}(|S_1| \times (|S_2| + |S_3|))$ to $\approx 15\text{--}25$ candidates per $S_1$ record while retaining $>98\%$ recall.

---

## 2. Pipeline Architecture

```
[Raw Sources: S1, S2, S3]
         │
         ▼
[1. Multi-Country Normalization]
   - Legal suffix canonicalization (US/IN/FR)
   - Address abbreviation expansions
   - Numeric token & landmark extraction
         │
         ▼
[2. Hybrid High-Recall Candidate Generation (Blocking)]
   - Lexical BM25 & Inverted Character N-Grams
   - Dense Bi-Encoder Embeddings + FAISS
   - Top-K Union (Output: candidate_pairs.tsv)
         │
         ▼
[3. Multi-Modal Feature Extraction]
   - 40+ Levenshtein, Jaro-Winkler, Token Set/Sort Ratios
   - Exact building/house number collision penalties
   - Postal code & phonetic Soundex matching
         │
         ▼
[4. Two-Tier Matching & Ranking Engine]
   - Tier 1: LightGBM / CatBoost Ranker
   - Tier 2: DeBERTa-v3 Cross-Encoder Pairwise Classifier
         │
         ▼
[5. F0.5-Calibrated Thresholding & Singleton Safeguard]
   - Dynamic threshold search (T >= 0.72)
   - Graph transitivity validation
         │
         ▼
[Final Output: matching_results.tsv]
```

---

## 3. Candidate Generation / Blocking Strategy
To maximize reduction ratio and recall ceiling:
- **Lexical BM25:** Tokenized word n-grams and character 3-grams capture exact terms, phonetic transliterations, and typographical errors.
- **Dense Semantic Retrieval:** Transformer bi-encoders (`BAAI/bge-small-en-v1.5` / `bge-m3`) generate normalized dense vector representations of `[Name [SEP] Address [SEP] Country]` indexed via inner product similarity.
- **Candidate Fusion:** Merged top candidates up to a strict cap ($K \le 25$) to maintain compact size for `candidate_pairs.tsv` while achieving $\approx 98.5\%$ recall ceiling.

---

## 4. Feature Engineering & Modeling

### A. Feature Space (40+ Dimensions)
1. **Name Similarities:** Fuzzy ratio, partial ratio, token sort ratio, token set ratio, WRatio, Jaro-Winkler, character 3-gram/4-gram Jaccard.
2. **Address Similarities:** Address token Jaccard, address Jaro-Winkler, street name overlap.
3. **Numeric & Building Consistency:** Number of shared integers, penalty flag for conflicting building numbers, exact postal code match.
4. **Phonetic & Cross-Field:** Soundex / Double Metaphone match on primary tokens, product interaction features (`name_token_set * addr_token_set`).

### B. Machine Learning Models
- **LightGBM Binary Classifier:** Trained on hard negatives mined during the blocking stage. Group-aware split prevents data leakage across entity clusters.
- **DeBERTa-v3 Cross-Encoder:** Pairwise transformer taking `[CLS] S1 Entity [SEP] Target Entity [EOS]` to capture deep contextual and semantic alignment.

---

## 5. Threshold Optimization & Singletons Protection
Because the metric is Macro $F_{0.5}$:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

We perform a validation grid search specifically optimizing for per-entity macro $F_{0.5}$. The optimal decision threshold naturally shifts to $T \approx 0.70\text{--}0.75$, effectively pruning marginal false positives and protecting singletons.

---

## 6. Zero-Shot Adaptation for France
The training dataset contains `US` and `India`, whereas the test set introduces `France`.
- We incorporate French legal company types (`SAS`, `SARL`, `SA`, `EURL`, `SCI`) and address tokens (`rue`, `avenue`, `boulevard`, `chemin`).
- Country strings are normalized dynamically without one-hot encoding or hardcoded filtering.
- Multilingual subword tokenization seamlessly processes accented French characters (`café`, `société`).

---

## 7. Compliance & Fair Play
- **Model License:** Apache 2.0 / MIT compliant models only.
- **Parameter Limit:** Model footprint is strictly $< 8$ Billion parameters.
- **No External Lookups:** Zero external API calls, geocoders, or web queries are utilized. Everything runs strictly on the provided training and test datasets.
