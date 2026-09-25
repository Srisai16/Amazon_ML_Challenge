# Amazon ML Challenge 2026 - Business Entity Resolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Evaluation Metric](https://img.shields.io/badge/Metric-Macro%20F0.5-orange.svg)]()

## 📌 Executive Summary & Architecture Overview

This repository contains our end-to-end, high-performance, precision-optimized solution for the **Amazon ML Challenge 2026: Business Entity Resolution**.

The objective is to resolve real-world business entities across three noisy, unlinked sources (`Source 1` reference, `Source 2`, `Source 3`) across multi-country distributions (US, India, France) and optimize for **Macro-averaged $F_{0.5}$ Score** while maintaining a compact candidate blocking space.

```
┌────────────────────────────────────────────────────────────────────────┐
│                          RAW INPUT DATASETS                            │
│  - Source 1 (Reference)    - Source 2 (Noisy)    - Source 3 (Noisy)    │
│  Columns: entity_id, business_name, business_address, country          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 1. ADVANCED MULTI-COUNTRY NORMALIZATION & PREPROCESSING ENGINE         │
│  • Legal suffix canonicalization (Corp, Pvt Ltd, LLC, SAS, SARL, etc.) │
│  • Address parsing (street/rd, landmarks, postal codes, pin codes)    │
│  • Country-agnostic cleaning (US, India, and zero-shot France)         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. HYBRID HIGH-RECALL MULTI-STAGE BLOCKING / CANDIDATE GENERATION      │
│  • Fast BM25 / Character n-gram Inverted Indexing                      │
│  • Dense Bi-Encoder Embeddings (BGE-M3 / MiniLM) + FAISS Indexing      │
│  • Core Token / Phonetic / Landmark Anchored Blocking                   │
│  • Output: candidate_pairs.tsv (Target: >97% recall, ~15-25 cand/S1)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. HIGH-PRECISION RERANKING & MATCHING SYSTEM (TWO-TIER ENSEMBLE)      │
│  Tier A: Deep Cross-Encoder Transformer (DeBERTa-v3-base/large)        │
│  Tier B: 40+ Advanced Engineered Features + LightGBM / CatBoost Ranker │
│          (Fuzzy, Phonetic, Address Number Overlap, Token Jaccard)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. $F_{0.5}$ CALIBRATED DECISION & SINGLETON PROTECTION LOGIC          │
│  • Precision-biased optimal threshold search ($T \approx 0.65 - 0.80$) │
│  • Singleton safeguard (protecting 1.0 macro reward for 0-matches)     │
│  • Cross-source transitivity & graph consistency filtering             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 5. VALIDATED SUBMISSION OUTPUTS                                        │
│  • output/matching_results.tsv   (Final Leaderboard Predictions)       │
│  • output/candidate_pairs.tsv    (Audit Candidate Pool)                │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 👥 Team Work Distribution (4 Members)

| Member / Role | Focus Area | Key Deliverables |
|---|---|---|
| **Member 1 (Lead / ML Architect)** | Pipeline orchestration, Cross-Encoder fine-tuning, Ensembling | `src/models/`, `src/pipeline.py`, Submission validation |
| **Member 2 (Data & Blocking Specialist)** | Text cleaning, Lexical BM25 & FAISS bi-encoder candidate retrieval | `src/preprocessing/`, `src/blocking/`, `candidate_pairs.tsv` |
| **Member 3 (Feature Engineering & GBDT)** | String similarities, address parsing, LightGBM/CatBoost ranker | `src/features/`, `src/models/gbdt_ranker.py` |
| **Member 4 (Evaluation, QA & Documentation)** | Local CV ($F_{0.5}$ metric), ablation studies, final report & zip | `src/evaluation/`, `Documentation_template.md`, CI/QA |

---

## 📂 Project Directory Structure

```
Amazon_ML_Challenge/
├── README.md                           # Master project guide & architecture
├── ROADMAP.md                          # 3-Day sprint milestones & task tracking
├── Documentation_template.md           # Submission methodology report template
├── requirements.txt                    # Pinned Python dependencies
├── dataset/                            # (Place downloaded datasets here)
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── output/                             # Generated submission files
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── utils/
│   └── validate_submission.py          # Strict format validator (stdlib)
└── src/
    ├── __init__.py
    ├── config.py                       # Global configuration & hyperparameters
    ├── preprocessing/
    │   ├── __init__.py
    │   └── normalizer.py               # Robust multi-country text & address cleaners
    ├── blocking/
    │   ├── __init__.py
    │   ├── lexical_blocker.py          # BM25 / Inverted Index candidate generator
    │   ├── dense_blocker.py            # SentenceTransformers + FAISS candidate generator
    │   └── hybrid_blocker.py           # Multi-index fusion & deduplication
    ├── features/
    │   ├── __init__.py
    │   └── feature_extractor.py        # 40+ string, token, number, and phonetic features
    ├── models/
    │   ├── __init__.py
    │   ├── gbdt_ranker.py              # LightGBM / CatBoost ranking model
    │   └── cross_encoder.py            # DeBERTa-v3 pairwise match scorer
    ├── evaluation/
    │   ├── __init__.py
    │   └── metrics.py                  # Exact Macro F0.5 & singleton scoring engine
    └── pipeline.py                     # Unified end-to-end execution script
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Srisai16/Amazon_ML_Challenge.git
cd Amazon_ML_Challenge

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Validation on Baseline
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

### 3. Run End-to-End Pipeline
```bash
python -m src.pipeline --mode full --train-dir dataset/train --test-dir dataset/test --output-dir output/
```

---

## 🎯 Evaluation Metric: Macro $F_{0.5}$
Submissions are evaluated on Macro $F_{0.5}$ calculated per Source 1 entity and averaged across all Source 1 entities:
$$\text{Macro } F_{0.5} = \frac{1}{|S_1|} \sum_{e \in S_1} \frac{1.25 \times \text{Precision}_e \times \text{Recall}_e}{0.25 \times \text{Precision}_e + \text{Recall}_e}$$

- **Singletons (0 true matches)**: If predicted empty $\rightarrow 1.0$, if any match predicted $\rightarrow 0.0$.
- **Precision weight**: Precision is weighted $2\times$ over recall.
