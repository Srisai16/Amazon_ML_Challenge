"""
Global Configuration and Hyperparameters for Amazon ML Challenge 2026.
"""

from dataclasses import dataclass, field
import os
from typing import List


@dataclass
class PathConfig:
    # Root paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR: str = os.path.join(BASE_DIR, "dataset")
    TRAIN_DIR: str = os.path.join(DATASET_DIR, "train")
    TEST_DIR: str = os.path.join(DATASET_DIR, "test")
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "output")
    MODELS_DIR: str = os.path.join(BASE_DIR, "saved_models")

    # Output files
    MATCHING_RESULTS_PATH: str = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    CANDIDATE_PAIRS_PATH: str = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")


@dataclass
class BlockingConfig:
    TOP_K_BM25: int = 15
    TOP_K_DENSE: int = 15
    MAX_CANDIDATES_PER_SOURCE1: int = 25
    MIN_SIMILARITY_THRESHOLD: float = 0.25
    DENSE_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"  # Lightweight, high performance
    MULTILINGUAL_MODEL_NAME: str = "BAAI/bge-m3"     # Optional for multilingual France support


@dataclass
class ModelConfig:
    # GBDT
    GBDT_N_ESTIMATORS: int = 400
    GBDT_LEARNING_RATE: float = 0.05
    GBDT_MAX_DEPTH: int = 6
    GBDT_NUM_LEAVES: int = 31

    # Cross Encoder
    TRANSFORMER_MODEL_NAME: str = "microsoft/deberta-v3-small"
    BATCH_SIZE: int = 32
    MAX_LENGTH: int = 256
    EPOCHS: int = 3
    LEARNING_RATE: float = 2e-5

    # Decision Threshold for Macro F0.5
    MATCH_THRESHOLD: float = 0.72  # High threshold prioritizing precision for F0.5
    COUNTRY_THRESHOLDS: dict = field(default_factory=lambda: {
        "US": 0.72,
        "India": 0.70,
        "France": 0.74,
        "default": 0.72
    })


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    SEED: int = 42


# Default global instance
cfg = Config()
