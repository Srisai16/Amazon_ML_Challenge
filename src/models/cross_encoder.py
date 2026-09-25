"""
Deep Learning Cross-Encoder Module (DeBERTa-v3 / RoBERTa).
Fine-tuned pairwise classifier predicting matching probability for (S1, Target) entities.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import cfg
from src.preprocessing.normalizer import normalizer


class EntityPairDataset(Dataset):
    def __init__(self, texts_a: List[str], texts_b: List[str], labels: Optional[List[int]], tokenizer, max_length: int = 256):
        self.texts_a = texts_a
        self.texts_b = texts_b
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts_a)

    def __getitem__(self, idx):
        item = self.tokenizer(
            self.texts_a[idx],
            self.texts_b[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in item.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class CrossEncoderModel:
    def __init__(self, model_name: str = cfg.model.TRANSFORMER_MODEL_NAME, max_length: int = 256):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(self.device)

    def _prepare_text(self, record: Dict) -> str:
        name = normalizer.normalize_business_name(str(record.get("business_name", "")))
        addr = normalizer.normalize_address(str(record.get("business_address", "")))
        country = str(record.get("country", "")).strip()
        return f"{name} | {addr} | {country}"

    def predict_pairs(self, s1_records: Dict[str, Dict], target_records: Dict[str, Dict], candidate_mapping: Dict[str, List[str]], batch_size: int = 32) -> Tuple[np.ndarray, List[str]]:
        """
        Runs batch cross-encoder inference over candidate pairs.
        """
        texts_a = []
        texts_b = []
        pair_ids = []

        for s1_id, c_ids in candidate_mapping.items():
            s1_text = self._prepare_text(s1_records.get(s1_id, {}))
            for t_id in c_ids:
                t_text = self._prepare_text(target_records.get(t_id, {}))
                texts_a.append(s1_text)
                texts_b.append(t_text)
                pair_ids.append(f"{s1_id}::{t_id}")

        dataset = EntityPairDataset(texts_a, texts_b, None, self.tokenizer, max_length=self.max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[:, 1]
                all_probs.extend(probs.cpu().numpy().tolist())

        return np.array(all_probs), pair_ids
