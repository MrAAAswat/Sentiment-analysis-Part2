"""Model and dataset helpers shared by training and inference."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

from .data import ID2LABEL, LABEL2ID


class TextSentimentDataset(Dataset):
    """Tokenizes one text per item; batch padding is handled by the collator."""

    def __init__(self, texts: List[str], labels: Optional[List[int]], tokenizer, max_length: int):
        self.texts = [str(text) for text in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
        )
        if self.labels is not None:
            item["labels"] = int(self.labels[index])
        return item


def create_tokenizer(model_name: str = "bert-base-uncased"):
    return AutoTokenizer.from_pretrained(model_name)


def create_model(model_name: str = "bert-base-uncased"):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label={index: label.upper() for index, label in ID2LABEL.items()},
        label2id={label.upper(): index for label, index in LABEL2ID.items()},
    )


def create_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer=tokenizer)
