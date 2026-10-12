"""Dataset loading, cleaning, and sentiment-label normalization."""
from __future__ import annotations

from typing import Dict

import pandas as pd

LABEL2ID: Dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL: Dict[int, str] = {value: key for key, value in LABEL2ID.items()}

# Text aliases only. Numeric labels are deliberately not guessed because
# datasets use incompatible conventions (e.g. 0 can mean negative or neutral).
LABEL_ALIASES = {
    "negative": "negative",
    "neg": "negative",
    "neutral": "neutral",
    "neu": "neutral",
    "positive": "positive",
    "pos": "positive",
}


def normalize_label(value: object) -> str:
    """Normalize a supported sentiment label or raise a useful ValueError."""
    if pd.isna(value):
        raise ValueError("Sentiment label is missing.")
    key = str(value).strip().lower()
    if key not in LABEL_ALIASES:
        raise ValueError(
            f"Unsupported sentiment label {value!r}. "
            "Expected a documented negative/neutral/positive label."
        )
    return LABEL_ALIASES[key]


def load_dataset(
    csv_path: str,
    text_column: str = "text",
    label_column: str = "sentiment",
) -> pd.DataFrame:
    """Load a CSV, validate required columns, and drop unusable text rows."""
    try:
        data = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        data = pd.read_csv(csv_path, encoding="latin1")

    missing = [column for column in (text_column, label_column) if column not in data.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. "
            f"Available columns: {list(data.columns)}"
        )

    data = data.copy()
    data[text_column] = data[text_column].astype("string")
    valid_text = data[text_column].notna() & data[text_column].str.strip().ne("")
    data = data.loc[valid_text].copy()

    if data.empty:
        raise ValueError("No rows with non-empty text remain after cleaning.")

    try:
        data["label_name"] = data[label_column].map(normalize_label)
    except ValueError as exc:
        raise ValueError(f"Could not normalize labels in column {label_column!r}: {exc}") from exc

    data["label"] = data["label_name"].map(LABEL2ID).astype("int64")
    return data.reset_index(drop=True)
