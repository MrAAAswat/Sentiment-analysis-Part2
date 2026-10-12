"""Rank movies/topics using predicted sentiment from their text entries."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .predict import predict_texts


def rank_groups(
    frame: pd.DataFrame,
    group_column: str,
    predictions: list[dict],
    prior_strength: float = 10.0,
) -> pd.DataFrame:
    """Aggregate sentiment and shrink low-volume groups toward the global mean."""
    if group_column not in frame.columns:
        raise ValueError(f"Group column {group_column!r} not found.")
    if prior_strength < 0:
        raise ValueError("prior_strength must be non-negative.")
    if len(frame) != len(predictions):
        raise ValueError("Each input row must have exactly one prediction.")

    work = frame[[group_column]].copy()
    work["probability_positive"] = [p["probability_positive"] for p in predictions]
    work["probability_negative"] = [p["probability_negative"] for p in predictions]
    work["sentiment_score"] = work["probability_positive"] - work["probability_negative"]
    work = work.dropna(subset=[group_column])
    if work.empty:
        raise ValueError("No rows have a valid group value.")

    global_prior = float(work["sentiment_score"].mean())
    grouped = work.groupby(group_column, dropna=True)["sentiment_score"].agg(["mean", "count"])
    grouped = grouped.rename(columns={"mean": "raw_sentiment_score", "count": "review_count"})
    grouped["adjusted_sentiment_score"] = (
        grouped["review_count"] / (grouped["review_count"] + prior_strength)
        * grouped["raw_sentiment_score"]
        + prior_strength / (grouped["review_count"] + prior_strength) * global_prior
    )
    grouped["global_prior"] = global_prior
    return grouped.reset_index().sort_values(
        ["adjusted_sentiment_score", "review_count"],
        ascending=[False, False],
    ).reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/bert-sentiment")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--group-column", default="movie")
    parser.add_argument("--output-csv", default="reports/recommendations.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--prior-strength", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()
    frame = pd.read_csv(args.input_csv)
    for column in (args.text_column, args.group_column):
        if column not in frame.columns:
            raise ValueError(f"Required column {column!r} not found in {args.input_csv}.")

    predictions = predict_texts(
        frame[args.text_column].fillna("").astype(str).tolist(),
        args.model_dir,
        args.batch_size,
        args.max_length,
    )
    ranked = rank_groups(frame, args.group_column, predictions, args.prior_strength)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output_path, index=False)
    print(ranked.to_string(index=False))
    print(f"\nWrote recommendations to {output_path.resolve()}")


if __name__ == "__main__":
    main()
