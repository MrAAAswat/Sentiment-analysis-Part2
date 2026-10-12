"""Predict sentiment for one text or a CSV batch."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from .data import ID2LABEL
from .model import TextSentimentDataset


def predict_texts(texts, model_dir: str, batch_size: int = 16, max_length: int = 128):
    if not texts:
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    dataset = TextSentimentDataset(texts, None, tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    results = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probabilities = torch.softmax(logits, dim=-1).cpu()
            for probs in probabilities:
                class_id = int(probs.argmax().item())
                result = {
                    "predicted_sentiment": ID2LABEL[class_id],
                    "confidence": float(probs[class_id].item()),
                }
                for label_id, label_name in ID2LABEL.items():
                    result[f"probability_{label_name}"] = float(probs[label_id].item())
                results.append(result)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/bert-sentiment")
    parser.add_argument("--text", help="A single text string to classify.")
    parser.add_argument("--input-csv", help="CSV to classify in batch.")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output-csv", default="reports/predictions.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    if bool(args.text) == bool(args.input_csv):
        raise ValueError("Provide exactly one of --text or --input-csv.")

    if args.text:
        print(predict_texts([args.text], args.model_dir, args.batch_size, args.max_length)[0])
        return

    frame = pd.read_csv(args.input_csv)
    if args.text_column not in frame.columns:
        raise ValueError(f"Column {args.text_column!r} not found in {args.input_csv}.")
    texts = frame[args.text_column].fillna("").astype(str).tolist()
    predictions = predict_texts(texts, args.model_dir, args.batch_size, args.max_length)
    result = pd.concat([frame.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Wrote predictions to {output_path.resolve()}")


if __name__ == "__main__":
    main()
