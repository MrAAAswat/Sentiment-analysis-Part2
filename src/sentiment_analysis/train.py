"""Train, select, and test a BERT sentiment classifier."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .data import ID2LABEL, load_dataset
from .model import TextSentimentDataset, create_collator, create_model, create_tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    all_labels, all_predictions = [], []
    total_loss, batches = 0.0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            total_loss += float(outputs.loss.item())
            batches += 1
            predictions = outputs.logits.argmax(dim=-1)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    return {
        "loss": total_loss / max(batches, 1),
        "accuracy": float(accuracy_score(all_labels, all_predictions)),
        "macro_f1": float(f1_score(all_labels, all_predictions, average="macro", zero_division=0)),
        "report": classification_report(
            all_labels,
            all_predictions,
            labels=[0, 1, 2],
            target_names=[ID2LABEL[i] for i in range(3)],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(all_labels, all_predictions, labels=[0, 1, 2]).tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/combined_dataset.csv")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="sentiment")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--output-dir", default="models/bert-sentiment")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 8:
        raise ValueError("epochs/batch-size must be positive and max-length must be at least 8.")
    if not 0 < args.test_size < 1 or not 0 < args.validation_size < 1:
        raise ValueError("test-size and validation-size must be between 0 and 1.")
    if args.test_size + args.validation_size >= 1:
        raise ValueError("test-size + validation-size must be less than 1.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = load_dataset(args.csv, args.text_column, args.label_column)
    counts = data["label"].value_counts()
    if len(counts) != 3 or counts.min() < 3:
        raise ValueError(
            "Need at least 3 examples for each of negative, neutral, and positive "
            "to create stratified train/validation/test splits. Check your label mapping and data."
        )

    # Split raw examples first, before any model training or learned preprocessing.
    train_val, test_df = train_test_split(
        data,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=data["label"],
    )
    relative_val_size = args.validation_size / (1.0 - args.test_size)
    train_df, val_df = train_test_split(
        train_val,
        test_size=relative_val_size,
        random_state=args.seed,
        stratify=train_val["label"],
    )
    print(f"Rows — train: {len(train_df)}, validation: {len(val_df)}, test: {len(test_df)}")

    tokenizer = create_tokenizer(args.model_name)
    collator = create_collator(tokenizer)

    def loader(frame, shuffle=False):
        dataset = TextSentimentDataset(
            frame[args.text_column].astype(str).tolist(),
            frame["label"].astype(int).tolist(),
            tokenizer,
            args.max_length,
        )
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=collator,
            pin_memory=(device.type == "cuda"),
        )

    train_loader = loader(train_df, shuffle=True)
    val_loader = loader(val_df)
    test_loader = loader(test_df)

    model = create_model(args.model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        val_metrics = evaluate(model, val_loader, device)
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(len(train_loader), 1),
            "validation_loss": val_metrics["loss"],
            "validation_accuracy": val_metrics["accuracy"],
            "validation_macro_f1": val_metrics["macro_f1"],
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, indent=2))

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)

    test_metrics = evaluate(model, test_loader, device)
    output_dir = Path(args.output_dir)
    reports_dir = Path(args.reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {
        "model_name": args.model_name,
        "seed": args.seed,
        "data_rows": len(data),
        "split_sizes": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "best_validation_macro_f1": best_f1,
        "history": history,
        "test": test_metrics,
        "label_order": [ID2LABEL[i] for i in range(3)],
        "notes": "Test metrics are for the best validation-macro-F1 checkpoint.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (reports_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nFinal held-out test metrics:")
    print(json.dumps({
        "accuracy": test_metrics["accuracy"],
        "macro_f1": test_metrics["macro_f1"],
        "confusion_matrix": test_metrics["confusion_matrix"],
    }, indent=2))
    print(f"Saved model and tokenizer to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
