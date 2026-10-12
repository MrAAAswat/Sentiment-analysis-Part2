# Sentiment Analysis — Part 2

A reproducible sentiment-analysis pipeline built around **BERT base uncased**. It trains a three-class classifier (negative, neutral, positive), evaluates it on held-out data, saves a reusable model, and can rank movies or other topics from a CSV of text entries.

> **Part 2 goal:** turn the original Colab-oriented training script into a maintainable project with reproducible splits, meaningful evaluation, reusable inference, tests, and a separate recommendation component.

## Features

- Fine-tunes `bert-base-uncased` using Hugging Face Transformers and PyTorch.
- Supports CSV data with configurable text and sentiment columns.
- Normalizes sentiment labels to `negative`, `neutral`, and `positive`.
- Creates stratified train/validation/test splits before model training.
- Uses dynamic padding to reduce unnecessary computation.
- Reports accuracy, macro F1, per-class precision/recall/F1, and a confusion matrix.
- Selects the best checkpoint by validation macro F1.
- Saves the model and tokenizer together with `save_pretrained`.
- Includes batch prediction and a topic/movie recommendation script.
- Includes unit tests for label handling and recommendation ranking.

## Repository layout

```text
sentiment-analysis-part2/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── data/
│   └── README.md
├── models/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── src/
│   └── sentiment_analysis/
│       ├── __init__.py
│       ├── data.py
│       ├── model.py
│       ├── train.py
│       ├── predict.py
│       └── recommend.py
└── tests/
    ├── test_data.py
    └── test_recommend.py
```

## Requirements

- Python **3.9.11** is the target version for this project.
- `pip` and a virtual environment.
- Internet access the first time the pretrained tokenizer/model is downloaded.
- A CUDA-capable GPU is optional. CPU training is supported but BERT fine-tuning may be slow.
- Docker Desktop is optional; Google Colab is also suitable for GPU training.

## 1. Create a virtual environment

### Windows PowerShell

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Check the active interpreter:

```bash
python --version
```

Expected version: `Python 3.9.11` (or another compatible Python 3.9 patch release).

## 2. Prepare the dataset

Place your merged CSV at `data/combined_dataset.csv` (or provide a different path with `--csv`).

The default expected columns are:

| Column | Description | Example |
|---|---|---|
| `text` | Text to classify | `The film was excellent.` |
| `sentiment` | Ground-truth sentiment label | `positive` |

Accepted sentiment labels are case-insensitive `negative`, `neutral`, and `positive`. The training script also accepts the text aliases `neg`, `neu`, and `pos`. Numeric labels are intentionally rejected because different datasets assign different meanings to the same number. Convert numeric labels to canonical text labels only after checking the source dataset's documentation.

**Important:** do not blindly merge datasets with different label meanings. For example, a binary IMDb dataset has no neutral class. Keep its original labels and map them deliberately; do not fabricate neutral examples. Keep a source column if you want to analyze cross-domain performance later.

Data files are intentionally excluded from Git. See [`data/README.md`](data/README.md) for guidance.

## 3. Train the model

From the repository root:

```bash
python -m sentiment_analysis.train \
  --csv data/combined_dataset.csv \
  --text-column text \
  --label-column sentiment \
  --output-dir models/bert-sentiment \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --max-length 128
```

On Windows Command Prompt, put the command on one line or replace `\` line continuations with `^`.

The script:
1. validates and cleans the data;
2. creates stratified train/validation/test partitions;
3. fine-tunes BERT using the training partition;
4. selects the best epoch using validation macro F1;
5. evaluates the selected model once on the held-out test partition;
6. saves the model/tokenizer and metrics under the output directory.

Outputs include:

```text
models/bert-sentiment/
├── config.json
├── model.safetensors (or pytorch_model.bin, depending on library version)
├── tokenizer_config.json
├── vocab.txt
├── special_tokens_map.json
└── metrics.json
```

The exact weight filename depends on the installed Transformers version. Training logs and test metrics are also written to `reports/` when specified.

### Suggested first run

For a quick smoke test, use a small CSV subset and one epoch. This checks that your environment and data format work; it is **not** a valid final performance result.

## 4. Evaluate results

The test report includes:

- **Accuracy:** overall fraction of correct predictions.
- **Macro F1:** unweighted average F1 across the sentiment classes, useful when classes are imbalanced.
- **Per-class precision and recall:** shows which sentiments are missed or over-predicted.
- **Confusion matrix:** shows which labels the model confuses.

Use the validation split for model selection and keep the test split untouched until final evaluation. Do not repeatedly tune the model against test results.

## 5. Predict sentiment

### Single text

```bash
python -m sentiment_analysis.predict \
  --model-dir models/bert-sentiment \
  --text "I loved the story and the performances."
```

### CSV batch prediction

Input CSV needs a `text` column by default:

```bash
python -m sentiment_analysis.predict \
  --model-dir models/bert-sentiment \
  --input-csv data/unlabeled_reviews.csv \
  --text-column text \
  --output-csv reports/predictions.csv
```

The output contains the original columns plus `predicted_sentiment`, `confidence`, and class probabilities. `confidence` is the maximum softmax probability; it is **not guaranteed to be a calibrated probability of correctness**.

## 6. Rank movies or topics

Provide a CSV with a text column and a grouping column, such as `movie`:

```bash
python -m sentiment_analysis.recommend \
  --model-dir models/bert-sentiment \
  --input-csv data/movie_posts.csv \
  --text-column text \
  --group-column movie \
  --output-csv reports/movie_rankings.csv \
  --prior-strength 10
```

The recommender predicts each row, calculates sentiment as `P(positive) - P(negative)`, then averages by group. It applies a simple shrinkage adjustment toward the overall mean so groups with very few texts are not ranked solely by a small sample. The output includes review count and raw/adjusted scores.

This is a sentiment-based ranking, **not a guarantee of objective quality or a personalized recommender system**. A production version should also show uncertainty, consider duplicate/spam reviews, and calibrate the model before relying on probabilities.

## 7. Run tests

```bash
pytest
```

Tests cover label normalization and recommendation score behavior. Add more tests when you extend data formats, training, or inference.

## 8. Google Colab

1. Upload or clone the repository in Colab.
2. Install dependencies with `pip install -r requirements.txt`.
3. Upload/mount your dataset and update `--csv` to the actual path.
4. Run the training command above in a notebook cell, removing shell line breaks as needed.
5. Save the generated model directory to Google Drive or download it.

Training on Colab is optional; local inference can load the saved model directory without retraining.

## 9. Docker Desktop (optional)

Docker can make inference environments more consistent, but this starter project does not require Docker for training. If you add a Docker image, avoid baking private datasets or large checkpoints into the image; mount them as volumes instead. GPU access from Docker depends on your operating system and NVIDIA/container configuration.

## Design improvements over the original script

The original training code is a useful prototype, but this version addresses several issues:

- **Split before tokenization:** avoids processing the full dataset before partitioning and makes the data flow clearer.
- **Stratified train/validation/test split:** preserves class proportions when possible and reserves a final test set.
- **Explicit label validation:** unknown or missing labels fail with a useful message instead of silently becoming invalid targets.
- **Dynamic padding:** pads batches to the longest sequence in that batch instead of padding every example to the maximum length.
- **Validation macro F1:** checkpoint selection is not based only on accuracy.
- **Best checkpoint saved:** the final model is the best validation checkpoint, not necessarily the last epoch.
- **Real test evaluation:** validation and final test results are kept separate.
- **Correct artifact saving:** saves the trained model and tokenizer together; the original script saved the tokenizer but did not explicitly save the trained model weights.
- **Reusable inference and recommendation modules:** separates model predictions from group-level ranking.
- **CLI configuration:** paths and important hyperparameters can be changed without editing source code.

## Limitations and next steps

1. Inspect each source dataset's label definitions and licensing.
2. Check for duplicates and train/test leakage, especially when merging sources.
3. Compare BERT against a TF-IDF + Logistic Regression baseline.
4. Report cross-domain evaluation (for example, train on reviews and test on social posts).
5. Check calibration before using probabilities for business decisions.
6. Consider mixed precision and gradient accumulation if GPU memory is limited.
7. Add CI (GitHub Actions) and a Streamlit/FastAPI demo after the core pipeline is stable.

## License

Add a license file if you intend to publish or accept contributions. Confirm that all datasets and pretrained model assets permit your intended use.
