# Sentiment Analysis — Part 2

A reproducible sentiment-analysis pipeline built around **BERT base uncased**. It trains a three-class classifier (negative, neutral, positive), evaluates it on held-out data, saves a reusable model, and ranks movies or other topics from a CSV of text entries.

> **Part 2 goal:** turn the original Colab-oriented training script (Part 1) into a maintainable project with reproducible splits, meaningful evaluation, reusable inference, tests, and a separate recommendation component.

> **Status:** Part 2 is a refactored scaffold. It has been syntax-checked for Python 3.9, but the full train → save → load → predict workflow and the test suite have not yet been verified end to end, and no Part 2 benchmark results are published. See the [roadmap](#roadmap).

Original project (*Big Data Mining* final project): Jorge Alberto Zuniga Lopez and Abdulahad Aswat.

## Contents

- [Overview](#overview)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [1. Create a virtual environment](#1-create-a-virtual-environment)
- [2. Prepare the dataset](#2-prepare-the-dataset)
- [3. Train the model](#3-train-the-model)
- [4. Evaluate results](#4-evaluate-results)
- [5. Predict sentiment](#5-predict-sentiment)
- [6. Rank movies or topics](#6-rank-movies-or-topics)
- [7. Run tests](#7-run-tests)
- [8. Google Colab](#8-google-colab)
- [9. Docker Desktop (optional)](#9-docker-desktop-optional)
- [Datasets](#datasets)
- [Part 1 vs Part 2](#part-1-vs-part-2)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [References](#references)
- [License](#license)

## Overview

Sentiment predictions are not the end goal. They feed a higher-level analysis that compares topics: classify each text, group the results by a topic column (for example a movie name), and rank the topics.

```
labelled CSV ──► clean + normalize labels ──► stratified train/val/test split
      │
      ▼
 fine-tune BERT (AdamW, dynamic padding) ──► best checkpoint by validation macro F1
      │
      ▼
 test-set report + saved model/tokenizer ──► predict (text or CSV) ──► rank topics
```

**Features**

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

## Quick start

```bash
git clone https://github.com/MrAAAswat/Sentiment-analysis-Part2.git
cd Sentiment-analysis-Part2

python3.9 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# data/combined_dataset.csv needs `text` and `sentiment` columns
python -m sentiment_analysis.train --csv data/combined_dataset.csv --output-dir models/bert-sentiment
python -m sentiment_analysis.predict --model-dir models/bert-sentiment --text "I loved the story and the performances."
```

Windows users: see the [PowerShell setup](#windows-powershell) below.

## Repository layout

```text
sentiment-analysis-part2/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── data/
│   └── README.md              # dataset guidance (data files are not committed)
├── models/
│   └── .gitkeep               # saved models go here (not committed)
├── reports/
│   └── .gitkeep               # metrics and prediction outputs
├── src/
│   └── sentiment_analysis/
│       ├── __init__.py
│       ├── data.py            # label/text normalization, data preparation
│       ├── model.py           # tokenizer, dataset, dynamic-padding collation
│       ├── train.py           # fine-tuning, checkpoint selection, evaluation, saving
│       ├── predict.py         # single-text and CSV prediction
│       └── recommend.py       # topic-level aggregation and ranking
└── tests/
    ├── test_data.py
    └── test_recommend.py
```

## Requirements

- Python **3.9.11** is the target version for this project.
- `pip` and a virtual environment.
- Internet access the first time the pretrained tokenizer/model is downloaded.
- A CUDA-capable GPU is optional. CPU training is supported, but BERT fine-tuning may be slow.
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

| Column | Description | Example |
|---|---|---|
| `text` | Text to classify | `The film was excellent.` |
| `sentiment` | Ground-truth sentiment label | `positive` |

To produce recommendations you additionally need a **topic column** (for example `movie`) in the CSV you rank. See [Rank movies or topics](#6-rank-movies-or-topics).

**Label rules**

- Accepted labels are case-insensitive `negative`, `neutral`, and `positive`, plus the aliases `neg`, `neu`, and `pos`.
- Rows with missing or empty text are removed.
- **Numeric labels are rejected on purpose**, because different datasets give the same number different meanings. Convert them to text labels only after checking the source dataset's documentation.

**Important:** do not blindly merge datasets with different label meanings. A binary IMDb dataset, for example, has no neutral class. Keep its labels and map them deliberately; do not fabricate neutral examples. Keep a `source` column if you want to analyze cross-domain performance later.

Data files are intentionally excluded from Git. See [`data/README.md`](data/README.md) and the [Datasets](#datasets) section.

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

On Windows Command Prompt, put the command on one line or replace `\` with `^`.

| Option | Meaning | Value used above |
|---|---|---|
| `--csv` | Path to the labelled CSV | `data/combined_dataset.csv` |
| `--text-column` | Column holding the text | `text` |
| `--label-column` | Column holding the sentiment label | `sentiment` |
| `--output-dir` | Where the model, tokenizer and metrics are saved | `models/bert-sentiment` |
| `--epochs` | Number of training epochs | `3` |
| `--batch-size` | Batch size | `16` |
| `--learning-rate` | AdamW learning rate | `2e-5` |
| `--max-length` | Maximum token length (longer texts are truncated) | `128` |

The script:

1. validates and cleans the data;
2. creates stratified train/validation/test partitions;
3. fine-tunes BERT using the training partition;
4. selects the best epoch using validation macro F1;
5. evaluates the selected model once on the held-out test partition;
6. saves the model/tokenizer and metrics under the output directory.

Outputs:

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

**Suggested first run:** use a small CSV subset and one epoch. This checks that your environment and data format work; it is **not** a valid final performance result.

## 4. Evaluate results

The test report includes:

- **Accuracy:** overall fraction of correct predictions.
- **Macro F1:** unweighted average F1 across classes. It is more informative than accuracy when classes are imbalanced.
- **Per-class precision and recall:** shows which sentiments are missed or over-predicted.
- **Confusion matrix:** shows which labels the model confuses.

Use the validation split for model selection and keep the test split untouched until final evaluation. Do not repeatedly tune against test results.

**Results:** Part 2 results will be added here once the pipeline has been run on a documented dataset. When you add them, record the dataset and version, split ratios, random seed, checkpoint, hyperparameters, package versions and hardware so the numbers can be reproduced.

| Model | Dataset | Accuracy | Macro F1 |
|---|---|---|---|
| BERT-base-uncased | _pending_ | _pending_ | _pending_ |

> Part 1 training logs and benchmark screenshots came from undocumented splits and metrics. Do not compare them directly with Part 2 numbers.

## 5. Predict sentiment

### Single text

```bash
python -m sentiment_analysis.predict \
  --model-dir models/bert-sentiment \
  --text "I loved the story and the performances."
```

### CSV batch prediction

The input CSV needs a `text` column by default:

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

**How the score works**

1. Each text gets a polarity: `P(positive) − P(negative)`, in the range −1 to 1.
2. Polarity is averaged per group.
3. Each group average is **shrunk toward the overall mean**, with `--prior-strength` controlling how strongly. Groups with very few texts are therefore not ranked on a tiny sample.

The output includes review count and raw and adjusted scores.

This is a sentiment-based ranking, **not a guarantee of objective quality or a personalized recommender**. Topic names must also be consistent: `Star Wars` and `star wars ` would be counted as two groups. A production version should show uncertainty, handle duplicate and spam reviews, and calibrate the model before relying on probabilities.

## 7. Run tests

```bash
pytest
```

Tests cover label normalization and recommendation score behavior. Add more tests when you extend data formats, training, or inference.

## 8. Google Colab

1. Upload or clone the repository in Colab.
2. Install dependencies with `pip install -r requirements.txt`.
3. Upload or mount your dataset and update `--csv` to the actual path.
4. Run the training command above in a notebook cell, removing shell line breaks as needed.
5. Save the generated model directory to Google Drive or download it.

Training on Colab is optional; local inference can load the saved model directory without retraining.

## 9. Docker Desktop (optional)

Docker can make inference environments more consistent, but this project does not require it for training. If you add a Docker image, avoid baking private datasets or large checkpoints into it; mount them as volumes instead. GPU access from Docker depends on your operating system and NVIDIA/container configuration.

## Datasets

Part 1 used the sources below. The counts are approximate figures from the original report, not verified against the files in this repository.

| Dataset | Notes |
|---|---|
| Sentiment Analysis Dataset | General sentiment statements, about 3,500 entries |
| Twitter and Reddit Sentiment Analysis | Social-media source; exact version and count not recorded |
| IMDb Movie Review NLP | Movie reviews, about 10,000 entries |

**Proposed addition: airline tweets.** A 14,640-record dataset labelled positive, negative or neutral would add a service-feedback domain (see the [PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC7514423/)). It is probably the US Airline Twitter Sentiment dataset, but that is **unconfirmed** until checked against the article's references. Reported class counts are positive 2,363, negative 9,178, neutral 3,099. That imbalance (about 63% negative) makes macro F1 and per-class metrics essential.

Before adding any dataset:

- Confirm its name, source, licence and exact column names.
- Map its labels to `negative` / `neutral` / `positive` explicitly.
- Remove empty text and inspect duplicates.
- Keep a `source` field so each domain can be evaluated separately.
- Use stratified splits, and consider group- or time-aware splits if user or time identifiers exist.
- Check licensing and privacy terms before redistributing anything.

## Part 1 vs Part 2

The original script was a useful prototype. This version changes the following:

| Area | Part 1 | Part 2 |
|---|---|---|
| Data processing | Inside the training script | Dedicated `data` module; explicit label validation (unknown or missing labels fail with a clear message) |
| Tokenization | Pre-tokenized and padded up front | Split before tokenization; dynamic padding per batch |
| Splits | 80/20 train/validation, no test set | Stratified train/validation/test |
| Checkpoint selection | Validation accuracy | Validation macro F1; best checkpoint saved, not just the last epoch |
| Evaluation | Accuracy-oriented | Report, per-class metrics, confusion matrix, JSON; validation and test kept separate |
| Persistence | Tokenizer saved, trained weights were not | Model and tokenizer saved together |
| Inference | Console menu | Separate single-text and CSV prediction with probabilities |
| Recommendations | Weighted sentiment and confidence | Polarity score with shrinkage and group counts |
| Configuration | Edit source code | CLI arguments |

## Limitations

- Three classes oversimplify sentiment. Sarcasm, mixed opinions and slang cause errors.
- Performance on one domain (movie reviews) does not automatically transfer to another (airline tweets).
- Softmax probabilities are not calibrated confidence estimates.
- Topic grouping depends on accurate, consistent entity names.
- Sentiment ranking is not a full recommender: recency, spam, duplicates, review volume and user context are ignored.
- The model is English-only. Chinese or multilingual text needs a suitable checkpoint (for example XLM-R).

## Roadmap

- [ ] Run `pytest` and verify the full train → save → load → predict workflow
- [ ] Publish reproducible Part 2 metrics
- [ ] Check duplicates and train/test leakage, especially when merging sources
- [ ] Add TF-IDF + Logistic Regression and Linear SVM baselines on identical splits
- [ ] Compare BERT-base-uncased with DistilBERT (RoBERTa and DeBERTa if compute allows)
- [ ] Integrate the airline dataset and run cross-domain evaluation (for example, train without airline data and test on it)
- [ ] Compare natural class distribution with class-weighted loss or balanced sampling, chosen by validation macro F1
- [ ] Evaluate probability calibration
- [ ] Compare raw average polarity with shrinkage-adjusted rankings at different group sizes
- [ ] Add mixed precision and gradient accumulation for limited GPU memory
- [ ] Add CI (GitHub Actions) and a Streamlit/FastAPI demo

Later ideas: entity normalization, aspect-based sentiment (delays, baggage, staff), multilingual support.

## References

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT.
- [PyTorch documentation](https://pytorch.org/docs/)
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/)

## License

Released under the [MIT License](LICENSE). Datasets keep their own licences.
