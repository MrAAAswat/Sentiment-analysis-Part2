# Dataset instructions

Place your merged dataset here, for example `combined_dataset.csv`.

Required columns (configurable in the CLI):
- `text`: text content
- `sentiment`: ground-truth label

The training script supports the canonical labels `negative`, `neutral`, and `positive` plus the explicit aliases documented in `src/sentiment_analysis/data.py`.

Before merging sources:
1. Check each dataset's label definitions and license.
2. Convert labels deliberately to the canonical scheme.
3. Do not map binary labels to three classes without a defensible rule.
4. Preserve a source identifier if you want to measure cross-domain performance.
5. Check for duplicate and near-duplicate texts before splitting.

Dataset files are excluded from Git by `.gitignore`; do not commit data you do not have permission to redistribute.
