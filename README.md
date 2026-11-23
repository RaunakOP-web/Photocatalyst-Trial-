# Glycerol Photocatalyst HER Prediction

## Overview

Machine learning pipeline to **predict and rank photocatalysts** for hydrogen
evolution via glycerol photoreforming. Trains XGBoost and LightGBM regressors on
an 886-row literature-mined dataset, using physically meaningful features to
predict the hydrogen evolution rate (HER, µmol g⁻¹ h⁻¹).

## Project Structure

```
Photocatalyst-Trial-/
├── data/
│   ├── raw/                  ← drop your dataset here
│   │   └── .gitkeep
│   ├── processed/            ← cleaned/encoded data (auto-generated)
│   │   └── .gitkeep
│   └── results/              ← model outputs, predictions, plots
│       └── .gitkeep
├── notebooks/
│   ├── 01_EDA.ipynb          ← exploratory data analysis
│   ├── 02_preprocessing.ipynb
│   └── 03_training_and_eval.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocess.py         ← data cleaning and feature engineering
│   ├── train.py              ← model training
│   ├── evaluate.py           ← metrics and plots
│   └── predict.py            ← inference on new catalysts
├── models/                   ← saved model files (.joblib)
│   └── .gitkeep
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
# Clone the repo
git clone https://github.com/RaunakOP-web/Photocatalyst-Trial-.git
cd Photocatalyst-Trial-

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage — Step by Step

### Step 1: Add your dataset

Drop your master dataset (CSV, JSON, or XLSX) into `data/raw/`.

### Step 2: Preprocess

```bash
python src/preprocess.py
```

Loads the raw data, cleans out outliers and duplicates, handles missing values dynamically, normalizes strings, splits dataset using stratified sampling, and target-encodes categorical variables.

### Step 3: Train

```bash
python src/train.py
```

Performs Optuna hyperparameter optimization for LightGBM and XGBoost, performs a final fit with early stopping, trains a Ridge baseline, runs Leave-One-Material-Out CV (LOMO-CV), and saves the trained models and training results.

### Step 4: Evaluate

```bash
python src/evaluate.py
```

Generates 10 performance, residual, distribution, learning curve, and SHAP plots, and outputs consolidated metrics.

### Step 5 (optional): Predict on new catalysts

```bash
python src/predict.py --input my_candidates.csv --bootstrap_n 100
```

Predicts HER for each row in the input CSV and computes bootstrap confidence intervals.

## Dataset

The master dataset contains **886 entries** of glycerol photoreforming
experiments mined from published literature; **838** remain after
deduplication and quality filtering (`data/processed/df_clean.csv`). Each entry includes catalyst
composition, synthesis conditions, light source parameters, and the reported
hydrogen evolution rate (HER).

> **Note:** The raw dataset is gitignored because it may contain unpublished
> research data. Add it locally to `data/raw/` before running the pipeline.

## Data Quality
- `data_quality_flag`: OK = verified; NEEDS_REVIEW = plausible but unverified; LIKELY_ERROR = confirmed duplicate or impossible value — all LIKELY_ERROR rows are dropped before training.
- Duplicate `experiment_hash` rows are deduplicated by keeping the row with the highest `metadata_completeness_score`.
- Confidence columns weight each row during training: HIGH=1.0, MEDIUM=0.7, LOW=0.3.
- **Primary generalization metric in the paper is LOMO-CV R²** (leave-one-material-out), not random CV R², because TiO₂ comprises 72% of the dataset.

## Model

| Aspect | Detail |
|---|---|
| Algorithms | XGBoost, LightGBM, Ridge |
| Validation | 5-fold cross-validation & LOMO-CV |
| Target | log₁₊ₓ(HER) — log-transformed to handle skewed distribution |
| Metrics | R² (log and original scale), MAE (µmol g⁻¹ h⁻¹), RMSE |
| Explainability | SHAP TreeExplainer |

## Results

Results will be populated after training. Run `python run_all.py`.

## Citation

```bibtex
@article{pending,
  title   = {Machine Learning-Guided Discovery of High-Performance Photocatalysts
             for Glycerol Photoreforming Hydrogen Evolution},
  author  = {[authors]},
  journal = {[journal]},
  year    = {2025},
  doi     = {pending}
}
```

## Dashboard

`index.html` (deployed on Vercel) reads all of its numbers from a data block
written by `build_dashboard.py`. After re-running the pipeline, refresh it with:

```bash
python build_dashboard.py          # rewrite the data block in index.html
python build_dashboard.py --check  # only report inconsistencies between outputs
```
