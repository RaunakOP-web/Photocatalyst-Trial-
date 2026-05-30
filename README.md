# Glycerol Photocatalyst HER Prediction

## Overview

Machine learning pipeline to **predict and rank photocatalysts** for hydrogen
evolution via glycerol photoreforming. Trains XGBoost and LightGBM regressors on
an 886-row literature-mined dataset, using 17 physically meaningful features to
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

Loads the raw data, removes error-flagged and zero-HER rows, drops leakage
columns, encodes categoricals, log-transforms the target, and saves train/test
splits to `data/processed/`.

### Step 3: Train

```bash
python src/train.py
```

Trains XGBoost and LightGBM with 5-fold cross-validation. Saves all models to
`models/` and training metrics to `data/results/training_results.json`.

### Step 4: Evaluate

```bash
python src/evaluate.py
```

Generates actual-vs-predicted plots, residual plots, SHAP feature importance
bar charts, and SHAP beeswarm plots. All saved to `data/results/`.

### Step 5 (optional): Predict on new catalysts

```bash
python src/predict.py --input my_candidates.csv
```

Predicts HER for each row in the input CSV and ranks candidates from highest to
lowest predicted activity.

## Dataset

The master dataset contains **886 entries** of glycerol photoreforming
experiments mined from published literature. Each entry includes catalyst
composition, synthesis conditions, light source parameters, and the reported
hydrogen evolution rate (HER).

> **Note:** The raw dataset is gitignored because it may contain unpublished
> research data. Add it locally to `data/raw/` before running the pipeline.

## Model

| Aspect | Detail |
|---|---|
| Algorithms | XGBoost, LightGBM |
| Validation | 5-fold cross-validation |
| Target | log₁₊ₓ(HER) — log-transformed to handle skewed distribution |
| Metrics | R² (log and original scale), MAE (µmol g⁻¹ h⁻¹) |
| Explainability | SHAP TreeExplainer |

## Features Used

The model uses 17 features:

| # | Feature | Type |
|---|---|---|
| 1 | `host_material` | Categorical |
| 2 | `co_catalyst` | Categorical |
| 3 | `co_catalyst_wt_pct` | Numeric |
| 4 | `semiconductor_2` | Categorical |
| 5 | `glycerol_concentration_std` | Numeric |
| 6 | `catalyst_loading_mg` | Numeric |
| 7 | `reaction_volume_mL` | Numeric |
| 8 | `temperature_C` | Numeric |
| 9 | `pH` | Numeric |
| 10 | `light_power_W` | Numeric |
| 11 | `wavelength_cutoff_nm` | Numeric |
| 12 | `is_xe_lamp` | Binary |
| 13 | `is_hg_lamp` | Binary |
| 14 | `is_led` | Binary |
| 15 | `is_uv` | Binary |
| 16 | `is_visible_light` | Binary |
| 17 | `is_solar_simulator` | Binary |

## Results

> _Placeholder — fill in after training with your dataset._
>
> | Model | CV R² | Test R² | Test MAE (µmol/g/h) |
> |---|---|---|---|
> | XGBoost | — | — | — |
> | LightGBM | — | — | — |

## Citation

> _Placeholder — add citation when the paper is published._
