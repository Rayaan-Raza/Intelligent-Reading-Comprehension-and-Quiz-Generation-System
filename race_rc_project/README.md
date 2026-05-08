# RACE Reading Comprehension Project

Detailed documentation of the work completed so far, the current project state, and exact commands to run each stage.

---

## 1) Project Goal

This project builds a reading-comprehension pipeline (RACE-style multiple-choice QA) with:

- dataset loading and validation,
- leakage-safe splitting by passage id,
- preprocessing into a binary verification format,
- feature engineering using TF-IDF + handcrafted numeric/sentence features,
- placeholder model training/evaluation/inference scripts ready for expansion,
- notebooks and report artifacts for EDA and experiments.

The core modeling idea currently implemented is to convert each MCQ item into **4 binary rows** (one per option A/B/C/D), then train a classifier to predict `is_correct` for each option.

---

## 2) Current Repository Structure

```text
race_rc_project/
├─ data/
│  ├─ raw/                      # input CSV(s), e.g. val.csv
│  ├─ splits/                   # leakage-safe split CSVs
│  └─ processed/                # verification data + feature artifacts
├─ src/
│  ├─ data_loader.py
│  ├─ preprocessing.py
│  ├─ create_splits.py
│  ├─ build_processed_data.py
│  ├─ features.py
│  ├─ train_model_a.py          # placeholder
│  ├─ train_model_b.py          # placeholder
│  ├─ evaluate.py               # placeholder metric helper
│  ├─ inference.py              # placeholder inference helper
│  └─ neural_model.py           # placeholder neural class
├─ notebooks/
│  ├─ 01_eda.ipynb
│  └─ 02_experiments.ipynb
├─ report/
│  ├─ eda_summary.csv
│  └─ final_report.pdf
├─ figures/
│  ├─ answer_distribution.png
│  └─ article_length_distribution.png
├─ ui/
│  └─ app.py                    # placeholder UI entrypoint
├─ models/
│  └─ model_a/traditional/
│     └─ tfidf_vectorizer.pkl
├─ requirements.txt
└─ README.md
```

---

## 3) Work Completed So Far (Detailed)

### 3.1 Data loading and schema validation

Implemented in `src/data_loader.py`:

- defines expected schema:
  - `id`, `article`, `question`, `A`, `B`, `C`, `D`, `answer`
- safely loads split CSV files,
- drops accidental `Unnamed:` index columns,
- validates required columns,
- supports empty-file fallback with a correctly shaped empty DataFrame.

Why this matters:
- prevents silent schema mismatch bugs early,
- keeps downstream preprocessing/model code stable.

### 3.2 Text preprocessing utilities

Implemented in `src/preprocessing.py`:

- `clean_text()`: lowercase + whitespace normalization,
- `normalize_answer_label()`: enforces valid labels `A/B/C/D`,
- `split_sentences()`: regex-based sentence chunking for feature extraction,
- `validate_dataframe()`: dedupe, null filtering, required-column checks,
- `build_verification_dataset()`: expands each MCQ row into 4 option-level binary rows.

Why this matters:
- standardizes text before vectorization,
- creates supervised labels suitable for binary classification.

### 3.3 Leakage-safe splitting

Implemented in `src/create_splits.py`:

- reads source CSV (default `data/raw/val.csv`),
- cleans and validates data,
- performs split by **unique `id`** (not by row),
- split ratio: `80% train / 10% val / 10% test`,
- asserts no overlap between train/val/test ids,
- writes:
  - `data/splits/train.csv`
  - `data/splits/val.csv`
  - `data/splits/test.csv`

Why this matters:
- prevents passage leakage across splits,
- makes validation/test estimates more trustworthy.

### 3.4 Processed verification datasets

Implemented in `src/build_processed_data.py`:

- takes split CSVs,
- runs `build_verification_dataset()` per split,
- saves:
  - `data/processed/train_verification.csv`
  - `data/processed/val_verification.csv`
  - `data/processed/test_verification.csv`
- reports per-split row expansion stats.

### 3.5 Feature engineering pipeline

Implemented in `src/features.py`:

- builds base text fields (`article`, `question`, `option_text`, combined),
- fits `TfidfVectorizer` on **train only** (leakage-safe),
- transforms train/val combined text,
- computes additional numeric features:
  - article-question cosine similarity,
  - article-option cosine similarity,
  - question-option cosine similarity,
  - text lengths,
  - token overlap metrics,
  - option exact occurrence / frequency in article,
  - sentence-level best-match similarities using token-based cosine.
- concatenates sparse TF-IDF + numeric features,
- saves artifacts:
  - `models/model_a/traditional/tfidf_vectorizer.pkl`
  - `data/processed/X_train_features.npz`
  - `data/processed/X_val_features.npz`
  - `data/processed/y_train.npy`
  - `data/processed/y_val.npy`

### 3.6 EDA outputs and reports

Artifacts present:

- `report/eda_summary.csv`
- `report/final_report.pdf`
- figures in `figures/`
- notebooks:
  - `notebooks/01_eda.ipynb`
  - `notebooks/02_experiments.ipynb`

Observed EDA summary values (`report/eda_summary.csv`):

- total rows: `87,851`
- split rows: train `70,331`, val `8,784`, test `8,736`
- missing required fields: all `0`
- average lengths:
  - article: `1560.05`
  - question: `52.63`
  - option: `31.95`

### 3.7 Placeholder modules ready for next implementation

- `src/train_model_a.py`: currently creates model directory and prints placeholder text.
- `src/train_model_b.py`: same for model B.
- `src/evaluate.py`: simple accuracy helper function.
- `src/inference.py`: simple `model.predict(x)` wrapper.
- `src/neural_model.py`: basic fit/predict scaffold with fitted-state guard.
- `ui/app.py`: UI placeholder entrypoint.

---

## 4) Artifacts Already Generated

Based on current workspace state, these generated artifacts already exist:

- `models/model_a/traditional/tfidf_vectorizer.pkl`
- `data/processed/X_train_features.npz`
- `data/processed/X_val_features.npz`
- `data/processed/y_train.npy`
- `data/processed/y_val.npy`

---

## 5) Environment Setup (PowerShell, Windows)

Run these commands from the project folder (`race_rc_project`):

```powershell
# 1) move into project
cd C:\Users\rayaan\Desktop\AI_Proj\race_rc_project

# 2) create virtual environment
python -m venv .venv

# 3) activate virtual environment
.\.venv\Scripts\Activate.ps1

# 4) install dependencies
pip install -r requirements.txt
```

Optional verification:

```powershell
python --version
pip --version
```

---

## 6) End-to-End Run Commands

### Step A: Create leakage-safe splits

Prerequisite:
- raw dataset CSV exists at `data/raw/val.csv` (or update script argument if you customize path in code).

Command:

```powershell
python src/create_splits.py
```

Expected output:
- prints split creation message and shapes,
- writes `train.csv`, `val.csv`, `test.csv` to `data/splits/`.

### Step B: Build processed verification datasets

Command:

```powershell
python src/build_processed_data.py
```

Expected output:
- prints per-split conversion counts,
- writes:
  - `data/processed/train_verification.csv`
  - `data/processed/val_verification.csv`
  - `data/processed/test_verification.csv`

### Step C: Build feature artifacts

Command:

```powershell
python src/features.py
```

Expected output:
- progress logs from feature pipeline,
- saves vectorizer and `.npz/.npy` files in `models/` and `data/processed/`.

### Step D: Run model placeholders (current scaffold)

```powershell
python src/train_model_a.py
python src/train_model_b.py
```

Current behavior:
- creates model directories,
- prints placeholder training messages.

---

## 7) Run Notebooks and UI

### Jupyter notebooks

From project root:

```powershell
jupyter notebook
```

Then open:
- `notebooks/01_eda.ipynb`
- `notebooks/02_experiments.ipynb`

### UI placeholder

```powershell
python ui/app.py
```

Current behavior:
- prints placeholder message.

---

## 8) Quick Validation Commands

Use these after running the pipeline:

```powershell
# check that split files exist
dir data\splits

# check processed files
dir data\processed

# check vectorizer artifact
dir models\model_a\traditional
```

If you want to inspect shapes quickly in Python:

```powershell
python -c "import numpy as np; from scipy.sparse import load_npz; Xtr=load_npz('data/processed/X_train_features.npz'); Xv=load_npz('data/processed/X_val_features.npz'); ytr=np.load('data/processed/y_train.npy'); yv=np.load('data/processed/y_val.npy'); print('X_train',Xtr.shape,'X_val',Xv.shape,'y_train',ytr.shape,'y_val',yv.shape)"
```

---

## 9) Known Notes and Current Limitations

- `train_model_a.py` and `train_model_b.py` are placeholders (no real model fitting yet).
- evaluation and inference modules are minimal helper scaffolds.
- only train/val feature generation is currently persisted in `features.py`.
- data files are generally ignored by git (`.gitignore`) except optional placeholders.

---

## 10) Suggested Next Build Steps

1. Implement real training logic in `train_model_a.py` using saved features.
2. Add test-set feature generation and final test evaluation.
3. Persist trained model weights/checkpoints.
4. Replace `ui/app.py` placeholder with Streamlit interface for prediction.
5. Add reproducible CLI arguments for input/output paths and random seeds.

---

## 11) One-Command Pipeline (Current)

If your raw file is already in place, run:

```powershell
python src/create_splits.py
python src/build_processed_data.py
python src/features.py
```

This is the full implemented pipeline up to feature artifact generation.
