# RACE Reading Comprehension Project

End-to-end documentation: goals, implemented phases (data → features → supervised traditional ML → neural MLP → unsupervised clustering → question generation), artifacts, and exact commands.

---

## 1) Project goal

Build a **RACE-style** reading comprehension pipeline where each multiple-choice item is turned into **four binary verification rows** (one per option A–D). Models predict whether each option is **correct (1)** or **wrong (0)**. Evaluation includes:

- **Binary metrics** on all option rows: accuracy, macro F1, precision, recall, confusion matrix.
- **MCQ exact match**: among the four scored options for one question, pick the highest-scoring label and compare to the gold **A/B/C/D**.

The feature stack combines **TF-IDF** on concatenated article + question + option text with **handcrafted numeric features** (similarities, overlaps, sentence-level cues).

---

## 2) Repository structure (current)

```text
race_rc_project/
├─ data/
│  ├─ raw/                      # source CSV (e.g. val.csv)
│  ├─ splits/                   # leakage-safe train/val/test CSVs
│  └─ processed/                # verification CSVs + sparse feature matrices
├─ models/
│  └─ model_a/
│     ├─ traditional/         # TF-IDF vectorizer, sklearn models, K-Means
│     └─ neural/              # PyTorch MLP checkpoint
├─ reports/                     # CSV metrics (gitignored if generated)
├─ figures/                     # confusion matrices, etc. (gitignored if generated)
├─ scripts/
│  └─ test_module8_question_generation.py   # manual tests for question generation
├─ src/
│  ├─ data_loader.py            # RACE schema load/validate
│  ├─ preprocessing.py          # clean_text, verification expansion, sample_id
│  ├─ create_splits.py          # split by passage id
│  ├─ build_processed_data.py  # writes *\_verification.csv per split
│  ├─ features.py               # TF-IDF + numeric features, saves .npz/.npy
│  ├─ train_model_a.py          # Phase 5: traditional ML + metrics + artifacts
│  ├─ train_neural_model.py     # Phase 6: MLP on same features
│  ├─ train_unsupervised.py     # Phase 7: K-Means + silhouette + purity
│  ├─ model_a_question_generation.py   # Phase 8: RACE vs custom question modes
│  ├─ question_ranking.py       # Rank/check with sklearn, MLP, or ensemble
│  ├─ evaluate_test.py          # Held-out test metrics for saved sklearn models
│  ├─ evaluate_neural_test.py   # Held-out test metrics for saved MLP model
│  ├─ build_model_a_comparison.py # One table: traditional + neural test comparison
│  ├─ inference.py              # predict + question generation + ranking exports
│  ├─ train_model_b.py          # placeholder (Model B)
│  ├─ evaluate.py               # small helpers
│  └─ neural_model.py           # legacy placeholder class (unused by train_neural_model)
├─ notebooks/
├─ report/
├─ ui/
├─ requirements.txt
└─ README.md
```

---

## 3) Work completed (detailed)

### 3.1 Data loading (`src/data_loader.py`)

- Expected columns: `id`, `article`, `question`, `A`, `B`, `C`, `D`, `answer`.
- Drops stray `Unnamed:` index columns; validates schema; tolerates empty files.

### 3.2 Preprocessing (`src/preprocessing.py`)

- **`clean_text`**, **`normalize_answer_label`**, **`split_sentences`**, **`validate_dataframe`**.
- **`build_verification_dataset()`** expands each MCQ row into **four rows** with:
  - **`sample_id`**: `{id}__{original_row_index}` so MCQ grouping does not collapse distinct questions that share the same `(id, question)` text.
  - **`label`** / **`is_correct`**: `1` if option letter equals gold answer, else `0`.
  - **`text`**: `article [QUESTION] question [OPTION] option_text` (for reference/analysis).
  - **`model_input`**: backward-compatible `article [SEP] question [SEP] option_text` used by the feature pipeline.

### 3.3 Leakage-safe splits (`src/create_splits.py`)

- Split by **unique `id`** (~80% / 10% / 10%), assert **no id overlap** across train/val/test.
- Outputs: `data/splits/train.csv`, `val.csv`, `test.csv`.

### 3.4 Processed verification CSVs (`src/build_processed_data.py`)

- Runs `build_verification_dataset()` per split.
- Outputs: `data/processed/train_verification.csv`, `val_verification.csv`, `test_verification.csv`.

### 3.5 Feature pipeline (`src/features.py`)

- Fits **`TfidfVectorizer` on train only** on combined article+question+option string (plus `[SEP]`-style plumbing in code).
- Numeric features include: pairwise TF-IDF cosines (article–question, article–option, question–option), lengths, token overlaps, option-in-article checks, and **sentence-level** token similarity features (best sentence vs option/question, etc.).
- Horizontally stacks **sparse TF-IDF + sparse numeric block**.
- Saves:
  - `models/model_a/traditional/tfidf_vectorizer.pkl`
  - `data/processed/X_train_features.npz`, `X_val_features.npz`
  - `data/processed/y_train.npy`, `y_val.npy`
  - If `data/processed/test_verification.csv` exists: **`X_test_features.npz`**, **`y_test.npy`** (vectorizer **fit on train only**; test is **transform-only** — no leakage).

**Inference helper:** `transform_verification_dataframe(df, vectorizer)` builds the same sparse feature rows as val/test for arbitrary verification-shaped tables (used by question ranking).

**Regeneration:** after changing preprocessing or splits, rerun **`build_processed_data.py`** then **`features.py`** so `sample_id` and labels stay aligned with features.

### 3.6 Phase 5 — Model A: traditional ML (`src/train_model_a.py`)

Trains on saved sparse features (CPU for sklearn). **MCQ exact match** groups by **`sample_id`** when present.

**Models implemented:**

| Variant | Notes |
|--------|--------|
| Logistic Regression | `class_weight=None` and `balanced` (saved as `logreg_model.pkl` / `logreg_model_unweighted.pkl`) |
| Linear SVM | `class_weight=None` and `balanced` (`svm_model.pkl` / `svm_model_unweighted.pkl`) |
| Multinomial Naive Bayes | `naive_bayes_model.pkl` |
| Random Forest | `random_forest_model.pkl` |
| XGBoost | `xgboost_model.pkl` (uses GPU tree method when CUDA + GPU build available) |

**Metrics:** binary accuracy, macro F1, precision, recall; MCQ exact match; confusion matrices for all variants.

**Outputs:**

- `models/model_a/traditional/*.pkl` (see above)
- `reports/model_a_results.csv`
- `figures/model_a_confusion_matrix.png`

### 3.7 Phase 6 — Neural MLP (`src/train_neural_model.py`)

- Same **`X_*` / `y_*`** tensors as Phase 5.
- **PyTorch** MLP: dense layers, **ReLU**, **Dropout**, **LayerNorm** on hidden units, **sigmoid** via **`BCEWithLogitsLoss`**.
- Training uses **`AdamW`**, optional **`pos_weight`** for class imbalance, **`ReduceLROnPlateau`**, early stopping, gradient clipping.
- Logs **weighted** loss (optimization) and **unweighted** validation loss (`val_u`) for comparison with older ~0.56-scale curves.
- **GPU:** uses CUDA when `torch.cuda.is_available()`. **Use Python 3.10/3.11** with a **CUDA wheel** for PyTorch (see §5); Python 3.14 often has **no** GPU `torch` wheels yet.

**Outputs:**

- `models/model_a/neural/mlp_model.pt` (state dict + hyperparameters + `input_dim`)
- `reports/neural_results.csv`

### 3.8 Phase 7 — Unsupervised / semi-supervised (`src/train_unsupervised.py`)

- **`MiniBatchKMeans`** on **training features** (labels **not** used in `.fit`).
- **Silhouette score** on a **subsample** (default 8000 points) for tractability.
- **Cluster purity** vs **`is_correct`**: weighted fraction of points matching the **majority class per cluster** (diagnostic: do “correct” vs “wrong” options separate in Euclidean cluster space?).
- Appends rows from `reports/model_a_results.csv` and `reports/neural_results.csv` when present for **comparison with supervised** metrics.

**Outputs:**

- `models/model_a/traditional/kmeans_model.pkl`
- `reports/unsupervised_results.csv`

### 3.9 Phase 8 — Question generation (`src/model_a_question_generation.py`, `src/inference.py`)

Two modes:

1. **RACE sample mode:** use the **original dataset question** (reliable).
2. **Custom passage mode:** split sentences → **TF-IDF across sentences** to pick salient sentence(s) → extract answer span (with heuristics such as **“is located in \<place\>”**) → **template** question (Where / When / generic cloze, etc.).

**Additional:**

- **`enumerate_candidate_questions(passage, top_k)`** — multiple template questions from different sentences (TF-IDF ranked).
- **`src/question_ranking.py`** — ranks candidates and scores A/B/C/D options by treating each row as a verification sample using the same fitted TF-IDF (`tfidf_vectorizer.pkl`) plus a trained verifier. Supported modes: **Random Forest**, **Linear SVM**, **Logistic Regression**, **MLP Neural**, and **ensemble**.

**Exports:** `inference.py` re-exports generation + ranking helpers and keeps `run_inference(model, x)`.

**Manual test script:** `scripts/test_module8_question_generation.py` (demos Mode 1, Mode 2, unified API; optional passage via CLI args).

### 3.10 Held-out test evaluation (`src/evaluate_test.py`, `src/evaluate_neural_test.py`)

After generating **`X_test_features.npz`** / **`y_test.npy`** with `features.py`, run:

```powershell
python src/evaluate_test.py
python src/evaluate_neural_test.py
python src/build_model_a_comparison.py
```

Writes:
- **`reports/test_evaluation.csv`** for sklearn models
- **`reports/test_evaluation_neural.csv`** for MLP neural model
- **`reports/model_a_comparison_table.csv`** for one combined Model A comparison table

### 3.11 Notebooks, UI, Model B

- **Notebooks** (`notebooks/`) may still be experiments; core training paths are the `src/train_*.py` scripts above.
- **UI (`ui/app.py`)** now includes a quiz checker with verifier selection: Logistic Regression / Linear SVM / Random Forest / MLP Neural Network / Ensemble.
- **`train_model_b.py`** remains a placeholder unless you extend it.

### 3.12 `.gitignore`

Ignores large/generated artifacts (e.g. `data/**`, `.venv/`, `models/**`, `reports/**`, `figures/**`, Python caches) while allowing directory placeholders where configured. Adjust if you need to commit specific small artifacts.

---

## 4) Generated artifacts (reference)

After a full train pipeline you typically have (paths relative to `race_rc_project`):

**Features**

- `models/model_a/traditional/tfidf_vectorizer.pkl`
- `data/processed/X_train_features.npz`, `X_val_features.npz`, `y_train.npy`, `y_val.npy`
- When test split exists: **`X_test_features.npz`**, **`y_test.npy`**

**Phase 5**

- `models/model_a/traditional/logreg_model.pkl`, `logreg_model_unweighted.pkl`, `svm_model.pkl`, `svm_model_unweighted.pkl`, `naive_bayes_model.pkl`, `random_forest_model.pkl`, `xgboost_model.pkl`
- `reports/model_a_results.csv`
- `figures/model_a_confusion_matrix.png`

**Phase 6**

- `models/model_a/neural/mlp_model.pt`
- `reports/neural_results.csv`

**Phase 7**

- `models/model_a/traditional/kmeans_model.pkl`
- `reports/unsupervised_results.csv`

**Processed tables**

- `data/processed/train_verification.csv`, `val_verification.csv`, `test_verification.csv`

---

## 5) Environment setup (Windows PowerShell)

```powershell
cd C:\Users\rayaan\Desktop\AI_Proj\race_rc_project

# Recommended: Python 3.10 or 3.11 for GPU PyTorch wheels (avoid 3.14 for CUDA torch until supported).
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**PyTorch with CUDA (optional, for Phase 6 GPU):** install the wheel that matches your CUDA stack from [PyTorch](https://pytorch.org/get-started/locally/), e.g. cu121:

```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**Note:** `scikit-learn` Phase 5 models run on **CPU** even if CUDA is available; only **PyTorch** / **XGBoost** (if built with GPU) may use the GPU.

---

## 6) End-to-end commands

### A) Splits

Prerequisite: raw CSV (default path expected by `create_splits.py`, often `data/raw/val.csv`).

```powershell
python src/create_splits.py
```

### B) Verification CSVs

```powershell
python src/build_processed_data.py
```

### C) Features

```powershell
python src/features.py
```

### D) Phase 5 — Traditional Model A

```powershell
python src/train_model_a.py
```

### E) Phase 6 — Neural MLP

```powershell
python src/train_neural_model.py
```

### F) Phase 7 — K-Means

```powershell
python src/train_unsupervised.py
```

### G) Phase 8 — Test question generation

```powershell
python scripts/test_module8_question_generation.py
python scripts/test_module8_question_generation.py "Your own passage text here."
```

### H) Held-out **test** metrics (after training + `features.py` has built test matrices)

Requires `data/processed/X_test_features.npz`, `y_test.npy`, and `test_verification.csv`.

```powershell
python src/evaluate_test.py
python src/evaluate_neural_test.py
python src/build_model_a_comparison.py
```

Writes `reports/test_evaluation.csv`, `reports/test_evaluation_neural.csv`, and `reports/model_a_comparison_table.csv`.

### I) Rank generated question candidates (SVM / RF / LogReg / MLP / Ensemble)

Requires Phase 5 pickles + `tfidf_vectorizer.pkl`. Example from Python:

```powershell
python -c "from src.question_ranking import generate_and_rank_questions; print(generate_and_rank_questions('Paris is the capital of France. The Louvre is a famous museum.', ranker='random_forest')[:2])"
python -c "from src.question_ranking import generate_and_rank_questions; print(generate_and_rank_questions('Paris is the capital of France. The Louvre is a famous museum.', ranker='mlp_neural')[:2])"
python -c "from src.question_ranking import generate_and_rank_questions; print(generate_and_rank_questions('Paris is the capital of France. The Louvre is a famous museum.', ranker='ensemble')[:2])"
```

---

## 7) One-shot pipeline (features + all train scripts)

```powershell
python src/create_splits.py
python src/build_processed_data.py
python src/features.py
python src/train_model_a.py
python src/train_neural_model.py
python src/train_unsupervised.py
```

Run Phase 8 tests separately (`scripts/test_module8_question_generation.py`). After models are trained, run **`python src/evaluate_test.py`**, **`python src/evaluate_neural_test.py`**, and optionally **`python src/build_model_a_comparison.py`**.

---

## 8) Quick validation

```powershell
dir data\splits
dir data\processed
dir models\model_a\traditional
dir models\model_a\neural
dir reports
```

Feature shapes:

```powershell
python -c "import numpy as np; from scipy.sparse import load_npz; Xtr=load_npz('data/processed/X_train_features.npz'); print('X_train', Xtr.shape)"
```

If test features were built:

```powershell
python -c "from scipy.sparse import load_npz; import numpy as np; Xt=load_npz('data/processed/X_test_features.npz'); yt=np.load('data/processed/y_test.npy'); print('X_test', Xt.shape, 'y_test', yt.shape)"
```

---

## 9) Known limitations / notes

- **Model artifact compatibility:** if old sklearn pickles were serialized under a different sklearn version, some loaders may warn or skip. Re-train in current env for clean runs.
- **Pickle compatibility:** Reinstall / match **scikit-learn** version if loading older `*.pkl` models raises errors (see sklearn persistence docs).
- **Git:** large binaries and reports are often gitignored; reproduce artifacts via the commands in §6.

---

## 10) Suggested extensions

1. **Improve UI experience:** add richer visual comparison between single-model and ensemble scoring.
2. **Neural calibration:** tune thresholding and calibration for MLP probability outputs.
3. **Model B** implementation in `train_model_b.py` if required by the brief.

---

This README reflects the implemented pipeline through **Phase 8** (question generation), **test features**, and **classifier-based question ranking**. Update when you add Model B or UI.
