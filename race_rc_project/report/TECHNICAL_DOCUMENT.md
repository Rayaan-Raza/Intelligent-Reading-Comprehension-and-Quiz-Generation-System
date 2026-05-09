# RACE RC Project - Technical Document

## 1. Problem definition

This project implements a reading-comprehension system for RACE-style MCQs using **option verification**:

- Input: `(article, question, option)`
- Output: `is_correct in {0, 1}`

Each MCQ is expanded into four verifier rows (A/B/C/D). A model scores all options and picks the highest score as the final answer.

---

## 2. System architecture

### 2.1 Pipeline

1. Load + validate RACE data
2. Create leakage-safe splits
3. Expand into verification rows
4. Build train/val/test feature matrices
5. Train traditional + neural models
6. Run unsupervised analysis
7. Generate questions (template mode)
8. Score/rank options with trained verifier

### 2.2 Core modules

- `src/preprocessing.py`: cleaning + verification dataset creation
- `src/features.py`: TF-IDF + numeric/similarity features
- `src/train_model_a.py`: traditional ML training/evaluation
- `src/train_neural_model.py`: MLP training/evaluation
- `src/train_unsupervised.py`: K-Means + silhouette + purity
- `src/model_a_question_generation.py`: RACE/custom question generation
- `src/question_ranking.py`: candidate scoring + `predict_mcq_answer(...)`
- `src/evaluate_test.py`: held-out test metrics
- `scripts/test_model_a_verifier_on_generated_question.py`: end-to-end demo

---

## 3. Data and preprocessing strategy

### 3.1 Verification conversion

Every MCQ row becomes 4 rows:

- option A (label 0/1)
- option B (label 0/1)
- option C (label 0/1)
- option D (label 0/1)

### 3.2 `sample_id` for safe grouping

To avoid grouping collisions:

`sample_id = f"{id}__{original_row_index}"`

This is used in MCQ-level exact-match grouping whenever available.

### 3.3 Cleaning

- lowercase normalization
- whitespace normalization
- answer label normalization to `A/B/C/D`
- sentence splitting for sentence-level features

---

## 4. Feature engineering

### 4.1 Feature dimensions

Model input uses:

- 20,000 TF-IDF features
- 13 numeric/similarity features

Total: **20,013 features per row**

### 4.2 Text channel

`combined_text = article + [SEP] + question + [SEP] + option_text`

### 4.3 Numeric/similarity channel

Implemented numeric features include:

1. article-question cosine
2. article-option cosine
3. question-option cosine
4. article-correct-sentence similarity
5. option-best-sentence similarity
6. question-best-sentence similarity
7. article length
8. question length
9. option length
10. question-option overlap
11. article-option overlap
12. option exact-in-article
13. option frequency-in-article

### 4.4 Leakage safety

- Vectorizer is fit on **train only**
- Val and test are transform-only
- `features.py` now persists optional test artifacts:
  - `X_test_features.npz`
  - `y_test.npy`

---

## 5. Supervised models (Model A)

### 5.1 Trained model families

- Logistic Regression (balanced + unweighted)
- Linear SVM (balanced + unweighted)
- Multinomial Naive Bayes
- Random Forest
- XGBoost (configured; model file currently missing in latest run)

### 5.2 Evaluation levels

1. **Binary verification** (row-level):
   - accuracy
   - macro F1
   - precision
   - recall
2. **MCQ exact match** (group-level):
   - pick argmax score over A/B/C/D per question

---

## 6. Neural model (MLP)

### 6.1 Motivation

A compact neural baseline was used before heavy transformer models.

### 6.2 Architecture

- Dense -> ReLU -> LayerNorm -> Dropout
- Dense -> ReLU -> Dropout
- Logit output (`BCEWithLogitsLoss`)

### 6.3 Training setup

- AdamW
- LR scheduler (`ReduceLROnPlateau`)
- early stopping
- gradient clipping
- weighted BCE + tracked unweighted loss (`val_u`)
- CUDA path enabled

---

## 7. Unsupervised analysis

### 7.1 Method

- MiniBatchKMeans (`n_clusters=2`)
- trained without labels

### 7.2 Metrics

- silhouette score (subsampled)
- cluster purity (post-hoc vs labels)

### 7.3 Observed values

From `reports/unsupervised_results.csv`:

- silhouette ~0.542 (train/val)
- cluster purity ~0.75 (train/val)

---

## 8. Question generation and ranking

### 8.1 Generation mode

Current generation is **template/heuristic-based**:

- sentence split + salience heuristic
- answer phrase extraction
- templates (Where/When/Who/generic cloze)

### 8.2 AI usage in this phase

Generation itself is largely rule-based.

AI is used in **ranking/verification**:

- candidate/questions/options are converted to verifier rows
- scored with trained models via same 20,013-dim feature pipeline

Implemented via `predict_mcq_answer(...)` and ranking helpers in `src/question_ranking.py`.

---

## 9. End-to-end demo behavior from latest terminal output

### 9.1 Synthetic generated examples (5)

Summary observed:

- logistic_regression: **5/5**
- linear_svm: **5/5**
- naive_bayes: **5/5**
- random_forest: **1/5**
- xgboost: skipped (missing model file)

Interpretation: synthetic demos verify integration and scorer behavior but are not strong generalization evidence.

### 9.2 Held-out test metrics (`python src/evaluate_test.py`)

From terminal/report:

- logistic_regression_unweighted: MCQ EM **0.3529**
- linear_svm_balanced: MCQ EM **0.3684**
- linear_svm_unweighted: MCQ EM **0.3551**
- multinomial_nb: MCQ EM **0.3465**
- random_forest: MCQ EM **0.3964** (highest in this table)
- xgboost: skipped (missing file)

### 9.3 Sampled RACE examples (`n=3` in end-to-end script)

Summary observed:

- logistic_regression: **2/3**
- linear_svm: **2/3**
- naive_bayes: **1/3**
- random_forest: **0/3**
- xgboost: skipped

This confirms real examples are harder than synthetic demos.

---

## 10. Current issues

1. **scikit-learn version mismatch warnings**
   - Existing pickles were serialized with sklearn 1.8.0 and loaded under 1.7.2.
   - This caused at least one skip (`LogisticRegression` missing `multi_class`) in test evaluation.

2. **Missing `xgboost_model.pkl`**
   - xgboost is configured but currently not available for scoring scripts.

3. **Pandas future warning in RACE sample grouping**
   - harmless now, but should be cleaned for forward compatibility.

---

## 11. Recommended stabilization steps

1. Retrain traditional models in current environment:
   - `python src/train_model_a.py`
2. Re-evaluate test:
   - `python src/evaluate_test.py`
3. Re-run end-to-end demo:
   - `python scripts/test_model_a_verifier_on_generated_question.py`
4. Ensure xgboost model artifact is produced and loaded.
5. Freeze dependency versions for reproducibility.

---

## 12. Reproducible command set

```powershell
# data pipeline
python src/create_splits.py
python src/build_processed_data.py
python src/features.py

# training
python src/train_model_a.py
python src/train_neural_model.py
python src/train_unsupervised.py

# evaluation
python src/evaluate_test.py

# demos
python scripts/test_module8_question_generation.py
python scripts/test_model_a_verifier_on_generated_question.py
```

---

## 13. Technical conclusion

The project currently provides a complete and modular RC system:

- leakage-safe preprocessing and splits
- unified 20,013-dim verifier features
- traditional and neural supervised baselines
- unsupervised cluster analysis
- rule-based generation with ML-based verifier/ranking
- end-to-end demo over synthetic and sampled RACE examples

Performance on held-out data indicates meaningful but non-saturated capability, with clear gains expected from environment alignment, missing-model completion (xgboost), and iterative feature/model tuning.

