"""Train and evaluate Model A traditional ML baselines."""

from __future__ import annotations

from pathlib import Path
import time
import sys
from typing import Any, Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.features import run_feature_pipeline


def _log(msg: str) -> None:
    print(f"[train_model_a] {msg}", flush=True)


def _class_balance(y: np.ndarray) -> str:
    total = len(y)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return f"total={total}, pos={pos} ({(pos / total):.2%}), neg={neg} ({(neg / total):.2%})"


def _ensure_feature_artifacts_exist() -> None:
    required = [
        PROJECT_ROOT / "models" / "model_a" / "traditional" / "tfidf_vectorizer.pkl",
        PROJECT_ROOT / "data" / "processed" / "X_train_features.npz",
        PROJECT_ROOT / "data" / "processed" / "X_val_features.npz",
        PROJECT_ROOT / "data" / "processed" / "y_train.npy",
        PROJECT_ROOT / "data" / "processed" / "y_val.npy",
    ]
    if all(path.exists() for path in required):
        _log("found existing feature artifacts")
        return

    _log("feature artifacts missing; running feature pipeline")
    run_feature_pipeline(
        train_csv=PROJECT_ROOT / "data" / "processed" / "train_verification.csv",
        val_csv=PROJECT_ROOT / "data" / "processed" / "val_verification.csv",
        project_root=PROJECT_ROOT,
        verbose=True,
    )


def _load_features() -> Tuple[Any, Any, np.ndarray, np.ndarray]:
    _log("loading precomputed feature matrices and labels")
    processed_dir = PROJECT_ROOT / "data" / "processed"
    X_train = load_npz(processed_dir / "X_train_features.npz")
    X_val = load_npz(processed_dir / "X_val_features.npz")
    y_train = np.load(processed_dir / "y_train.npy")
    y_val = np.load(processed_dir / "y_val.npy")
    _log(f"X_train shape={X_train.shape}, X_val shape={X_val.shape}")
    _log(f"y_train balance: {_class_balance(y_train)}")
    _log(f"y_val balance: {_class_balance(y_val)}")
    return X_train, X_val, y_train, y_val


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "binary_accuracy": float(accuracy_score(y_true, y_pred)),
        "binary_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "binary_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "binary_recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def _mcq_exact_match(val_df: pd.DataFrame, scores: np.ndarray) -> float:
    eval_df = val_df.copy()
    eval_df["score"] = scores
    group_col = "sample_id" if "sample_id" in eval_df.columns else ["id", "question"]
    idx = eval_df.groupby(group_col, sort=False)["score"].idxmax()
    selected = eval_df.loc[idx, ["answer", "option_label"]].copy()
    return float((selected["answer"] == selected["option_label"]).mean())


def _model_scores(model: Any, X_val: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_val)[:, 1]
    return model.decision_function(X_val)


def train() -> None:
    start_time = time.perf_counter()
    _log("starting Model A traditional training")
    cuda_available = False
    try:
        import torch

        if torch.cuda.is_available():
            cuda_available = True
            _log("CUDA detected; sklearn baselines still train on CPU.")
    except Exception:
        pass

    model_dir = PROJECT_ROOT / "models" / "model_a" / "traditional"
    report_dir = PROJECT_ROOT / "reports"
    figure_dir = PROJECT_ROOT / "figures"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    _ensure_feature_artifacts_exist()
    X_train, X_val, y_train, y_val = _load_features()
    _log("loading validation verification rows for MCQ exact-match evaluation")
    val_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "val_verification.csv")
    if len(val_df) != len(y_val):
        raise ValueError(
            "Validation rows mismatch between val_verification.csv and y_val.npy. "
            "Re-run src/features.py first."
        )

    models = {
        "logistic_regression_unweighted": LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight=None,
            random_state=42,
        ),
        "logistic_regression_balanced": LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            random_state=42,
        ),
        "linear_svm_unweighted": LinearSVC(
            C=1.0,
            class_weight=None,
            random_state=42,
        ),
        "linear_svm_balanced": LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=42,
        ),
        "multinomial_nb": MultinomialNB(alpha=0.5),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            tree_method="gpu_hist" if cuda_available else "hist",
        ),
    }

    all_results = []
    confusion_data: Dict[str, np.ndarray] = {}

    for model_name, model in models.items():
        _log(f"training {model_name}")
        model_start = time.perf_counter()
        model.fit(X_train, y_train)
        _log(f"{model_name} fit complete in {time.perf_counter() - model_start:.2f}s")

        _log(f"running validation predictions for {model_name}")
        y_pred = model.predict(X_val)
        scores = _model_scores(model, X_val)

        metrics = _binary_metrics(y_val, y_pred)
        metrics["mcq_exact_match"] = _mcq_exact_match(val_df, scores)
        metrics["model"] = model_name
        metrics["baseline"] = "Baseline 1"
        all_results.append(metrics)
        confusion_data[model_name] = confusion_matrix(y_val, y_pred, labels=[0, 1])
        _log(
            f"{model_name} metrics | "
            f"acc={metrics['binary_accuracy']:.4f}, "
            f"macro_f1={metrics['binary_macro_f1']:.4f}, "
            f"precision={metrics['binary_precision']:.4f}, "
            f"recall={metrics['binary_recall']:.4f}, "
            f"mcq_em={metrics['mcq_exact_match']:.4f}"
        )

        model_filename_map = {
            "logistic_regression_unweighted": "logreg_model_unweighted.pkl",
            "logistic_regression_balanced": "logreg_model.pkl",
            "linear_svm_unweighted": "svm_model_unweighted.pkl",
            "linear_svm_balanced": "svm_model.pkl",
            "multinomial_nb": "naive_bayes_model.pkl",
            "random_forest": "random_forest_model.pkl",
            "xgboost": "xgboost_model.pkl",
        }
        model_path = model_dir / model_filename_map[model_name]
        joblib.dump(model, model_path)
        _log(f"saved {model_name} to {model_path}")

    results_df = pd.DataFrame(all_results)[
        [
            "baseline",
            "model",
            "binary_accuracy",
            "binary_macro_f1",
            "binary_precision",
            "binary_recall",
            "mcq_exact_match",
        ]
    ]
    results_csv = report_dir / "model_a_results.csv"
    results_df.to_csv(results_csv, index=False)
    _log(f"saved results to {results_csv}")
    _log(f"\n{results_df.to_string(index=False)}")

    figure_model_order = [
        "logistic_regression_unweighted",
        "logistic_regression_balanced",
        "linear_svm_unweighted",
        "linear_svm_balanced",
        "multinomial_nb",
        "random_forest",
        "xgboost",
    ]
    n_models = len(figure_model_order)
    n_cols = 3
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, model_name in zip(axes_flat, figure_model_order):
        cm = confusion_data[model_name]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(ax=ax, values_format="d", colorbar=False)
        ax.set_title(model_name)
    for ax in axes_flat[n_models:]:
        ax.axis("off")
    fig.tight_layout()
    cm_path = figure_dir / "model_a_confusion_matrix.png"
    fig.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _log(f"saved confusion matrix figure to {cm_path}")
    _log(f"total runtime: {time.perf_counter() - start_time:.2f}s")

    print("\nModel A training complete. Saved artifacts:")
    print("models/model_a/traditional/logreg_model.pkl")
    print("models/model_a/traditional/logreg_model_unweighted.pkl")
    print("models/model_a/traditional/svm_model.pkl")
    print("models/model_a/traditional/svm_model_unweighted.pkl")
    print("models/model_a/traditional/naive_bayes_model.pkl")
    print("models/model_a/traditional/random_forest_model.pkl")
    print("models/model_a/traditional/xgboost_model.pkl")
    print("reports/model_a_results.csv")
    print("figures/model_a_confusion_matrix.png")


if __name__ == "__main__":
    train()
