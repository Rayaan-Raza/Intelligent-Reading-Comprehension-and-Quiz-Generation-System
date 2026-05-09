"""Evaluate saved Model A classifiers on held-out test features (no refit, no leakage)."""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


def _log(msg: str) -> None:
    print(f"[evaluate_test] {msg}", flush=True)


def _mcq_exact_match(val_df: pd.DataFrame, scores: np.ndarray) -> float:
    eval_df = val_df.copy()
    eval_df["score"] = scores
    group_col = "sample_id" if "sample_id" in eval_df.columns else ["id", "question"]
    idx = eval_df.groupby(group_col, sort=False)["score"].idxmax()
    selected = eval_df.loc[idx, ["answer", "option_label"]].copy()
    return float((selected["answer"] == selected["option_label"]).mean())


def _model_scores(model: Any, X: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def _unwrap_model(obj: Any) -> Any:
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


TEST_MODELS: List[Tuple[str, str]] = [
    ("logistic_regression_balanced", "logreg_model.pkl"),
    ("logistic_regression_unweighted", "logreg_model_unweighted.pkl"),
    ("linear_svm_balanced", "svm_model.pkl"),
    ("linear_svm_unweighted", "svm_model_unweighted.pkl"),
    ("multinomial_nb", "naive_bayes_model.pkl"),
    ("random_forest", "random_forest_model.pkl"),
    ("xgboost", "xgboost_model.pkl"),
]


def evaluate() -> None:
    start = time.perf_counter()
    processed = PROJECT_ROOT / "data" / "processed"
    test_npz = processed / "X_test_features.npz"
    test_y = processed / "y_test.npy"
    test_csv = processed / "test_verification.csv"

    if not test_npz.exists() or not test_y.exists():
        raise FileNotFoundError(
            f"Missing {test_npz} or {test_y}. Run: python src/build_processed_data.py && python src/features.py"
        )
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing {test_csv}.")

    X_test = load_npz(test_npz)
    y_test = np.load(test_y).astype(int)
    test_df = pd.read_csv(test_csv)
    if len(test_df) != len(y_test):
        raise ValueError("test_verification row count does not match y_test.npy")

    model_dir = PROJECT_ROOT / "models" / "model_a" / "traditional"
    rows: List[Dict[str, Any]] = []

    for name, fname in TEST_MODELS:
        path = model_dir / fname
        if not path.exists():
            _log(f"skip {name} (file missing: {path.name})")
            continue
        raw = joblib.load(path)
        model = _unwrap_model(raw)
        try:
            y_pred = model.predict(X_test)
            scores = _model_scores(model, X_test)
        except Exception as exc:
            _log(f"skip {name} ({exc})")
            continue

        rows.append(
            {
                "model": name,
                "file": fname,
                "split": "test",
                "binary_accuracy": float(accuracy_score(y_test, y_pred)),
                "binary_macro_f1": float(
                    f1_score(y_test, y_pred, average="macro", zero_division=0)
                ),
                "binary_precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "binary_recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "mcq_exact_match": _mcq_exact_match(test_df, scores),
                "notes": "",
            }
        )
        _log(
            f"{name}: test acc={rows[-1]['binary_accuracy']:.4f} "
            f"macro_f1={rows[-1]['binary_macro_f1']:.4f} "
            f"mcq_em={rows[-1]['mcq_exact_match']:.4f}"
        )

    if not rows:
        raise RuntimeError("No models evaluated successfully.")

    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_csv = report_dir / "test_evaluation.csv"
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    _log(f"saved {out_csv}")
    _log("model comparison (test):")
    print(
        out_df[
            ["model", "binary_accuracy", "binary_macro_f1", "binary_precision", "binary_recall", "mcq_exact_match"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    _log(f"done in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    evaluate()
