"""Build a single Model A comparison table including neural test results."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


DISPLAY_ORDER = [
    ("logistic_regression_unweighted", "Logistic Regression"),
    ("linear_svm_unweighted", "Linear SVM"),
    ("multinomial_nb", "Naive Bayes"),
    ("random_forest", "Random Forest"),
    ("xgboost", "XGBoost"),
    ("mlp_neural", "MLP Neural Network"),
]


def build_table() -> pd.DataFrame:
    reports = PROJECT_ROOT / "reports"
    sk_path = reports / "test_evaluation.csv"
    nn_path = reports / "test_evaluation_neural.csv"

    if not sk_path.exists():
        raise FileNotFoundError(f"Missing {sk_path}. Run: python src/evaluate_test.py")
    if not nn_path.exists():
        raise FileNotFoundError(
            f"Missing {nn_path}. Run: python src/evaluate_neural_test.py"
        )

    sk_df = pd.read_csv(sk_path)
    nn_df = pd.read_csv(nn_path)
    all_df = pd.concat([sk_df, nn_df], ignore_index=True)

    rows = []
    for raw_name, display_name in DISPLAY_ORDER:
        hit = all_df[all_df["model"] == raw_name]
        if hit.empty:
            continue
        r = hit.iloc[0]
        rows.append(
            {
                "Model": display_name,
                "Binary Acc": float(r["binary_accuracy"]),
                "Macro F1": float(r["binary_macro_f1"]),
                "Precision": float(r["binary_precision"]),
                "Recall": float(r["binary_recall"]),
                "MCQ EM": float(r["mcq_exact_match"]),
            }
        )

    out_df = pd.DataFrame(rows)
    out_csv = reports / "model_a_comparison_table.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[build_model_a_comparison] saved {out_csv}")
    if not out_df.empty:
        print(out_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return out_df


if __name__ == "__main__":
    build_table()
