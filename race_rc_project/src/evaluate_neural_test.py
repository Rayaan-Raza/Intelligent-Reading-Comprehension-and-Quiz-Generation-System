"""Evaluate the trained MLP neural verifier on held-out test features."""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.exceptions import InconsistentVersionWarning
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", message=".*LF will be replaced by CRLF.*")


def _log(msg: str) -> None:
    print(f"[evaluate_neural_test] {msg}", flush=True)


def _mcq_exact_match(eval_df: pd.DataFrame, scores: np.ndarray) -> float:
    df = eval_df.copy()
    df["score"] = scores
    group_col = "sample_id" if "sample_id" in df.columns else ["id", "question"]
    idx = df.groupby(group_col, sort=False)["score"].idxmax()
    selected = df.loc[idx, ["answer", "option_label"]].copy()
    return float((selected["answer"] == selected["option_label"]).mean())


class MLPBinaryClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_1: int, hidden_2: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(),
            nn.LayerNorm(hidden_1),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


@torch.no_grad()
def _predict_scores(model: nn.Module, X_test, batch_size: int = 1024) -> np.ndarray:
    n = X_test.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        dense = X_test[start:end].toarray().astype(np.float32)
        logits = model(torch.from_numpy(dense))
        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
        out[start:end] = probs
    return out


def evaluate() -> Dict[str, float]:
    start = time.perf_counter()
    processed = PROJECT_ROOT / "data" / "processed"
    model_path = PROJECT_ROOT / "models" / "model_a" / "neural" / "mlp_model.pt"
    test_npz = processed / "X_test_features.npz"
    test_y = processed / "y_test.npy"
    test_csv = processed / "test_verification.csv"

    missing = [str(p) for p in [model_path, test_npz, test_y, test_csv] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files for neural test evaluation:\n- " + "\n- ".join(missing)
        )

    ckpt = torch.load(model_path, map_location="cpu")
    model = MLPBinaryClassifier(
        input_dim=int(ckpt["input_dim"]),
        hidden_1=int(ckpt["hidden_1"]),
        hidden_2=int(ckpt["hidden_2"]),
        dropout=float(ckpt.get("dropout", 0.35)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    X_test = load_npz(test_npz)
    y_test = np.load(test_y).astype(int)
    test_df = pd.read_csv(test_csv)
    if len(test_df) != len(y_test):
        raise ValueError("test_verification row count does not match y_test.npy")

    scores = _predict_scores(model, X_test)
    y_pred = (scores >= 0.5).astype(int)

    metrics: Dict[str, float] = {
        "binary_accuracy": float(accuracy_score(y_test, y_pred)),
        "binary_macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "binary_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "binary_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "mcq_exact_match": _mcq_exact_match(test_df, scores),
    }

    out_row = {
        "model": "mlp_neural",
        "file": "mlp_model.pt",
        "split": "test",
        **metrics,
        "notes": "",
    }
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_csv = report_dir / "test_evaluation_neural.csv"
    pd.DataFrame([out_row]).to_csv(out_csv, index=False)

    _log(
        "mlp_neural: "
        f"acc={metrics['binary_accuracy']:.4f} "
        f"macro_f1={metrics['binary_macro_f1']:.4f} "
        f"precision={metrics['binary_precision']:.4f} "
        f"recall={metrics['binary_recall']:.4f} "
        f"mcq_em={metrics['mcq_exact_match']:.4f}"
    )
    _log(f"saved {out_csv}")
    _log(f"done in {time.perf_counter() - start:.2f}s")
    return metrics


if __name__ == "__main__":
    evaluate()
