"""Train and evaluate a small MLP neural model for Model A."""

from __future__ import annotations

from pathlib import Path
import copy
import sys
import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.features import run_feature_pipeline


def _log(msg: str) -> None:
    print(f"[train_neural_model] {msg}", flush=True)


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
    processed_dir = PROJECT_ROOT / "data" / "processed"
    X_train = load_npz(processed_dir / "X_train_features.npz")
    X_val = load_npz(processed_dir / "X_val_features.npz")
    y_train = np.load(processed_dir / "y_train.npy").astype(np.float32)
    y_val = np.load(processed_dir / "y_val.npy").astype(np.float32)
    return X_train, X_val, y_train, y_val


def _mcq_exact_match(val_df: pd.DataFrame, scores: np.ndarray) -> float:
    eval_df = val_df.copy()
    eval_df["score"] = scores
    group_col = "sample_id" if "sample_id" in eval_df.columns else ["id", "question"]
    idx = eval_df.groupby(group_col, sort=False)["score"].idxmax()
    selected = eval_df.loc[idx, ["answer", "option_label"]].copy()
    return float((selected["answer"] == selected["option_label"]).mean())


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "binary_accuracy": float(accuracy_score(y_true, y_pred)),
        "binary_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "binary_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "binary_recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


class SparseBinaryDataset(Dataset):
    """Dataset that converts sparse rows to dense float tensors on demand."""

    def __init__(self, X, y: np.ndarray) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].toarray().ravel().astype(np.float32)
        y = np.float32(self.y[idx])
        return torch.from_numpy(x), torch.tensor(y)


class MLPBinaryClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_1: int = 512, hidden_2: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


@torch.no_grad()
def _predict_scores(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    all_scores = []
    for x_batch, _ in loader:
        x_batch = x_batch.to(device)
        logits = model(x_batch)
        probs = torch.sigmoid(logits)
        all_scores.append(probs.detach().cpu().numpy())
    return np.concatenate(all_scores, axis=0)


def train() -> None:
    start = time.perf_counter()
    _log("starting neural training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"device={device}")

    _ensure_feature_artifacts_exist()
    X_train, X_val, y_train, y_val = _load_features()
    val_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "val_verification.csv")
    if len(val_df) != len(y_val):
        raise ValueError(
            "Validation rows mismatch between val_verification.csv and y_val.npy. "
            "Re-run src/features.py first."
        )

    train_ds = SparseBinaryDataset(X_train, y_train)
    val_ds = SparseBinaryDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    input_dim = X_train.shape[1]
    model = MLPBinaryClassifier(input_dim=input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    patience = 4
    patience_counter = 0
    max_epochs = 20

    _log(f"input_dim={input_dim}, train_rows={len(train_ds)}, val_rows={len(val_ds)}")
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = y_batch.size(0)
            train_loss_sum += float(loss.item()) * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / max(train_count, 1)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                batch_size = y_batch.size(0)
                val_loss_sum += float(loss.item()) * batch_size
                val_count += batch_size
        val_loss = val_loss_sum / max(val_count, 1)

        _log(f"epoch={epoch:02d} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                _log("early stopping triggered")
                break

    model.load_state_dict(best_state)
    scores = _predict_scores(model, val_loader, device=device)
    y_pred = (scores >= 0.5).astype(int)
    y_true = y_val.astype(int)

    metrics = _binary_metrics(y_true, y_pred)
    metrics["mcq_exact_match"] = _mcq_exact_match(val_df, scores)
    metrics["model"] = "mlp_neural"
    metrics["baseline"] = "Neural"

    model_dir = PROJECT_ROOT / "models" / "model_a" / "neural"
    report_dir = PROJECT_ROOT / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "mlp_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_1": 512,
            "hidden_2": 128,
            "dropout": 0.3,
            "best_val_loss": best_val_loss,
        },
        model_path,
    )
    _log(f"saved model to {model_path}")

    results_df = pd.DataFrame(
        [
            {
                "baseline": metrics["baseline"],
                "model": metrics["model"],
                "binary_accuracy": metrics["binary_accuracy"],
                "binary_macro_f1": metrics["binary_macro_f1"],
                "binary_precision": metrics["binary_precision"],
                "binary_recall": metrics["binary_recall"],
                "mcq_exact_match": metrics["mcq_exact_match"],
                "best_val_loss": best_val_loss,
                "device": str(device),
            }
        ]
    )
    results_csv = report_dir / "neural_results.csv"
    results_df.to_csv(results_csv, index=False)
    _log(f"saved results to {results_csv}")
    _log(
        "metrics | "
        f"acc={metrics['binary_accuracy']:.4f}, "
        f"macro_f1={metrics['binary_macro_f1']:.4f}, "
        f"precision={metrics['binary_precision']:.4f}, "
        f"recall={metrics['binary_recall']:.4f}, "
        f"mcq_em={metrics['mcq_exact_match']:.4f}"
    )
    _log(f"total runtime: {time.perf_counter() - start:.2f}s")

    print("\nNeural training complete. Saved artifacts:")
    print("models/model_a/neural/mlp_model.pt")
    print("reports/neural_results.csv")


if __name__ == "__main__":
    train()
