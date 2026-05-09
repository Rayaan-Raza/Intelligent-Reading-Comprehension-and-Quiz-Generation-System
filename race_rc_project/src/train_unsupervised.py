"""Unsupervised clustering on article+question+option feature vectors (K-Means)."""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.features import run_feature_pipeline


def _log(msg: str) -> None:
    print(f"[train_unsupervised] {msg}", flush=True)


def _ensure_feature_artifacts_exist() -> None:
    required = [
        PROJECT_ROOT / "models" / "model_a" / "traditional" / "tfidf_vectorizer.pkl",
        PROJECT_ROOT / "data" / "processed" / "X_train_features.npz",
        PROJECT_ROOT / "data" / "processed" / "X_val_features.npz",
        PROJECT_ROOT / "data" / "processed" / "y_train.npy",
        PROJECT_ROOT / "data" / "processed" / "y_val.npy",
    ]
    if all(p.exists() for p in required):
        _log("found existing feature artifacts")
        return
    _log("feature artifacts missing; running feature pipeline")
    run_feature_pipeline(
        train_csv=PROJECT_ROOT / "data" / "processed" / "train_verification.csv",
        val_csv=PROJECT_ROOT / "data" / "processed" / "val_verification.csv",
        project_root=PROJECT_ROOT,
        verbose=True,
    )


def cluster_purity(y_true: np.ndarray, cluster_labels: np.ndarray) -> float:
    """
    Fraction of points that agree with the majority class label within their cluster.
    Range [0, 1]; higher means clusters align better with true correct/wrong labels.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    cluster_labels = np.asarray(cluster_labels).astype(int).ravel()
    n = len(y_true)
    if n == 0:
        return 0.0
    total = 0
    for k in np.unique(cluster_labels):
        mask = cluster_labels == k
        if not np.any(mask):
            continue
        labels_in_cluster = y_true[mask]
        counts = np.bincount(labels_in_cluster, minlength=2)
        total += int(counts.max())
    return float(total / n)


def _subsample_silhouette(
    X: Any,
    cluster_labels: np.ndarray,
    max_samples: int = 8000,
    random_state: int = 42,
) -> Optional[float]:
    """Silhouette on Euclidean distance; subsample for speed/memory on high-dim data."""
    n = X.shape[0]
    if n < 2 or len(np.unique(cluster_labels)) < 2:
        return None
    rng = np.random.default_rng(random_state)
    size = min(n, max_samples)
    idx = rng.choice(n, size=size, replace=False)
    X_sub = X[idx]
    labels_sub = cluster_labels[idx]
    if len(np.unique(labels_sub)) < 2:
        return None
    # Dense slice for sklearn metric (sparse compatible via .toarray)
    if hasattr(X_sub, "toarray"):
        X_dense = X_sub.toarray().astype(np.float32)
    else:
        X_dense = np.asarray(X_sub, dtype=np.float32)
    return float(silhouette_score(X_dense, labels_sub, metric="euclidean", random_state=random_state))


def _load_supervised_comparison() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    paths = [
        ("model_a_results", PROJECT_ROOT / "reports" / "model_a_results.csv"),
        ("neural_results", PROJECT_ROOT / "reports" / "neural_results.csv"),
    ]
    for name, path in paths:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            for _, r in df.iterrows():
                model = str(r.get("model", name))
                rows.append(
                    {
                        "metric_type": "supervised_reference",
                        "source_report": path.name,
                        "model": model,
                        "binary_accuracy": r.get("binary_accuracy"),
                        "binary_macro_f1": r.get("binary_macro_f1"),
                        "mcq_exact_match": r.get("mcq_exact_match"),
                    }
                )
        except Exception as exc:
            _log(f"could not read {path}: {exc}")
    return rows


def train(
    n_clusters: int = 2,
    batch_size: int = 4096,
    random_state: int = 42,
    silhouette_max_samples: int = 8000,
) -> None:
    """
    Fit K-Means (MiniBatch) on training features without using labels in .fit.
    Labels are used only for purity evaluation afterward.
    """
    start = time.perf_counter()
    _log("starting unsupervised K-Means")

    _ensure_feature_artifacts_exist()
    processed = PROJECT_ROOT / "data" / "processed"
    X_train = load_npz(processed / "X_train_features.npz")
    X_val = load_npz(processed / "X_val_features.npz")
    y_train = np.load(processed / "y_train.npy").astype(int).ravel()
    y_val = np.load(processed / "y_val.npy").astype(int).ravel()

    model_dir = PROJECT_ROOT / "models" / "model_a" / "traditional"
    report_dir = PROJECT_ROOT / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    _log(
        f"K-Means MiniBatch: n_clusters={n_clusters}, batch_size={batch_size}, "
        f"train_shape={X_train.shape}"
    )

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=batch_size,
        max_iter=100,
        reassignment_ratio=0.01,
        n_init=3,
    )
    t_fit = time.perf_counter()
    cluster_train = kmeans.fit_predict(X_train)
    _log(f"fit_predict(train) done in {time.perf_counter() - t_fit:.2f}s")

    cluster_val = kmeans.predict(X_val)

    purity_train = cluster_purity(y_train, cluster_train)
    purity_val = cluster_purity(y_val, cluster_val)

    sil_train = _subsample_silhouette(X_train, cluster_train, max_samples=silhouette_max_samples)
    sil_val = _subsample_silhouette(X_val, cluster_val, max_samples=min(silhouette_max_samples, len(y_val)))

    _log(f"cluster_purity train={purity_train:.4f} val={purity_val:.4f}")
    _log(f"silhouette (subsampled) train={sil_train} val={sil_val}")

    artifact = {
        "model": kmeans,
        "n_clusters": n_clusters,
        "random_state": random_state,
        "batch_size": batch_size,
        "feature_dim": int(X_train.shape[1]),
        "train_rows": int(X_train.shape[0]),
    }
    out_model = model_dir / "kmeans_model.pkl"
    joblib.dump(artifact, out_model)
    _log(f"saved {out_model}")

    result_rows: List[Dict[str, Any]] = [
        {
            "metric_type": "unsupervised_kmeans",
            "split": "train",
            "n_clusters": n_clusters,
            "silhouette_score": sil_train,
            "cluster_purity": purity_train,
            "n_samples": int(X_train.shape[0]),
            "notes": "Silhouette on subsample; purity uses true correct/wrong labels for analysis only.",
        },
        {
            "metric_type": "unsupervised_kmeans",
            "split": "val",
            "n_clusters": n_clusters,
            "silhouette_score": sil_val,
            "cluster_purity": purity_val,
            "n_samples": int(X_val.shape[0]),
            "notes": "Same K-Means model as train; labels only for evaluation.",
        },
    ]

    for ref in _load_supervised_comparison():
        result_rows.append(
            {
                "metric_type": ref["metric_type"],
                "split": "reference",
                "n_clusters": np.nan,
                "silhouette_score": np.nan,
                "cluster_purity": np.nan,
                "n_samples": np.nan,
                "notes": f"source={ref.get('source_report')} model={ref.get('model')} "
                f"acc={ref.get('binary_accuracy')} macro_f1={ref.get('binary_macro_f1')} "
                f"mcq_em={ref.get('mcq_exact_match')}",
            }
        )

    out_csv = report_dir / "unsupervised_results.csv"
    pd.DataFrame(result_rows).to_csv(out_csv, index=False)
    _log(f"saved {out_csv}")
    _log(f"total runtime: {time.perf_counter() - start:.2f}s")

    print("\nUnsupervised training complete. Saved artifacts:")
    print("models/model_a/traditional/kmeans_model.pkl")
    print("reports/unsupervised_results.csv")


if __name__ == "__main__":
    train()
