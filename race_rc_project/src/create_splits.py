"""Create leakage-safe train/val/test splits by unique passage id."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import EXPECTED_COLUMNS


def create_id_based_splits(
    source_csv: Path | str = Path("data/raw/val.csv"),
    output_dir: Path | str = Path("data/splits"),
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create 80/10/10 train/val/test splits by unique id."""
    source_path = Path(source_csv)
    split_dir = Path(output_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    full_df = pd.read_csv(source_path)
    full_df = full_df.drop(columns=[c for c in full_df.columns if str(c).startswith("Unnamed:")], errors="ignore")

    for col in EXPECTED_COLUMNS:
        if col not in full_df.columns:
            raise ValueError(f"Missing column: {col}")

    full_df = full_df.drop_duplicates().reset_index(drop=True)
    full_df = full_df.dropna(subset=EXPECTED_COLUMNS).reset_index(drop=True)
    full_df["answer"] = full_df["answer"].astype(str).str.strip().str.upper()
    full_df = full_df[full_df["answer"].isin(["A", "B", "C", "D"])].reset_index(drop=True)

    unique_ids = full_df["id"].astype(str).unique().tolist()
    train_ids, temp_ids = train_test_split(
        unique_ids, test_size=0.20, random_state=random_state, shuffle=True
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.50, random_state=random_state, shuffle=True
    )

    train_df = full_df[full_df["id"].isin(train_ids)].reset_index(drop=True)
    val_df = full_df[full_df["id"].isin(val_ids)].reset_index(drop=True)
    test_df = full_df[full_df["id"].isin(test_ids)].reset_index(drop=True)

    train_set = set(train_df["id"])
    val_set = set(val_df["id"])
    test_set = set(test_df["id"])

    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0

    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "val.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)

    return train_df, val_df, test_df


if __name__ == "__main__":
    train_df, val_df, test_df = create_id_based_splits()
    print("Created leakage-safe splits in data/splits")
    print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
