"""Utilities for loading RACE dataset splits and generating EDA metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
from pandas.errors import EmptyDataError

EXPECTED_COLUMNS = ["id", "article", "question", "A", "B", "C", "D", "answer"]


def _empty_race_frame() -> pd.DataFrame:
    """Return an empty dataframe with the expected RACE schema."""
    return pd.DataFrame(columns=EXPECTED_COLUMNS)


def _normalize_index_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop accidental unnamed index columns.

    Some CSV exports include an unnamed first index column. If it exists,
    remove it so downstream schema checks stay stable.
    """
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return df


def _validate_columns(df: pd.DataFrame, split_name: str) -> None:
    """Ensure the dataframe has all required RACE columns."""
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {split_name}.csv: {missing}. "
            f"Expected columns: {EXPECTED_COLUMNS}"
        )


def load_split(csv_path: Path | str, split_name: str | None = None) -> pd.DataFrame:
    """
    Load one RACE split CSV as a dataframe.

    Returns an empty dataframe with expected schema if the file is empty.
    """
    path = Path(csv_path)
    resolved_split_name = split_name or path.stem

    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        return _empty_race_frame()

    df = _normalize_index_column(df)
    if df.empty:
        return _empty_race_frame()

    _validate_columns(df, resolved_split_name)
    return df


def load_dataset_splits(
    data_dir: Path | str = Path("data/splits"),
    split_names: Iterable[str] = ("train", "val", "test"),
) -> Dict[str, pd.DataFrame]:
    """Load multiple dataset splits from a directory."""
    base = Path(data_dir)
    splits: Dict[str, pd.DataFrame] = {}

    for split in split_names:
        splits[split] = load_split(base / f"{split}.csv", split_name=split)

    return splits
