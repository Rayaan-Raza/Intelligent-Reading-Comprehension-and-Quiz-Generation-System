"""Build and persist preprocessed answer-verification datasets."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.preprocessing import build_verification_dataset


def build_all_processed_splits(
    input_dir: Path | str = Path("data/splits"),
    output_dir: Path | str = Path("data/processed"),
) -> dict[str, tuple[int, int]]:
    """
    Create preprocessed verification CSVs for train/val/test splits.

    Returns:
        Mapping split -> (input_rows, output_rows)
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, tuple[int, int]] = {}
    for split in ("train", "val", "test"):
        input_path = in_dir / f"{split}.csv"
        output_path = out_dir / f"{split}_verification.csv"

        split_df = pd.read_csv(input_path)
        verification_df = build_verification_dataset(split_df)
        verification_df.to_csv(output_path, index=False)

        stats[split] = (len(split_df), len(verification_df))

    return stats


if __name__ == "__main__":
    build_stats = build_all_processed_splits()
    print("Saved processed verification datasets to data/processed")
    for split_name, (input_rows, output_rows) in build_stats.items():
        print(f"{split_name}: {input_rows} -> {output_rows}")
