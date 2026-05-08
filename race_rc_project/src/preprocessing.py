"""Preprocessing utilities for RACE answer-verification training."""

from __future__ import annotations

import re
from typing import Iterable, List

import pandas as pd

REQUIRED_COLUMNS = ["id", "article", "question", "A", "B", "C", "D", "answer"]
VALID_ANSWER_LABELS = {"A", "B", "C", "D"}


def clean_text(text: object) -> str:
    """
    Normalize text for downstream modeling.

    Steps:
    - convert to string
    - lowercase
    - collapse repeated whitespace
    - strip leading/trailing spaces
    """
    if pd.isna(text):
        return ""
    normalized = str(text).lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def normalize_answer_label(label: object) -> str:
    """Normalize answer label to one of A/B/C/D, else empty string."""
    if pd.isna(label):
        return ""
    normalized = str(label).strip().upper()
    return normalized if normalized in VALID_ANSWER_LABELS else ""


def split_sentences(article: object) -> List[str]:
    """
    Split an article into rough sentence-like chunks.

    Keeps punctuation context while segmenting on sentence boundaries.
    """
    cleaned = clean_text(article)
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [part.strip() for part in parts if part.strip()]


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean a raw RACE dataframe.

    Returns a cleaned dataframe containing only valid rows:
    - required columns present
    - no missing required values
    - valid normalized answer label (A/B/C/D)
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Expected columns: {REQUIRED_COLUMNS}"
        )

    cleaned_df = df.copy()
    cleaned_df = cleaned_df.drop(columns=[c for c in cleaned_df.columns if str(c).startswith("Unnamed:")], errors="ignore")
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    cleaned_df = cleaned_df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)

    for col in ["id", "article", "question", "A", "B", "C", "D"]:
        cleaned_df[col] = cleaned_df[col].map(clean_text)

    cleaned_df["answer"] = cleaned_df["answer"].map(normalize_answer_label)
    cleaned_df = cleaned_df[cleaned_df["answer"].isin(VALID_ANSWER_LABELS)].reset_index(drop=True)
    return cleaned_df


def build_verification_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert MCQ rows into binary answer-verification samples.

    For each row, creates 4 rows:
    - (article, question, option=A/B/C/D, is_correct=0/1)
    """
    validated = validate_dataframe(df)
    records = []

    for row in validated.itertuples(index=False):
        for option_label in ("A", "B", "C", "D"):
            option_text = getattr(row, option_label)
            records.append(
                {
                    "id": row.id,
                    "article": row.article,
                    "question": row.question,
                    "option_label": option_label,
                    "option_text": option_text,
                    "answer": row.answer,
                    "is_correct": 1 if row.answer == option_label else 0,
                    "model_input": f"{row.article} [SEP] {row.question} [SEP] {option_text}",
                }
            )

    verification_df = pd.DataFrame.from_records(records)
    return verification_df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compatible helper that validates and cleans a dataframe."""
    return validate_dataframe(df)
