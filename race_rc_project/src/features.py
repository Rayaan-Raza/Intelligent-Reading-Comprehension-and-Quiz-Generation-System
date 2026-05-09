"""Feature engineering for answer-verification models."""

from __future__ import annotations

from pathlib import Path
import pickle
import re
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.preprocessing import split_sentences


NUMERIC_FEATURE_COLUMNS = [
    "article_question_similarity",
    "article_option_similarity",
    "question_option_similarity",
    "article_correct_sentence_similarity",
    "option_best_sentence_similarity",
    "question_best_sentence_similarity",
    "article_length",
    "question_length",
    "option_length",
    "question_option_word_overlap",
    "article_option_word_overlap",
    "option_exact_in_article",
    "option_frequency_in_article",
]


def _log(message: str, verbose: bool = True) -> None:
    if verbose:
        print(f"[features] {message}", flush=True)


def _safe_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _tokenize_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _rowwise_cosine(a, b) -> np.ndarray:
    numerator = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    a_norm = np.sqrt(np.asarray(a.multiply(a).sum(axis=1)).ravel())
    b_norm = np.sqrt(np.asarray(b.multiply(b).sum(axis=1)).ravel())
    denom = a_norm * b_norm
    denom[denom == 0] = 1e-12
    return numerator / denom


def _build_base_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in ["id", "article", "question", "option_text"]:
        work[col] = work[col].map(_safe_text)
    work["combined_text"] = (
        work["article"] + " [SEP] " + work["question"] + " [SEP] " + work["option_text"]
    )
    return work


def _build_group_correct_option_map(df: pd.DataFrame) -> Dict[tuple[str, str], str]:
    if "sample_id" in df.columns:
        group_cols = ["sample_id"]
    else:
        group_cols = ["id", "question"]

    correct_rows = df[df["is_correct"] == 1][group_cols + ["option_text"]]
    if correct_rows.empty:
        return {}
    grouped = correct_rows.drop_duplicates(subset=group_cols, keep="first")
    if group_cols == ["sample_id"]:
        return {(row.sample_id, ""): row.option_text for row in grouped.itertuples(index=False)}
    return {(row.id, row.question): row.option_text for row in grouped.itertuples(index=False)}


def _token_cosine(a_tokens: set[str], b_tokens: set[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens.intersection(b_tokens))
    return float(inter / np.sqrt(len(a_tokens) * len(b_tokens)))


def _compute_sentence_similarity_features(
    df: pd.DataFrame, verbose: bool = False, log_every: int = 20000
) -> pd.DataFrame:
    """
    Compute sentence-level cosine similarity features.

    Uses sentence splits from each article and compares with:
    - option text
    - question text
    - correct option text (within the same question group)
    """
    correct_option_map = _build_group_correct_option_map(df)
    cache: Dict[tuple[str, str, str, str], tuple[float, float, float]] = {}
    article_cache: Dict[str, tuple[list[str], list[set[str]], set[str]]] = {}

    article_correct_sentence = np.zeros(len(df), dtype=np.float32)
    option_best_sentence = np.zeros(len(df), dtype=np.float32)
    question_best_sentence = np.zeros(len(df), dtype=np.float32)

    for idx, row in enumerate(df.itertuples(index=False)):
        sample_id = getattr(row, "sample_id", None)
        group_key = (sample_id, "") if sample_id is not None else (row.id, row.question)
        correct_option_text = correct_option_map.get(group_key, "")
        cache_key = (row.article, row.question, row.option_text, correct_option_text)

        if cache_key not in cache:
            if row.article not in article_cache:
                sentences = split_sentences(row.article)
                if not sentences:
                    sentences = [row.article]
                sentence_tokens = [_tokenize_words(s) for s in sentences]
                article_tokens = _tokenize_words(row.article)
                article_cache[row.article] = (sentences, sentence_tokens, article_tokens)

            _, sentence_tokens, article_tokens = article_cache[row.article]

            option_tokens = _tokenize_words(row.option_text)
            question_tokens = _tokenize_words(row.question)
            correct_tokens = _tokenize_words(correct_option_text)

            option_sims = [_token_cosine(sent, option_tokens) for sent in sentence_tokens]
            question_sims = [_token_cosine(sent, question_tokens) for sent in sentence_tokens]

            option_best = max(option_sims) if option_sims else 0.0
            question_best = max(question_sims) if question_sims else 0.0

            if correct_tokens and sentence_tokens:
                best_idx = int(np.argmax([_token_cosine(sent, correct_tokens) for sent in sentence_tokens]))
                best_sentence_tokens = sentence_tokens[best_idx]
                article_correct = _token_cosine(article_tokens, best_sentence_tokens)
            else:
                article_correct = 0.0

            cache[cache_key] = (article_correct, option_best, question_best)

        article_correct_sentence[idx], option_best_sentence[idx], question_best_sentence[idx] = (
            cache[cache_key]
        )
        if verbose and (idx + 1) % log_every == 0:
            _log(
                f"sentence similarity progress: {idx + 1}/{len(df)} rows "
                f"({((idx + 1) / len(df)) * 100:.1f}%)",
                verbose=verbose,
            )

    return pd.DataFrame(
        {
            "article_correct_sentence_similarity": article_correct_sentence,
            "option_best_sentence_similarity": option_best_sentence,
            "question_best_sentence_similarity": question_best_sentence,
        }
    )


def _compute_numeric_features(
    base_df: pd.DataFrame, vectorizer: TfidfVectorizer, verbose: bool = False
) -> pd.DataFrame:
    _log(f"building numeric/cosine features for {len(base_df)} rows", verbose=verbose)
    article_vec = vectorizer.transform(base_df["article"])
    question_vec = vectorizer.transform(base_df["question"])
    option_vec = vectorizer.transform(base_df["option_text"])

    article_question_sim = _rowwise_cosine(article_vec, question_vec)
    article_option_sim = _rowwise_cosine(article_vec, option_vec)
    question_option_sim = _rowwise_cosine(question_vec, option_vec)

    question_tokens = base_df["question"].map(_tokenize_words)
    option_tokens = base_df["option_text"].map(_tokenize_words)
    article_tokens = base_df["article"].map(_tokenize_words)

    numeric_df = pd.DataFrame(
        {
            "article_question_similarity": article_question_sim,
            "article_option_similarity": article_option_sim,
            "question_option_similarity": question_option_sim,
            "article_length": base_df["article"].str.len().astype(np.float32),
            "question_length": base_df["question"].str.len().astype(np.float32),
            "option_length": base_df["option_text"].str.len().astype(np.float32),
            "question_option_word_overlap": [
                float(len(q_toks.intersection(o_toks)))
                for q_toks, o_toks in zip(question_tokens, option_tokens)
            ],
            "article_option_word_overlap": [
                float(len(a_toks.intersection(o_toks)))
                for a_toks, o_toks in zip(article_tokens, option_tokens)
            ],
            "option_exact_in_article": [
                float(option in article and option != "")
                for article, option in zip(base_df["article"], base_df["option_text"])
            ],
            "option_frequency_in_article": [
                float(article.count(option)) if option else 0.0
                for article, option in zip(base_df["article"], base_df["option_text"])
            ],
        }
    )

    sentence_df = _compute_sentence_similarity_features(base_df, verbose=verbose)
    return pd.concat([numeric_df, sentence_df], axis=1)[NUMERIC_FEATURE_COLUMNS]


def _get_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=20000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


def transform_verification_dataframe(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    verbose: bool = False,
) -> csr_matrix:
    """
    Apply a **fitted** TF-IDF vectorizer + numeric features to a verification-shaped DataFrame.

    Use for test-time inference, held-out test rows, or ranking — **do not** refit the vectorizer.
    """
    base = _build_base_feature_frame(df)
    if "is_correct" not in base.columns:
        base["is_correct"] = df["is_correct"].values if "is_correct" in df.columns else 1
    X_tfidf = vectorizer.transform(base["combined_text"])
    numeric = csr_matrix(_compute_numeric_features(base, vectorizer, verbose=verbose).values)
    return hstack([X_tfidf, numeric], format="csr")


def build_feature_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    verbose: bool = False,
) -> Tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray, TfidfVectorizer, Optional[csr_matrix], Optional[np.ndarray]]:
    """
    Build leakage-safe train/val (+ optional test) feature matrices.

    - fits TF-IDF on **train only**
    - transforms val / test with the **same** fitted vectorizer
    - combines TF-IDF + numeric features with sparse hstack

    Test matrices are ``None`` when ``test_df`` is omitted or empty.
    """
    start = time.perf_counter()
    train_base = _build_base_feature_frame(train_df)
    val_base = _build_base_feature_frame(val_df)
    test_nonempty = test_df is not None and len(test_df) > 0
    test_base = _build_base_feature_frame(test_df) if test_nonempty else None
    _log(
        f"prepared base frames: train={len(train_base)} rows, val={len(val_base)} rows"
        + (f", test={len(test_base)}" if test_base is not None else ""),
        verbose=verbose,
    )

    vectorizer = _get_vectorizer()
    _log("fitting TF-IDF on train combined text", verbose=verbose)
    X_train_tfidf = vectorizer.fit_transform(train_base["combined_text"])
    _log("transforming val combined text with fitted TF-IDF", verbose=verbose)
    X_val_tfidf = vectorizer.transform(val_base["combined_text"])
    _log(f"TF-IDF vocab size: {len(vectorizer.vocabulary_)}", verbose=verbose)

    train_numeric = csr_matrix(_compute_numeric_features(train_base, vectorizer, verbose=verbose).values)
    val_numeric = csr_matrix(_compute_numeric_features(val_base, vectorizer, verbose=verbose).values)

    X_train = hstack([X_train_tfidf, train_numeric], format="csr")
    X_val = hstack([X_val_tfidf, val_numeric], format="csr")

    y_train = train_base["is_correct"].astype(int).to_numpy()
    y_val = val_base["is_correct"].astype(int).to_numpy()

    X_test: Optional[csr_matrix] = None
    y_test: Optional[np.ndarray] = None
    if test_base is not None:
        _log("transforming test combined text with fitted TF-IDF", verbose=verbose)
        X_test_tfidf = vectorizer.transform(test_base["combined_text"])
        test_numeric = csr_matrix(_compute_numeric_features(test_base, vectorizer, verbose=verbose).values)
        X_test = hstack([X_test_tfidf, test_numeric], format="csr")
        y_test = test_base["is_correct"].astype(int).to_numpy()

    elapsed = time.perf_counter() - start
    msg = (
        f"feature matrices built in {elapsed:.1f}s | "
        f"X_train={X_train.shape}, X_val={X_val.shape}"
    )
    if X_test is not None:
        msg += f", X_test={X_test.shape}"
    _log(msg, verbose=verbose)

    return X_train, X_val, y_train, y_val, vectorizer, X_test, y_test


def save_feature_artifacts(
    X_train: csr_matrix,
    X_val: csr_matrix,
    y_train: np.ndarray,
    y_val: np.ndarray,
    vectorizer: TfidfVectorizer,
    project_root: Path | str = Path("."),
    X_test: csr_matrix | None = None,
    y_test: np.ndarray | None = None,
) -> None:
    """Save vectorizer and train/val (+ optional test) features to required paths."""
    root = Path(project_root)
    vectorizer_path = root / "models" / "model_a" / "traditional" / "tfidf_vectorizer.pkl"
    processed_dir = root / "data" / "processed"

    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    with vectorizer_path.open("wb") as f:
        pickle.dump(vectorizer, f)

    from scipy.sparse import save_npz

    save_npz(processed_dir / "X_train_features.npz", X_train)
    save_npz(processed_dir / "X_val_features.npz", X_val)
    np.save(processed_dir / "y_train.npy", y_train)
    np.save(processed_dir / "y_val.npy", y_val)
    if X_test is not None and y_test is not None:
        save_npz(processed_dir / "X_test_features.npz", X_test)
        np.save(processed_dir / "y_test.npy", y_test)


def run_feature_pipeline(
    train_csv: Path | str = Path("data/processed/train_verification.csv"),
    val_csv: Path | str = Path("data/processed/val_verification.csv"),
    test_csv: Path | str | None = Path("data/processed/test_verification.csv"),
    project_root: Path | str = Path("."),
    verbose: bool = True,
) -> None:
    """Load processed splits, build features, and persist artifacts (including test when CSV exists)."""
    start = time.perf_counter()
    root = Path(project_root)
    _log(f"loading train data from {train_csv}", verbose=verbose)
    train_df = pd.read_csv(train_csv)
    _log(f"loading val data from {val_csv}", verbose=verbose)
    val_df = pd.read_csv(val_csv)

    test_df: pd.DataFrame | None = None
    test_path = Path(test_csv) if test_csv is not None else None
    if test_path is not None and test_path.exists():
        _log(f"loading test data from {test_path}", verbose=verbose)
        test_df = pd.read_csv(test_path)
    else:
        _log(
            "test_verification.csv not found — skipping test features (optional)",
            verbose=verbose,
        )

    X_train, X_val, y_train, y_val, vectorizer, X_test, y_test = build_feature_matrices(
        train_df, val_df, test_df=test_df, verbose=verbose
    )
    _log("saving vectorizer and feature artifacts", verbose=verbose)
    save_feature_artifacts(
        X_train,
        X_val,
        y_train,
        y_val,
        vectorizer,
        project_root=project_root,
        X_test=X_test,
        y_test=y_test,
    )

    print("Saved feature artifacts:")
    print("models/model_a/traditional/tfidf_vectorizer.pkl")
    print("data/processed/X_train_features.npz")
    print("data/processed/X_val_features.npz")
    print("data/processed/y_train.npy")
    print("data/processed/y_val.npy")
    if X_test is not None and y_test is not None:
        print("data/processed/X_test_features.npz")
        print("data/processed/y_test.npy")
    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
    print(f"y_train positives: {int(y_train.sum())} / {len(y_train)}")
    print(f"y_val positives: {int(y_val.sum())} / {len(y_val)}")
    if X_test is not None:
        print(f"X_test shape: {X_test.shape}, y_test positives: {int(y_test.sum())} / {len(y_test)}")
    _log(f"pipeline completed in {time.perf_counter() - start:.1f}s", verbose=verbose)


if __name__ == "__main__":
    run_feature_pipeline(project_root=PROJECT_ROOT)
