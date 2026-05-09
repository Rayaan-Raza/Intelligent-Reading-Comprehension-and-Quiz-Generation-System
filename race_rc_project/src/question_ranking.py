"""Rank generated question candidates with trained sklearn verification models (SVM / RF / LogReg)."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.features import transform_verification_dataframe
from src.model_a_question_generation import GeneratedQuestion

RankerName = Literal["random_forest", "linear_svm", "logistic_regression"]
VerifierModelName = Literal[
    "random_forest",
    "linear_svm",
    "logistic_regression",
    "xgboost",
    "naive_bayes",
]

_RANKER_FILES: dict[RankerName, str] = {
    "random_forest": "random_forest_model.pkl",
    "linear_svm": "svm_model.pkl",
    "logistic_regression": "logreg_model.pkl",
}
_VERIFIER_FILES: dict[VerifierModelName, str] = {
    "random_forest": "random_forest_model.pkl",
    "linear_svm": "svm_model.pkl",
    "logistic_regression": "logreg_model.pkl",
    "xgboost": "xgboost_model.pkl",
    "naive_bayes": "naive_bayes_model.pkl",
}


def _load_training_vectorizer(project_root: Path) -> Any:
    path = project_root / "models" / "model_a" / "traditional" / "tfidf_vectorizer.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python src/features.py"
        )
    with path.open("rb") as f:
        return pickle.load(f)


def _load_sklearn_classifier(ranker: RankerName, project_root: Path) -> Any:
    fname = _RANKER_FILES[ranker]
    path = project_root / "models" / "model_a" / "traditional" / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python src/train_model_a.py"
        )
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


def _verification_rows_for_candidates(
    passage: str,
    candidates: List[GeneratedQuestion],
) -> pd.DataFrame:
    rows = []
    for i, c in enumerate(candidates):
        ans = (c.correct_answer or "").strip()
        if not ans:
            continue
        rows.append(
            {
                "sample_id": f"qg_rank_{i}",
                "id": "custom_qg",
                "article": passage,
                "question": c.question,
                "option_label": "A",
                "option_text": ans,
                "answer": "A",
                "is_correct": 1,
            }
        )
    if not rows:
        raise ValueError(
            "No candidates with non-empty correct_answer; cannot build verification features."
        )
    return pd.DataFrame(rows)


def classifier_scores(model: Any, X: Any) -> np.ndarray:
    """Higher = more plausible as 'correct option' under the verification model."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1].astype(np.float64)
        except Exception:
            pass
    return model.decision_function(X).astype(np.float64)


def rank_candidates_with_sklearn(
    passage: str,
    candidates: List[GeneratedQuestion],
    *,
    ranker: RankerName = "random_forest",
    project_root: Path | None = None,
) -> List[Tuple[GeneratedQuestion, float]]:
    """
    Score each candidate by treating ``(passage, generated_question, answer_span)`` as one
    verification row — same representation as training — using a **fitted** TF-IDF + classifier.

    Returns candidates sorted by score **descending**.
    """
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    short_rows = [c for c in candidates if (c.correct_answer or "").strip()]
    if not short_rows:
        raise ValueError("No rankable candidates (need correct_answer).")

    df = _verification_rows_for_candidates(passage, short_rows)
    vectorizer = _load_training_vectorizer(root)
    model = _load_sklearn_classifier(ranker, root)

    X = transform_verification_dataframe(df, vectorizer, verbose=False)
    scores = classifier_scores(model, X)

    scored = list(zip(short_rows, scores))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


def generate_and_rank_questions(
    passage: str,
    *,
    top_k_candidates: int = 5,
    ranker: RankerName = "random_forest",
    project_root: Path | None = None,
) -> List[Tuple[GeneratedQuestion, float]]:
    """
    Enumerate template candidates with ``enumerate_candidate_questions``, then rank with sklearn.
    """
    from src.model_a_question_generation import enumerate_candidate_questions

    cands = enumerate_candidate_questions(passage, top_k=top_k_candidates)
    return rank_candidates_with_sklearn(
        passage, cands, ranker=ranker, project_root=project_root
    )


def predict_mcq_answer(
    article: str,
    question: str,
    options: Dict[str, str],
    model_name: VerifierModelName = "random_forest",
    project_root: Path | None = None,
) -> Tuple[str, Dict[str, float]]:
    """
    Score A/B/C/D options with a trained Model A verifier and return top choice.

    Important: this uses the exact same feature path as training
    (TF-IDF + numeric/similarity features = 20,013 dims in this project),
    via `transform_verification_dataframe(...)`.
    """
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    required_labels = ["A", "B", "C", "D"]
    missing = [label for label in required_labels if label not in options]
    if missing:
        raise ValueError(f"options is missing labels: {missing}. Required: A/B/C/D")

    rows = []
    for label in required_labels:
        rows.append(
            {
                "sample_id": "custom_mcq_0",
                "id": "custom_demo",
                "article": str(article),
                "question": str(question),
                "option_label": label,
                "option_text": str(options[label]),
                "answer": "A",
                "is_correct": 0,
            }
        )
    df = pd.DataFrame(rows)

    vectorizer = _load_training_vectorizer(root)
    model_path = root / "models" / "model_a" / "traditional" / _VERIFIER_FILES[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}. Run: python src/train_model_a.py")
    model_obj = joblib.load(model_path)
    model = model_obj["model"] if isinstance(model_obj, dict) and "model" in model_obj else model_obj

    X = transform_verification_dataframe(df, vectorizer, verbose=False)
    raw_scores = classifier_scores(model, X)
    score_map = {label: float(raw_scores[idx]) for idx, label in enumerate(required_labels)}
    predicted_label = max(score_map, key=score_map.get)
    return predicted_label, score_map
