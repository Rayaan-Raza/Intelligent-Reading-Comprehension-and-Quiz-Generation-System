"""Inference helpers and Model A question generation entrypoints."""

from __future__ import annotations

from typing import Any

from src.model_a_question_generation import (
    GeneratedQuestion,
    generate_from_custom_passage,
    generate_from_race_sample,
    generate_question,
)
from src.question_ranking import (
    generate_and_rank_questions,
    predict_mcq_answer,
    rank_candidates_with_sklearn,
)


def run_inference(model: Any, x: Any) -> Any:
    """Run batch inference."""
    return model.predict(x)


__all__ = [
    "run_inference",
    "GeneratedQuestion",
    "generate_question",
    "generate_from_race_sample",
    "generate_from_custom_passage",
    "rank_candidates_with_sklearn",
    "generate_and_rank_questions",
    "predict_mcq_answer",
]
