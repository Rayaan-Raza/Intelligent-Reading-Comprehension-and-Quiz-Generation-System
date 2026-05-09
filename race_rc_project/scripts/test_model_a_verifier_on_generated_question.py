"""
End-to-end demo:
1) Generate a question/answer from passage text.
2) Build MCQ options.
3) Score A/B/C/D with trained Model A verifier.
4) Print prediction + correctness.

Run (from race_rc_project):
  python scripts/test_model_a_verifier_on_generated_question.py
"""

from __future__ import annotations

import random
import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.model_a_question_generation import generate_from_custom_passage  # noqa: E402
from src.question_ranking import predict_mcq_answer  # noqa: E402

ALL_MODELS = [
    "logistic_regression",
    "linear_svm",
    "naive_bayes",
    "random_forest",
    "xgboost",
    "mlp_neural",
    "ensemble",
]
DEMO_PASSAGES = [
    "the bells rang at 2pm and school was over",
    "the eiffel tower is located in paris and many tourists visit every year",
    "water boils at 100 degrees celsius at sea level",
    "the first moon landing happened in 1969",
    "albert einstein developed the theory of relativity",
]
RACE_EXAMPLE_COUNT = 3


def _print_block(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def _build_mcq_options(correct_answer: str) -> tuple[dict[str, str], str]:
    """Create A/B/C/D options with one correct answer."""
    correct = (correct_answer or "").strip()
    if not correct:
        correct = "N/A"
    pool = [
        "teacher",
        "student",
        "London",
        "energy",
        "gravity",
        "history",
        "science",
        "Paris",
        "water",
        "Tokyo",
    ]
    distractors = [x for x in pool if x.lower() != correct.lower()]
    random.shuffle(distractors)
    options = [correct] + distractors[:3]
    random.shuffle(options)
    labels = ["A", "B", "C", "D"]
    labeled = {label: options[i] for i, label in enumerate(labels)}
    correct_label = next(label for label, text in labeled.items() if text == correct)
    return labeled, correct_label


def main() -> None:
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    random.seed(42)
    stats = {
        model: {"correct": 0, "total": 0, "skipped": 0}
        for model in ALL_MODELS
    }

    for idx, article in enumerate(DEMO_PASSAGES, start=1):
        generated = generate_from_custom_passage(article)
        options, gold_label = _build_mcq_options(generated.correct_answer)

        _print_block(f"Generated Question Test #{idx}")
        print("Passage:")
        print(article)
        print()
        print("Generated question:")
        print(generated.question)
        print()
        print("Options:")
        for label in ["A", "B", "C", "D"]:
            print(f"{label}) {options[label]}")
        print()
        print(f"Generator correct answer:\n{gold_label}) {options[gold_label]}")

        print("\nModel A Verifier Test (All Trained Models)")
        for model_name in ALL_MODELS:
            print(f"\n[{model_name}]")
            try:
                predicted_label, scores = predict_mcq_answer(
                    article=article,
                    question=generated.question,
                    options=options,
                    model_name=model_name,  # type: ignore[arg-type]
                )
            except Exception as exc:
                print(f"Skipped ({exc})")
                stats[model_name]["skipped"] += 1
                continue

            for label in ["A", "B", "C", "D"]:
                print(f"{label}) {options[label]:<12} -> score: {scores[label]:.4f}")
            is_correct = predicted_label == gold_label
            print("Predicted:", f"{predicted_label}) {options[predicted_label]}")
            print("Result   :", "Correct" if is_correct else "Incorrect")
            stats[model_name]["total"] += 1
            if is_correct:
                stats[model_name]["correct"] += 1

    _print_block("Summary Across All Demo Questions")
    for model_name in ALL_MODELS:
        total = stats[model_name]["total"]
        correct = stats[model_name]["correct"]
        skipped = stats[model_name]["skipped"]
        acc = (correct / total) if total else 0.0
        print(
            f"{model_name:<20} correct={correct}/{total} "
            f"accuracy={acc:.2%} skipped={skipped}"
        )

    race_path = _ROOT / "data" / "processed" / "test_verification.csv"
    if not race_path.exists():
        return

    _print_block(f"RACE Dataset Examples (n={RACE_EXAMPLE_COUNT})")
    race_df = pd.read_csv(race_path)
    group_cols = ["sample_id"] if "sample_id" in race_df.columns else ["id", "question"]
    grouped = race_df.groupby(group_cols, sort=False)
    chosen = list(grouped.groups.keys())[:RACE_EXAMPLE_COUNT]
    race_stats = {model: {"correct": 0, "total": 0, "skipped": 0} for model in ALL_MODELS}

    for i, key in enumerate(chosen, start=1):
        # pandas uses tuple keys for multi-index groups, keep both cases safe
        group_df = grouped.get_group(key)
        article = str(group_df["article"].iloc[0])
        question = str(group_df["question"].iloc[0])
        gold_label = str(group_df["answer"].iloc[0]).strip().upper()
        options = {
            str(lbl): str(txt)
            for lbl, txt in zip(group_df["option_label"], group_df["option_text"])
            if str(lbl) in {"A", "B", "C", "D"}
        }
        for label in ["A", "B", "C", "D"]:
            options.setdefault(label, "")

        _print_block(f"RACE MCQ Test #{i}")
        print("Passage (truncated):")
        print(article[:220] + ("..." if len(article) > 220 else ""))
        print("\nQuestion:")
        print(question)
        print("\nOptions:")
        for label in ["A", "B", "C", "D"]:
            print(f"{label}) {options[label]}")
        print(f"\nGold answer:\n{gold_label}) {options[gold_label]}")
        print("\nModel A Verifier Test (All Trained Models)")

        for model_name in ALL_MODELS:
            print(f"\n[{model_name}]")
            try:
                predicted_label, scores = predict_mcq_answer(
                    article=article,
                    question=question,
                    options=options,
                    model_name=model_name,  # type: ignore[arg-type]
                )
            except Exception as exc:
                print(f"Skipped ({exc})")
                race_stats[model_name]["skipped"] += 1
                continue

            for label in ["A", "B", "C", "D"]:
                print(f"{label}) {options[label]:<12} -> score: {scores[label]:.4f}")
            is_correct = predicted_label == gold_label
            print("Predicted:", f"{predicted_label}) {options[predicted_label]}")
            print("Result   :", "Correct" if is_correct else "Incorrect")
            race_stats[model_name]["total"] += 1
            if is_correct:
                race_stats[model_name]["correct"] += 1

    _print_block("Summary Across RACE Examples")
    for model_name in ALL_MODELS:
        total = race_stats[model_name]["total"]
        correct = race_stats[model_name]["correct"]
        skipped = race_stats[model_name]["skipped"]
        acc = (correct / total) if total else 0.0
        print(
            f"{model_name:<20} correct={correct}/{total} "
            f"accuracy={acc:.2%} skipped={skipped}"
        )


if __name__ == "__main__":
    main()

