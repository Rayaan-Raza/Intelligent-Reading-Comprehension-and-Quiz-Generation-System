"""
Run manual checks for Module 8 (Model A question generation).

Usage (from project root `race_rc_project`):

  python scripts/test_module8_question_generation.py

Optional: pass your own passage as a single string argument.

  python scripts/test_module8_question_generation.py "Your passage text here."
"""

from __future__ import annotations

import sys
from pathlib import Path
import random

# Project root: .../race_rc_project
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.model_a_question_generation import (  # noqa: E402
    generate_from_custom_passage,
    generate_from_race_sample,
    generate_question,
)


def _print_block(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def _build_mcq_options(correct_answer: str) -> tuple[dict[str, str], str]:
    """Build a simple 4-option MCQ with one correct answer."""
    correct = (correct_answer or "").strip()
    if not correct:
        correct = "N/A"

    pool = [
        "London",
        "Rome",
        "Berlin",
        "Tokyo",
        "1990",
        "2001",
        "energy",
        "water",
        "gravity",
        "teacher",
        "student",
        "Paris",
    ]
    distractors = [x for x in pool if x.lower() != correct.lower()]
    random.shuffle(distractors)
    options = [correct] + distractors[:3]
    random.shuffle(options)
    labels = ["A", "B", "C", "D"]
    mcq = {label: value for label, value in zip(labels, options)}
    answer_label = next(label for label, value in mcq.items() if value == correct)
    return mcq, answer_label


def _run_interactive_custom_passage() -> None:
    _print_block("Interactive custom passage")
    passage = input("Paste your passage: ").strip()
    if not passage:
        print("No passage entered.")
        return

    g = generate_from_custom_passage(passage)
    mcq, label = _build_mcq_options(g.correct_answer)
    print("Generated question :", g.question)
    print("Source sentence    :", g.source_sentence)
    print("Masked sentence    :", g.masked_sentence)
    print("Template           :", g.template_name)
    print()
    print("MCQ options:")
    for key in ["A", "B", "C", "D"]:
        print(f"  {key}) {mcq[key]}")
    print(f"Correct answer: {label}) {mcq[label]}")


def main() -> None:
    _print_block("Mode 1 - RACE sample (original question from dataset)")
    g1 = generate_from_race_sample("What is the main idea of the passage?")
    print("question :", g1.question)
    print("mode     :", g1.mode)
    print("template :", g1.template_name)

    _print_block("Mode 2 - Custom passage (example: location)")
    demo = (
        "The Eiffel Tower is located in Paris. "
        "Many tourists visit the city every year."
    )
    g2 = generate_from_custom_passage(demo)
    print("source sentence :", g2.source_sentence)
    print("question        :", g2.question)
    print("correct_answer  :", g2.correct_answer)
    print("masked          :", g2.masked_sentence)
    print("template        :", g2.template_name)
    print("mode            :", g2.mode)

    _print_block("Unified API: with race_question (Mode 1)")
    g3 = generate_question(
        "This passage is ignored when RACE question is set.",
        race_question="Why did the author write this text?",
        use_race_question_if_provided=True,
    )
    print("question :", g3.question)
    print("mode     :", g3.mode)

    _print_block("Unified API: no race question (Mode 2)")
    g4 = generate_question(
        "Water boils at 100 degrees Celsius at sea level. It is a physical change.",
        race_question="",
        use_race_question_if_provided=True,
    )
    print("question        :", g4.question)
    print("correct_answer  :", g4.correct_answer)
    print("template        :", g4.template_name)

    if len(sys.argv) > 1:
        custom = " ".join(sys.argv[1:]).strip()
        _print_block("Your custom passage (argv)")
        g5 = generate_from_custom_passage(custom)
        print("question        :", g5.question)
        print("correct_answer  :", g5.correct_answer)
        print("source_sentence :", g5.source_sentence)
        print("template        :", g5.template_name)
        mcq, label = _build_mcq_options(g5.correct_answer)
        print("mcq             :", mcq)
        print(f"correct         : {label}) {mcq[label]}")

    _print_block("Interactive menu")
    print("Press 1: Enter your own passage -> generate question + MCQ + correct answer")
    print("Press 2: Skip (already ran demo output)")
    print("Press 0: Exit")
    choice = input("Your choice: ").strip()
    if choice == "1":
        _run_interactive_custom_passage()
    elif choice == "2":
        print("Skipped interactive custom passage.")
    else:
        print("Exiting.")

    print()
    print("Done.")
    print()


if __name__ == "__main__":
    main()
