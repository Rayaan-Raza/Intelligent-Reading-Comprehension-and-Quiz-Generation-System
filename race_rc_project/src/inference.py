"""Unified Inference Pipeline tying together Model A and Model B."""

from __future__ import annotations

import sys
import time
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.model_b import DistractorGenerator, HintGenerator
from src.question_ranking import predict_mcq_answer, generate_and_rank_questions
from src.model_a_question_generation import GeneratedQuestion

class Pipeline:
    def __init__(self, project_root: Path | None = None):
        self.root = project_root if project_root is not None else PROJECT_ROOT
        
        # Load Model B ranker
        self.distractor_ranker = None
        model_b_path = self.root / "models" / "model_b" / "traditional" / "distractor_ranker.pkl"
        if model_b_path.exists():
            try:
                self.distractor_ranker = joblib.load(model_b_path)
            except Exception as e:
                print(f"Warning: Failed to load Distractor Ranker: {e}")
        
        # Initialize generators
        self.distractor_gen = DistractorGenerator(ranker_model=self.distractor_ranker)
        self.hint_gen = HintGenerator()

    def run_pipeline(
        self, 
        article: str, 
        question: Optional[str] = None, 
        options: Optional[Dict[str, str]] = None,
        correct_answer: Optional[str] = None,
        model_a_name: str = "logistic_regression"
    ) -> Dict[str, Any]:
        """
        Unified inference connecting article to question, options, predictions, and hints.
        """
        start_time = time.perf_counter()
        
        # 1. Generate or accept question
        if not question:
            # Generate top question using Model A
            try:
                ranked_qs = generate_and_rank_questions(article, top_k_candidates=1, ranker=model_a_name, project_root=self.root)
                if ranked_qs:
                    question = ranked_qs[0][0].question
                    correct_answer = ranked_qs[0][0].correct_answer
                else:
                    question = "Failed to generate question."
            except Exception as e:
                question = f"Error generating question: {e}"
        else:
            # Assuming if question is provided, we either have options or we need to extract a correct answer somehow
            # For simplicity, if custom question, but no options, we assume a dummy correct answer or require options.
            pass

        # 2. Generate or accept options
        # `correct_answer` may currently be either:
        #   - a letter ("A"/"B"/"C"/"D") when the caller passed RACE-style options
        #   - the answer text (e.g. "curie") when Model A generated the question
        #   - None
        # We normalize it to a letter (`correct_letter`) and a text (`correct_text`)
        # so downstream code (and the UI) can rely on a consistent shape.
        if not options:
            if not correct_answer:
                correct_answer = "Default Answer"
            distractors = self.distractor_gen.generate_distractors_custom(article, question, correct_answer)
            
            all_opts = distractors + [correct_answer]
            random.shuffle(all_opts)
            
            options = {
                "A": all_opts[0],
                "B": all_opts[1],
                "C": all_opts[2],
                "D": all_opts[3]
            }
            correct_text = correct_answer
            correct_letter = next(
                (k for k, v in options.items() if v == correct_answer),
                None,
            )
        else:
            if correct_answer in options:
                correct_letter = correct_answer
                correct_text = options[correct_answer]
            elif correct_answer is not None:
                correct_letter = next(
                    (k for k, v in options.items() if str(v) == str(correct_answer)),
                    None,
                )
                correct_text = correct_answer
            else:
                correct_letter = None
                correct_text = None

        # 3. Score options and pick predicted answer using Model A
        try:
            predicted_answer, confidence_scores = predict_mcq_answer(
                article=article,
                question=question,
                options=options,
                model_name=model_model_a_map(model_a_name),
                project_root=self.root
            )
        except Exception as e:
            predicted_answer = "A"
            confidence_scores = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
            print(f"Error in prediction: {e}")

        # Pick the best text we have to seed hint generation.
        hint_answer_text = correct_text or options.get(predicted_answer, "")

        # 4. Generate hints
        try:
            hints = self.hint_gen.generate_hints(article, question, hint_answer_text)
        except Exception as e:
            hints = ["Hint 1 unavailable.", "Hint 2 unavailable.", "Hint 3 unavailable."]
            print(f"Error generating hints: {e}")

        inference_time = time.perf_counter() - start_time

        return {
            "article": article,
            "question": question,
            "options": options,
            "actual_answer": correct_letter,
            "actual_answer_text": correct_text,
            "predicted_answer": predicted_answer,
            "confidence_scores": confidence_scores,
            "hints": hints,
            "inference_time": inference_time
        }

def model_model_a_map(name: str) -> str:
    """Helper to map ranker names if needed"""
    valid = ["random_forest", "linear_svm", "logistic_regression", "xgboost", "naive_bayes", "mlp_neural", "ensemble"]
    if name in valid:
        return name
    return "random_forest"

# Add convenience method so UI can just call run_pipeline directly without managing the class
_GLOBAL_PIPELINE = None

def run_pipeline(
    article: str, 
    question: Optional[str] = None, 
    options: Optional[Dict[str, str]] = None,
    correct_answer: Optional[str] = None,
    model_a_name: str = "logistic_regression"
) -> Dict[str, Any]:
    global _GLOBAL_PIPELINE
    if _GLOBAL_PIPELINE is None:
        _GLOBAL_PIPELINE = Pipeline()
    return _GLOBAL_PIPELINE.run_pipeline(article, question, options, correct_answer, model_a_name)

if __name__ == "__main__":
    # Test script
    pipe = Pipeline()
    res = pipe.run_pipeline("The quick brown fox jumps over the lazy dog.", "Who jumps over the dog?", {"A": "fox", "B": "cat", "C": "bird", "D": "fish"})
    print(res)
