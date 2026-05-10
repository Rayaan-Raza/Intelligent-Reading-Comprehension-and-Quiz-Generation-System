"""Phase 13: Evaluation and Experiments Script."""

import pandas as pd
import numpy as np
import time
from pathlib import Path
import sys
import json

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.inference import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_evaluation(num_samples=50):
    print("Starting Phase 13 Evaluation...")
    pipe = Pipeline()
    
    # Load test data
    try:
        df = pd.read_csv("data/splits/test.csv").head(num_samples)
    except FileNotFoundError:
        print("Test split not found. Using val split.")
        df = pd.read_csv("data/splits/val.csv").head(num_samples)

    results = []
    latencies = []
    
    print(f"Evaluating on {len(df)} samples...")
    
    for idx, row in df.iterrows():
        article = row['article']
        question = row['question']
        ground_truth = row['answer']
        options = {
            "A": row['A'],
            "B": row['B'],
            "C": row['C'],
            "D": row['D']
        }
        
        start = time.perf_counter()
        res = pipe.run_pipeline(article, question, options, correct_answer=ground_truth, model_a_name="logistic_regression")
        latencies.append(time.perf_counter() - start)
        
        prediction = res['predicted_answer']
        
        results.append({
            "id": row['id'],
            "actual": ground_truth,
            "predicted": prediction,
            "correct": ground_truth == prediction,
            "max_confidence": max(res['confidence_scores'].values()),
            "avg_confidence": np.mean(list(res['confidence_scores'].values()))
        })
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(df)}...")

    eval_df = pd.DataFrame(results)
    
    # Model A Metrics
    acc = accuracy_score(eval_df['actual'], eval_df['predicted'])
    
    print("\n--- Model A (Traditional ML) Results ---")
    print(f"MCQ Exact Match Accuracy: {acc:.4f}")
    print(f"Average Inference Latency: {np.mean(latencies):.4f}s")
    
    # ------------------------------------------------------------------------
    # MODEL B: DISTRACTOR EVALUATION
    # ------------------------------------------------------------------------
    print("\nEvaluating Model B (Distractors & Hints)...")
    dist_results = []
    hint_precisions = []

    for idx, row in df.iterrows():
        article = row['article']
        question = row['question']
        ground_truth_letter = row['answer']
        options = {"A": row['A'], "B": row['B'], "C": row['C'], "D": row['D']}
        correct_text = options[ground_truth_letter]
        reference_distractors = [v for k, v in options.items() if k != ground_truth_letter]
        
        # Generate distractors using Model B
        generated_res = pipe.distractor_gen.generate_distractors_custom(article, question, correct_text)
        
        # Recall: How many reference distractors were recovered (similiarity > 0.3)
        recovered = 0
        for ref in reference_distractors:
            ref_words = set(ref.lower().split())
            for gen in generated_res:
                gen_words = set(gen.lower().split())
                if not ref_words or not gen_words: continue
                
                # Jaccard-like overlap
                intersection = ref_words & gen_words
                sim = len(intersection) / len(ref_words)
                
                if sim >= 0.3: # If 30% of words match, it's a "conceptual hit"
                    recovered += 1
                    break
        
        recall = recovered / 3
        
        # ------------------------------------------------------------------------
        # MODEL B: HINT EVALUATION
        # ------------------------------------------------------------------------
        hints = pipe.hint_gen.generate_hints(article, question, correct_text)
        
        # Find gold sentence (sentence containing correct_text)
        sentences = [s.strip() for s in article.split('.') if s.strip()]
        gold_sentence = ""
        for s in sentences:
            if correct_text.lower() in s.lower():
                gold_sentence = s
                break
        
        # Hint Precision: Did any hint (especially Hint 2 or 3) contain words from the gold sentence?
        hit = 0
        if gold_sentence:
            gold_words = set(gold_sentence.lower().split())
            for h in hints:
                hint_words = set(h.lower().split())
                if len(gold_words & hint_words) / len(gold_words) > 0.3:
                    hit = 1
                    break
        hint_precisions.append(hit)
        
        dist_results.append({
            "recall": recall,
            "hints_hit": hit
        })

    dist_df = pd.DataFrame(dist_results)
    avg_recall = dist_df['recall'].mean()
    avg_hint_prec = np.mean(hint_precisions)

    print("\n--- Model B (Generation) Results ---")
    print(f"Distractor Recall (Overlap > 0.3): {avg_recall:.4f}")
    print(f"Hint Extraction Precision: {avg_hint_prec:.4f}")

    # Update Comparison
    comparison = pd.DataFrame([{
        "Metric": "MCQ Exact Match (Model A)",
        "Value": acc
    }, {
        "Metric": "Distractor Recall (Model B)",
        "Value": avg_recall
    }, {
        "Metric": "Hint Precision",
        "Value": avg_hint_prec
    }])
    
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    comparison.to_csv(report_dir / "model_comparison.csv", index=False)
    eval_df.to_csv(report_dir / "final_test_results.csv", index=False)
    dist_df.to_csv(report_dir / "model_b_distractor_results.csv", index=False)
    
    print(f"\nAll reports saved to {report_dir}")
    print("Evaluation Complete!")

if __name__ == "__main__":
    run_evaluation(num_samples=20)

