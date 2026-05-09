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
        res = pipe.run_pipeline(article, question, options, correct_answer=ground_truth)
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
    
    # Model B Distractor Metrics (Plausibility)
    # We can measure how often Model A is "tricked" or how high the confidence is for distractors
    avg_max_conf = eval_df['max_confidence'].mean()
    print("\n--- Model B (Distractor Generation) Results ---")
    print(f"Mean Max Confidence Score: {avg_max_conf:.4f} (Higher indicates more plausible options)")
    
    # Save Model Comparison
    comparison = pd.DataFrame([{
        "Metric": "MCQ Exact Match",
        "Model A (Traditional)": acc,
        "Model B (Plausibility)": avg_max_conf
    }])
    
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    comparison.to_csv(report_dir / "model_comparison.csv", index=False)
    eval_df.to_csv(report_dir / "final_test_results.csv", index=False)
    
    print(f"\nReports saved to {report_dir}")
    print("Evaluation Complete!")

if __name__ == "__main__":
    run_evaluation(num_samples=30)
