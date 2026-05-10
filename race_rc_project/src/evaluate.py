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
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

def calculate_generation_metrics(reference, hypothesis):
    """Calculate BLEU, ROUGE-L, and METEOR scores."""
    if not reference or not hypothesis:
        return {"bleu": 0.0, "rougeL": 0.0, "meteor": 0.0}
    
    # Tokenization
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    
    # BLEU (with smoothing for short sequences)
    smoothing = SmoothingFunction().method1
    bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
    
    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, hypothesis)
    rougel = rouge_scores['rougeL'].fmeasure
    
    # METEOR
    try:
        meteor = meteor_score([ref_tokens], hyp_tokens)
    except:
        meteor = 0.0
        
    return {
        "bleu": bleu,
        "rougeL": rougel,
        "meteor": meteor
    }

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
        
        # --- NEW: Question Generation Evaluation ---
        # Generate a question without providing one to the pipeline
        gen_start = time.perf_counter()
        gen_res = pipe.run_pipeline(article, model_a_name="logistic_regression")
        gen_q = gen_res['question']
        q_metrics = calculate_generation_metrics(question, gen_q)
        
        results.append({
            "id": row['id'],
            "actual": ground_truth,
            "predicted": prediction,
            "correct": ground_truth == prediction,
            "max_confidence": max(res['confidence_scores'].values()),
            "avg_confidence": np.mean(list(res['confidence_scores'].values())),
            "q_bleu": q_metrics['bleu'],
            "q_rougeL": q_metrics['rougeL'],
            "q_meteor": q_metrics['meteor']
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
        
        # Conceptual Recall: Did any generated keyword appear in the reference distractor?
        recovered = 0
        for ref in reference_distractors:
            ref_words = set(ref.lower().split())
            for gen in generated_res:
                gen_words = set(gen.lower().split())
                if not ref_words or not gen_words: continue
                
                # Check if any non-trivial word from gen is in ref
                if any(w in ref_words for w in gen_words if len(w) > 2):
                    recovered += 1
                    break
        
        conceptual_recall = recovered / 3
        
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
        
        # ------------------------------------------------------------------------
        # MODEL B: DISTRACTOR EVALUATION (Enhanced with Metrics)
        # ------------------------------------------------------------------------
        dist_metrics_list = []
        for ref_dist in reference_distractors:
            # Compare each reference distractor with the best matching generated one
            best_m = {"bleu": 0.0, "rougeL": 0.0, "meteor": 0.0}
            for gen_dist in generated_res:
                m = calculate_generation_metrics(ref_dist, gen_dist)
                if m['rougeL'] > best_m['rougeL']:
                    best_m = m
            dist_metrics_list.append(best_m)
        
        avg_dist_bleu = np.mean([m['bleu'] for m in dist_metrics_list])
        avg_dist_rouge = np.mean([m['rougeL'] for m in dist_metrics_list])
        avg_dist_meteor = np.mean([m['meteor'] for m in dist_metrics_list])

        dist_results.append({
            "conceptual_recall": conceptual_recall,
            "hints_hit": hit,
            "dist_bleu": avg_dist_bleu,
            "dist_rougeL": avg_dist_rouge,
            "dist_meteor": avg_dist_meteor
        })

    dist_df = pd.DataFrame(dist_results)
    avg_conceptual_recall = dist_df['conceptual_recall'].mean()
    avg_hint_prec = np.mean(hint_precisions)
    
    # Question Gen Averages
    avg_q_bleu = eval_df['q_bleu'].mean()
    avg_q_rouge = eval_df['q_rougeL'].mean()
    avg_q_meteor = eval_df['q_meteor'].mean()
    
    # Distractor Gen Averages
    avg_d_bleu = dist_df['dist_bleu'].mean()
    avg_d_rouge = dist_df['dist_rougeL'].mean()
    avg_d_meteor = dist_df['dist_meteor'].mean()

    print("\n--- Model B (Generation) Results ---")
    print(f"Conceptual Recall (Keyword Match): {avg_conceptual_recall:.4f}")
    print(f"Distractor BLEU: {avg_d_bleu:.4f}")
    print(f"Distractor ROUGE-L: {avg_d_rouge:.4f}")
    print(f"Distractor METEOR: {avg_d_meteor:.4f}")
    print(f"Hint Extraction Precision: {avg_hint_prec:.4f}")
    
    print("\n--- Question Generation Results ---")
    print(f"Question BLEU: {avg_q_bleu:.4f}")
    print(f"Question ROUGE-L: {avg_q_rouge:.4f}")
    print(f"Question METEOR: {avg_q_meteor:.4f}")

    # Update Comparison
    comparison = pd.DataFrame([{
        "Metric": "MCQ Exact Match (Model A)",
        "Value": acc
    }, {
        "Metric": "Question BLEU",
        "Value": avg_q_bleu
    }, {
        "Metric": "Question ROUGE-L",
        "Value": avg_q_rouge
    }, {
        "Metric": "Question METEOR",
        "Value": avg_q_meteor
    }, {
        "Metric": "Conceptual Recall (Model B)",
        "Value": avg_conceptual_recall
    }, {
        "Metric": "Distractor BLEU",
        "Value": avg_d_bleu
    }, {
        "Metric": "Distractor ROUGE-L",
        "Value": avg_d_rouge
    }, {
        "Metric": "Distractor METEOR",
        "Value": avg_d_meteor
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

