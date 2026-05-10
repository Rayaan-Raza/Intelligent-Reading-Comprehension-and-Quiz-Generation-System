"""EDA Visualization Script for RACE RC Project."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def generate_eda_plots():
    print("Generating EDA Visualizations...")
    
    # Load dataset
    try:
        df = pd.read_csv("data/raw/train_verification.csv")
    except:
        df = pd.read_csv("data/splits/train.csv")
    
    viz_dir = Path("reports/visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Distribution of Answer Keys
    plt.figure(figsize=(8, 6))
    sns.countplot(x='answer', data=df, palette='viridis')
    plt.title("Distribution of Correct Answer Keys (A, B, C, D)")
    plt.savefig(viz_dir / "answer_distribution.png")
    
    # 2. Article Length Distribution
    df['article_len'] = df['article'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(10, 6))
    sns.histplot(df['article_len'], bins=50, kde=True, color='#C47FA8')
    plt.title("Distribution of Passage Lengths (Word Count)")
    plt.xlabel("Word Count")
    plt.savefig(viz_dir / "passage_length_dist.png")
    
    # 3. Correlation between Length and (Hypothetical) Complexity
    # For now, just show the distribution of question lengths
    df['question_len'] = df['question'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(10, 6))
    sns.jointplot(x='article_len', y='question_len', data=df, kind='hex', color='#8B5A8B')
    plt.savefig(viz_dir / "length_correlation.png")
    
    print(f"EDA Plots saved to {viz_dir}")

if __name__ == "__main__":
    generate_eda_plots()
