import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import joblib
import sys

# Add src to path to import model_b
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.model_b import DistractorGenerator

def train():
    model_dir = Path("models/model_b/traditional")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data for training Model B distractor ranker...")
    try:
        # We load the full processed dataset
        df = pd.read_csv("data/processed/train_verification.csv")
    except FileNotFoundError:
        print("Data not found. Please ensure data/processed/train_verification.csv exists.")
        return

    # Extract the correct option text for each question group (sample_id)
    # The feature generator needs the correct answer text to compute similarity metrics
    correct_mask = df["is_correct"] == 1
    correct_df = df[correct_mask][["sample_id", "option_text"]].rename(columns={"option_text": "correct_text"})
    
    # Merge correct texts back into the main dataframe
    df = df.merge(correct_df, on="sample_id", how="left")
    df = df.dropna(subset=["correct_text", "option_text"])
    
    # Sample 5000 unique questions (~20,000 options) so NLTK loop doesn't take 3 hours
    unique_samples = df["sample_id"].unique()
    np.random.seed(42)
    sampled_ids = np.random.choice(unique_samples, size=min(5000, len(unique_samples)), replace=False)
    df = df[df["sample_id"].isin(sampled_ids)]
    
    generator = DistractorGenerator()
    
    X = []
    y = []
    
    print(f"Extracting features from {len(df)} options... (this may take a minute)")
    for idx, row in df.reset_index(drop=True).iterrows():
        # Correct answer = bad distractor (0)
        # Wrong options = good distractors (1)
        label = 0 if row["is_correct"] == 1 else 1
        
        feats = generator._extract_features(
            candidate=str(row["option_text"]),
            correct_answer=str(row["correct_text"]),
            question=str(row["question"]),
            passage=str(row["article"])
        )
        X.append(feats)
        y.append(label)
        
        if idx > 0 and idx % 2500 == 0:
            print(f"Processed {idx} rows...")

    X = np.array(X)
    y = np.array(y)
    
    print("Training Random Forest ranker...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    
    model_path = model_dir / "distractor_ranker.pkl"
    joblib.dump(clf, model_path)
    print(f"Saved distractor ranker to {model_path}")

if __name__ == "__main__":
    train()
