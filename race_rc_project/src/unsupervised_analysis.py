"""Unsupervised Analysis for RACE RC - K-Means Clustering for Passage Categorization."""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

def run_unsupervised_clustering(n_clusters=5):
    print("Starting Unsupervised Analysis (K-Means Clustering)...")
    
    # Load a subset of data for clustering
    try:
        df = pd.read_csv("data/splits/train.csv").head(5000)
    except:
        df = pd.read_csv("data/raw/train_verification.csv").head(5000)
        
    print(f"Clustering {len(df)} passages into {n_clusters} topics...")
    
    # Vectorize passages
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df['article'])
    
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    df['cluster'] = clusters
    
    # Identify top words per cluster
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    
    cluster_keywords = {}
    for i in range(n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        cluster_keywords[i] = top_terms
        print(f"Cluster {i} Keywords: {', '.join(top_terms)}")
    
    # Save Model and Keywords
    report_dir = Path("models/unsupervised")
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans, report_dir / "kmeans_model.pkl")
    joblib.dump(vectorizer, report_dir / "unsupervised_vectorizer.pkl")
    
    # Visualization (PCA to 2D)
    print("Generating Clustering Visualization...")
    pca = PCA(n_components=2)
    scatter_plot_points = pca.fit_transform(X.toarray())
    
    plt.figure(figsize=(10, 7))
    plt.scatter(scatter_plot_points[:, 0], scatter_plot_points[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    plt.title("Unsupervised Passage Clustering (K-Means + PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    
    viz_dir = Path("reports/visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(viz_dir / "passage_clusters.png")
    print(f"Visualization saved to {viz_dir}/passage_clusters.png")
    
    return cluster_keywords

if __name__ == "__main__":
    run_unsupervised_clustering()
