#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import re
import nltk
import joblib
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

nltk.download("wordnet")
nltk.download("omw-1.4")

# -----------------------------
# Preprocessing
# -----------------------------
def load_data(filepath):
    df = pd.read_csv(filepath).dropna(subset=["names", "overview"])
    df = df.drop_duplicates(subset=["names", "overview"]).reset_index(drop=True)
    print(df.shape)
    return df

def preprocess_text(text):
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.lower().strip()
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in text.split()]
    return " ".join(words)

# -----------------------------
# Train and Save Model
# -----------------------------
def train_and_save_model(df, model_path="amrs_tfidf.pkl"):
    texts = df["overview"].tolist()

    # Define TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        ngram_range=(1,2),
        max_features=50000,
        strip_accents="unicode",
        min_df=2,
        analyzer="word",
        stop_words="english"
    )

    X = tfidf.fit_transform(texts)

    # Search best K for KMeans
    best_score, best_k, best_model = -1, None, None
    for k in range(5, 50):  # test K from 5 to 15
        kmeans = KMeans(n_clusters=k, random_state=42, init="k-means++",n_init="auto",tol=1e-4,algorithm="lloyd")
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        print(f"K={k}, Silhouette={score:.4f}")
        if score > best_score:
            best_score, best_k, best_model = score, k, kmeans

    print(f"\n✅ Best K={best_k}, Silhouette={best_score:.4f}")

    # Build final pipeline with best K
    pipe = Pipeline([
        ("tfidf", tfidf),
        ("kmeans", best_model)
    ])

    # Fit pipeline fully
    pipe.fit(texts)

    # Save pipeline and dataframe
    joblib.dump((pipe, df), model_path)
    print(f"🎉 Model saved to {model_path}")

if __name__ == "__main__":
    dataset_path = "imdb_movies.csv"
    df = load_data(dataset_path)
    df["overview"] = df["overview"].apply(preprocess_text)
    train_and_save_model(df, "amrs_tfidf.pkl")