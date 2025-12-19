# tfidf_embed_save_light.py
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

DATA_PATH = "/home/detsu/Documents/vscode/NLP/NLP/data/texts_clean.csv"
ART_DIR   = "/home/detsu/Documents/vscode/NLP/NLP/artifacts"

X_PATH   = os.path.join(ART_DIR, "tfidf_embeddings.npz")   # ⬅️ sparse
Y_PATH   = os.path.join(ART_DIR, "tfidf_labels.npy")
VEC_PATH = os.path.join(ART_DIR, "tfidf_vectorizer.pkl")

def main():
    os.makedirs(ART_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].values

    vectorizer = TfidfVectorizer(
        max_features=10000,   # ⬅️ 20k → 10k (RAM 2× багасна)
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        norm="l2"
    )

    print("Building TF-IDF (sparse)...")
    X = vectorizer.fit_transform(texts)   # ❌ toarray() байхгүй

    sparse.save_npz(X_PATH, X)
    np.save(Y_PATH, labels)

    with open(VEC_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print("Saved:", X_PATH, X.shape)
    print("Saved:", Y_PATH, labels.shape)
    print("Saved:", VEC_PATH)

if __name__ == "__main__":
    main()
