# sbert_embed_save.py
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATA_PATH = "/home/detsu/Documents/vscode/NLP/NLP/data/texts_clean.csv"
ART_DIR   = "/home/detsu/Documents/vscode/NLP/NLP/artifacts"

EMB_PATH = os.path.join(ART_DIR, "sbert_cls.npy")
Y_PATH   = os.path.join(ART_DIR, "sbert_labels.npy")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32   # SBERT хөнгөн

def main():
    os.makedirs(ART_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].values

    model = SentenceTransformer(MODEL_NAME)

    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="SBERT embedding"):
        batch = texts[i:i+BATCH_SIZE]
        emb = model.encode(
            batch,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False
        )
        embeddings.append(emb)

    X = np.vstack(embeddings).astype(np.float32)

    np.save(EMB_PATH, X)
    np.save(Y_PATH, labels)

    print("Saved:", EMB_PATH, X.shape)
    print("Saved:", Y_PATH, labels.shape)

if __name__ == "__main__":
    main()
