# w2v_sg_embed_save.py
import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

DATA_PATH = "/home/detsu/Documents/vscode/NLP/NLP/data/texts_clean.csv"
ART_DIR   = "/home/detsu/Documents/vscode/NLP/NLP/artifacts"

X_PATH = os.path.join(ART_DIR, "w2v_sg_embeddings.npy")
Y_PATH = os.path.join(ART_DIR, "w2v_sg_labels.npy")
MODEL_PATH = os.path.join(ART_DIR, "w2v_sg.model")

VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 2
SG = 1              # 👉 Skip-gram

def main():
    os.makedirs(ART_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].values

    tokenized = [t.split() for t in texts]

    print("Training Word2Vec (Skip-gram)...")
    w2v = Word2Vec(
        sentences=tokenized,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=SG,
        workers=4
    )

    # sentence embedding = average word vectors
    X = []
    for sent in tokenized:
        vecs = [w2v.wv[w] for w in sent if w in w2v.wv]
        if len(vecs) == 0:
            X.append(np.zeros(VECTOR_SIZE))
        else:
            X.append(np.mean(vecs, axis=0))

    X = np.array(X, dtype=np.float32)

    np.save(X_PATH, X)
    np.save(Y_PATH, labels)
    w2v.save(MODEL_PATH)

    print("Saved:", X_PATH, X.shape)
    print("Saved:", Y_PATH, labels.shape)
    print("Saved:", MODEL_PATH)

if __name__ == "__main__":
    main()
