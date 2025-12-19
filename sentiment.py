
"""
sentiment_realtime.py — Real-time training feedback
Displays live model progress with timestamps and metrics updates.
"""
import argparse, time, sys, os, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

try:
    from gensim.models import Word2Vec
except ImportError:
    Word2Vec = None

def nowstamp():
    return time.strftime("%H:%M:%S")


def print_step(step, msg):
    print(f"[{nowstamp()}] [STEP {step}] {msg}")
    sys.stdout.flush()


def train_word2vec(texts, vector_size=100, window=5, min_count=2, sg=1):
    print_step(2, "Training Word2Vec embeddings ...")
    from gensim.models import Word2Vec
    model = Word2Vec([t.split() for t in texts],
                     vector_size=vector_size, window=window, min_count=min_count, sg=sg)
    return model



def to_avg_vectors(texts, model):
    print_step(3, "Converting text to sentence embeddings ...")
    
    max_len = max(len(t.split()) for t in texts)
    vec_size = model.vector_size

    X = []

    for sent in [t.split() for t in texts]:
        vectors = []

        for w in sent:
            if w in model.wv:
                vectors.append(model.wv[w])
            else:
                vectors.append(np.zeros(vec_size))

        while len(vectors) < max_len:
            vectors.append(np.zeros(vec_size))

        X.append(vectors)   

    return np.array(X)






    return np.array(X)


def train_and_eval(X, y, out_root, args):
    print_step(4, "Splitting train/test data ...")
    results_dir, logs_dir = out_root/"results", out_root/"logs"
    results_dir.mkdir(exist_ok=True), logs_dir.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y)

    print_step(5, "Initializing models ...")
    models = {
        "LogisticRegression": LogisticRegression(max_iter=args.max_iter),
        "RandomForest": RandomForestClassifier(n_estimators=args.rf_estimators,
                                               random_state=args.random_state,
                                               n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(n_estimators=args.ada_estimators,
                                       learning_rate=args.ada_lr,
                                       random_state=args.random_state)
    }

    records, loglines = [], []
    for i, (name, model) in enumerate(models.items(), 1):
        print_step(6, f"Training model {i}/3 → {name} ...")
        t0 = time.time()
        model.fit(X_train, y_train)
        duration = time.time() - t0
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        p_, r_, f1_, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
        print(f"   ↳ {name} done in {duration:.2f}s | acc={acc:.4f} | f1={f1_:.4f}")
        cm = confusion_matrix(y_test, pred)
        pd.DataFrame(cm).to_csv(results_dir/f"confmat_{name.lower()}.csv", index=False, header=False)
        records.append([name, acc, p_, r_, f1_])
        loglines.append(f"{name}: acc={acc:.4f} p={p_:.4f} r={r_:.4f} f1={f1_:.4f}")

    print_step(7, "Saving results ...")
    df = pd.DataFrame(records, columns=["model","accuracy","precision","recall","f1"])
    df.sort_values("accuracy", ascending=False, inplace=True)
    df.to_csv(results_dir/"metrics.csv", index=False)
    best_model = df.iloc[0]["model"]
    shutil.copy(results_dir/f"confmat_{best_model.lower()}.csv", results_dir/"confmat_best.csv")

    log_header = f"Date: {time.ctime()}\nMode: {'Word2Vec' if args.use_word2vec else 'TF-IDF'}\n"
    (logs_dir/f"train_{time.strftime('%Y%m%d_%H%M%S')}.log").write_text(log_header + "\n".join(loglines))
    print_step(8, f"Training complete  Best model → {best_model}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--use_word2vec", action="store_true")
    p.add_argument("--clean_csv", default="data/texts_clean.csv")
    p.add_argument("--features", default="/home/detsu/Documents/vscode/NLP/NLP/artifacts/features.npz")
    p.add_argument("--labels", default="/home/detsu/Documents/vscode/NLP/NLP/artifacts/labels.npy")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--max_iter", type=int, default=1000)
    p.add_argument("--rf_estimators", type=int, default=300)
    p.add_argument("--ada_estimators", type=int, default=200)
    p.add_argument("--ada_lr", type=float, default=0.5)
    args = p.parse_args()

    out_root = Path(".")
    print("="*60)    #hypertuning __ cross validation   _ ingej min 20 udaa ajluulna  ^ Kfold - strait 
    print(" REAL-TIME SENTIMENT TRAINING MONITOR")
    print("="*60)

    print_step(1, f"Loading {'Word2Vec' if args.use_word2vec else 'TF-IDF'} data ...")

    if args.use_word2vec:
        if Word2Vec is None:
            raise ImportError(" gensim not installed. Run: pip install gensim")
        df = pd.read_csv(args.clean_csv)
        w2v_model = train_word2vec(df["text"].tolist())
        X = to_avg_vectors(df["text"].tolist(), w2v_model)
        y = df["label"].values
        w2v_model.save("artifacts/word2vec.model")
    else:
        X = sparse.load_npz(args.features)
        y = np.load(args.labels)
        print(f"[{nowstamp()}] [INFO] TF-IDF features loaded: {X.shape}")

    train_and_eval(X, y, out_root, args)
#ymar taskaas hamaaraad f1 score macro ymuu microgoor unelne 

if __name__ == "__main__":
    main()
