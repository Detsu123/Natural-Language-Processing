
import os
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/bert_training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

X = np.load("/home/detsu/Documents/vscode/NLP/NLP/artifacts/bert_base_cls.npy")
y = np.load("/home/detsu/Documents/vscode/NLP/NLP/artifacts/bert_labels.npy")

logging.info(f"Loaded BERT embeddings: {X.shape}")
print("Loaded:", X.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "logistic": LogisticRegression(max_iter=500),
    "randomforest": RandomForestClassifier(n_estimators=200),
    "adaboost": AdaBoostClassifier(n_estimators=200)
}

results = []

os.makedirs("results", exist_ok=True)


for name, model in models.items():
    print(f"\nTraining {name} ...")
    logging.info(f"Training {name}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prc = precision_score(y_test, y_pred)
    rcl = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
  # ats gej report gej func bdgiin b 

    cm = confusion_matrix(y_test, y_pred)
    cm_path = f"results/confmat_bert_{name}.csv"
    pd.DataFrame(cm).to_csv(cm_path, index=False)

    logging.info(
        f"{name}: acc={acc:.4f}, precision={prc:.4f}, recall={rcl:.4f}, f1={f1:.4f}"
    )

    results.append([name, acc, prc, rcl, f1])

    print(f"{name} → accuracy={acc:.4f}, f1={f1:.4f}")


df_metrics = pd.DataFrame(results, columns=["model","accuracy","precision","recall","f1"])
df_metrics.to_csv("results/bert_metrics.csv", index=False)

print("\nSaved metrics → results/bert_metrics.csv")
logging.info("Training finished.")
