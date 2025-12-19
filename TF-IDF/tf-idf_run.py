# tf-idf_run.py
import os, json
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# ================= PATH =================
ART_DIR = "/home/detsu/Documents/vscode/NLP/NLP/artifacts"

X = sparse.load_npz(
    os.path.join(ART_DIR, "tfidf_embeddings.npz")
)
y = np.load(os.path.join(ART_DIR, "tfidf_labels.npy"))

OUT_DIR = os.path.join(ART_DIR, "tfidf_cv_top10_40runs")
os.makedirs(OUT_DIR, exist_ok=True)

# =============== SPLIT ==================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =====================================================
# 1️⃣ CROSS VALIDATION → OPTIMAL PARAMETER (LogReg)
# =====================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    "C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
}

grid = GridSearchCV(
    LogisticRegression(
        max_iter=300,
        solver="liblinear"
    ),
    param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=2
)

print("Running cross-validation for optimal C...")
grid.fit(X_train, y_train)

cv_df = pd.DataFrame(grid.cv_results_)
cv_df.to_csv(
    os.path.join(OUT_DIR, "cv_logreg_results.csv"),
    index=False
)

top10_C = (
    cv_df.sort_values("mean_test_score", ascending=False)
         .head(10)["param_C"]
         .astype(float)
         .tolist()
)

json.dump(
    {"top10_C": top10_C},
    open(os.path.join(OUT_DIR, "top10_params.json"), "w"),
    indent=2
)

# ================= METRICS =================
def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

# =====================================================
# 2️⃣ FINAL EXPERIMENTS → 10 × 3 = 30 RESULTS
# =====================================================
rows = []

for run_id, C in enumerate(top10_C, 1):
    print(f"RUN {run_id}/10 | C={C}")

    # -------- Logistic Regression --------
    lr = LogisticRegression(
        C=C,
        max_iter=300,
        solver="liblinear"
    )
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)

    rows.append({
        "run": run_id,
        "model": "LogisticRegression",
        "C": C,
        **metrics(y_test, pred)
    })

    # -------- Random Forest --------
    rf = RandomForestClassifier(
        n_estimators=60,
        n_jobs=2,
        random_state=run_id
    )
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    rows.append({
        "run": run_id,
        "model": "RandomForest",
        "C": C,
        **metrics(y_test, pred)
    })

    # -------- AdaBoost --------
    ada = AdaBoostClassifier(
        n_estimators=60,
        learning_rate=0.7,
        random_state=run_id
    )
    ada.fit(X_train, y_train)
    pred = ada.predict(X_test)

    rows.append({
        "run": run_id,
        "model": "AdaBoost",
        "C": C,
        **metrics(y_test, pred)
    })

# ================= SAVE =================
df = pd.DataFrame(rows)
df.to_csv(
    os.path.join(OUT_DIR, "tfidf_40_results.csv"),
    index=False
)

print("✅ TF-IDF EXPERIMENT DONE")
print("Saved →", os.path.join(OUT_DIR, "tfidf_40_results.csv"))
print("Total rows:", len(df))
