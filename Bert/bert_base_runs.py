# bert_base_cv_top10_40runs_light.py
import os, json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ================= PATH =================
ART_DIR = "/home/detsu/Documents/vscode/NLP/NLP/artifacts"
X = np.load(os.path.join(ART_DIR, "bert_base_cls.npy")).astype(np.float32)
y = np.load(os.path.join(ART_DIR, "bert_labels.npy"))

OUT_DIR = os.path.join(ART_DIR, "bert_base_cv_top10_40runs")
os.makedirs(OUT_DIR, exist_ok=True)

# =============== SPLIT ==================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =====================================================
# 1️⃣ CROSS VALIDATION (Logistic Regression)
# =====================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr_grid = {"C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]}

grid = GridSearchCV(
    LogisticRegression(max_iter=300, solver="liblinear"),
    lr_grid,
    scoring="f1",
    cv=cv,
    n_jobs=2            # ⬅️ CPU-г хязгаарлав
)
grid.fit(X_train, y_train)

cv_df = pd.DataFrame(grid.cv_results_)
cv_df.to_csv(os.path.join(OUT_DIR, "cv_logreg_results.csv"), index=False)

top10_C = (
    cv_df.sort_values("mean_test_score", ascending=False)
         .head(10)["param_C"]
         .astype(float)
         .tolist()
)

json.dump({"top10_C": top10_C},
          open(os.path.join(OUT_DIR, "top10_params.json"), "w"),
          indent=2)

# ================= LSTM =================
class TinyLSTM(nn.Module):
    def __init__(self, dim, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

# ============ PREPARE LSTM DATA (ONCE) ============
train_ds = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(y_train.astype(np.float32))
)
test_ds = TensorDataset(
    torch.from_numpy(X_test),
    torch.from_numpy(y_test.astype(np.float32))
)

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=64)

# ============ FINAL EXPERIMENTS ===================
rows = []

for run_id, C in enumerate(top10_C, 1):
    print(f"RUN {run_id}/10  |  C={C}")

    # ---- Logistic Regression ----
    lr = LogisticRegression(C=C, max_iter=300, solver="liblinear")
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    rows.append({"run": run_id, "model": "LogReg", "C": C, **metrics(y_test, pred)})

    # ---- Random Forest (lighter) ----
    rf = RandomForestClassifier(
        n_estimators=60,          # ⬅️ багасгасан
        max_depth=None,
        n_jobs=2,
        random_state=run_id
    )
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    rows.append({"run": run_id, "model": "RandomForest", "C": C, **metrics(y_test, pred)})

    # ---- AdaBoost (lighter) ----
    ada = AdaBoostClassifier(
        n_estimators=60,          # ⬅️ багасгасан
        learning_rate=0.7,
        random_state=run_id
    )
    ada.fit(X_train, y_train)
    pred = ada.predict(X_test)
    rows.append({"run": run_id, "model": "AdaBoost", "C": C, **metrics(y_test, pred)})

    # ---- LSTM (SAFE, MINI-BATCH) ----
    model = TinyLSTM(X_train.shape[1], hidden=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(1):            # ⬅️ epochs=1
        for xb, yb in train_dl:
            xb = xb.unsqueeze(1).to(device)
            yb = yb.to(device)

            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in test_dl:
            xb = xb.unsqueeze(1).to(device)
            logits = model(xb)
            preds.append(torch.sigmoid(logits).cpu().numpy())

    pred = (np.concatenate(preds) > 0.5).astype(int)
    rows.append({"run": run_id, "model": "LSTM", "C": C, **metrics(y_test, pred)})

# ============== SAVE =====================
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT_DIR, "bert_base_40_results.csv"), index=False)

print("✅ ALL DONE")
print("Saved →", os.path.join(OUT_DIR, "bert_base_40_results.csv"))
