#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualization.py — IMDB Sentiment Analysis (Advanced Visualization)
---------------------------------------------------------
Бэлтгэсэн үр дүнгийн (metrics.csv, TF-IDF vectorizer, dataset) 
үндсэн дээр нэмэлт график, статистик шинжилгээ гаргана:
 - WordCloud (positive / negative)
 - Feature importance (Logistic Regression weights)
 - Confusion matrix heatmap (seaborn)
 - Precision–Recall ба ROC curve
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from wordcloud import WordCloud
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from scipy import sparse

def plot_wordclouds(df_csv, out_dir):
    """Эерэг ба сөрөг текстүүдийн WordCloud"""
    df = pd.read_csv(df_csv)
    pos_text = " ".join(df[df["label"] == 1]["text"].tolist())
    neg_text = " ".join(df[df["label"] == 0]["text"].tolist())

    wc = WordCloud(width=1000, height=500, background_color="white", max_words=200)
    plt.figure(figsize=(10,5))
    plt.imshow(wc.generate(pos_text))
    plt.axis("off")
    plt.title("Positive Reviews WordCloud")
    plt.savefig(out_dir/"wordcloud_positive.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10,5))
    plt.imshow(wc.generate(neg_text))
    plt.axis("off")
    plt.title("Negative Reviews WordCloud")
    plt.savefig(out_dir/"wordcloud_negative.png", dpi=150)
    plt.close()

def plot_feature_importance(vectorizer_pkl, X_npz, y_npy, out_dir):
    """TF-IDF онцлогуудаас хамгийн нөлөө бүхий үгсийг харуулах"""
    with open(vectorizer_pkl, "rb") as f:
        vect = pickle.load(f)
    X = sparse.load_npz(X_npz)
    y = np.load(y_npy)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    feature_names = np.array(vect.get_feature_names_out())
    coefs = model.coef_.ravel()

    top_pos = np.argsort(coefs)[-20:]
    top_neg = np.argsort(coefs)[:20]

    plt.figure(figsize=(8,6))
    plt.barh(feature_names[top_pos], coefs[top_pos], color="green")
    plt.title("Top Positive Words (LogReg Coefficients)")
    plt.tight_layout()
    plt.savefig(out_dir/"top_positive_words.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8,6))
    plt.barh(feature_names[top_neg], coefs[top_neg], color="red")
    plt.title("Top Negative Words (LogReg Coefficients)")
    plt.tight_layout()
    plt.savefig(out_dir/"top_negative_words.png", dpi=150)
    plt.close()

def plot_confmat_heatmap(confmat_csv, out_png):
    cm = pd.read_csv(confmat_csv, header=None).values
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix (Heatmap)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_precision_recall_roc(X_npz, y_npy, out_dir):
    X = sparse.load_npz(X_npz)
    y = np.load(y_npy)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    scores = clf.decision_function(X)
    precision, recall, _ = precision_recall_curve(y, scores)
    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(recall, precision)
    plt.title("Precision–Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(out_dir/"precision_recall_curve.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],"--",color="gray")
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Logistic Regression)")
    plt.tight_layout()
    plt.savefig(out_dir/"roc_curve.png", dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_csv", default="data/texts_clean.csv")
    ap.add_argument("--vectorizer", default="artifacts/vectorizer.pkl")
    ap.add_argument("--features", default="artifacts/features.npz")
    ap.add_argument("--labels", default="artifacts/labels.npy")
    ap.add_argument("--confmat_csv", default="results/confmat_best.csv")
    ap.add_argument("--out_dir", default="graphs")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    plot_wordclouds(args.clean_csv, out)
    plot_feature_importance(args.vectorizer, args.features, args.labels, out)
    plot_confmat_heatmap(args.confmat_csv, out/"confmat_heatmap.png")
    plot_precision_recall_roc(args.features, args.labels, out)
    print("[DONE] Advanced visualizations saved to", out)

if __name__ == "__main__":
    main()
