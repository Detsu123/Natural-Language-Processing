#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graph.py — үр дүнгийн графикууд (accuracy bar + confusion matrix)
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_accuracy(csv_path, out_path):
    df = pd.read_csv(csv_path)
    plt.figure()
    plt.bar(df["model"], df["accuracy"])
    plt.title("Model Accuracy (IMDB TF-IDF)")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_confmat(csv_path, out_path):
    cm = pd.read_csv(csv_path, header=None).values
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i,j], ha="center", va="center", color="black")
    plt.title("Confusion Matrix (Best Model)")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", default="results/metrics.csv")
    ap.add_argument("--confmat_csv", default="results/confmat_best.csv")
    ap.add_argument("--out_dir", default="graphs")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(exist_ok=True)
    plot_accuracy(Path(args.results_csv), out/"accuracy.png")
    plot_confmat(Path(args.confmat_csv), out/"confmat_best.png")
    print("[DONE] graphs saved in", out)

if __name__ == "__main__":
    main()
