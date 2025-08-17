
import argparse, os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", type=str, default="outputs/features.npy")
    ap.add_argument("--labels", type=str, default="outputs/labels.npy")
    args = ap.parse_args()

    X = np.load(args.feats)
    y = np.load(args.labels)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    print(f"Linear probe accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
