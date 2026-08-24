"""Train and evaluate the total_amount reliability model (logistic regression).

Trains on ml/data/total_amount_train.csv and evaluates on the dataset's own
held-out test split (total_amount_test.csv). The data is imbalanced (~87% wrong),
so we use balanced class weights and judge with ROC-AUC, balanced accuracy, and
per-class precision/recall — NOT raw accuracy (predicting 'always wrong' would
score 87% and be useless). Prints the learned weights and saves the model.

Usage:  python ml/train_model.py
"""

import csv
import json
import sys
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_DIR = Path(__file__).resolve().parent / "models"


def load(path):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    features = [c for c in rows[0].keys() if c != "label"]
    X = [[float(r[c]) for c in features] for r in rows]
    y = [int(r["label"]) for r in rows]
    return features, X, y


def main():
    features, X_tr, y_tr = load(DATA_DIR / "total_amount_train.csv")
    print(f"train: {len(X_tr)} rows, {len(features)} features")
    print(f"features: {features}\n")

    # Balanced class weights counter the 87/13 imbalance so the minority
    # ('correct') class isn't ignored.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )

    # Honest cross-validated estimates on the training data.
    for metric in ("roc_auc", "balanced_accuracy"):
        s = cross_val_score(model, X_tr, y_tr, cv=5, scoring=metric)
        print(f"5-fold CV {metric:18s}: {s.mean():.3f}  (folds {[round(v,3) for v in s]})")

    model.fit(X_tr, y_tr)

    # True held-out evaluation on the dataset's own test split, if present.
    test_csv = DATA_DIR / "total_amount_test.csv"
    if test_csv.exists():
        _, X_te, y_te = load(test_csv)
        proba = [p[1] for p in model.predict_proba(X_te)]
        pred = model.predict(X_te)
        print(f"\nofficial test split ({len(X_te)} rows):")
        print(f"  ROC-AUC          : {roc_auc_score(y_te, proba):.3f}")
        print(f"  balanced accuracy: {balanced_accuracy_score(y_te, pred):.3f}")
        print("  confusion matrix [rows=actual correct/wrong, cols=pred correct/wrong]:")
        print("   ", confusion_matrix(y_te, pred).tolist())
        print(classification_report(y_te, pred, target_names=["correct(0)", "wrong(1)"], zero_division=0))

    # Learned weights (from the model fit on train).
    lr = model.named_steps["logisticregression"]
    weights = sorted(zip(features, lr.coef_[0]), key=lambda kv: -abs(kv[1]))
    print("learned weights (+ = pushes toward 'wrong', standardized):")
    for name, w in weights:
        print(f"  {name:14s} {w:+.3f}  {'→ wrong' if w > 0 else '→ correct'}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / "reliability_total.joblib"
    joblib.dump({"model": model, "features": features}, out)
    print(f"\nsaved -> {out}")

    # Publish the measured numbers so the demo UI can never show stale metrics.
    metrics = {
        "train_rows": len(X_tr),
        "train_wrong": sum(y_tr),
        "cv_roc_auc": round(float(cross_val_score(model, X_tr, y_tr, cv=5, scoring="roc_auc").mean()), 3),
    }
    if test_csv.exists():
        metrics.update({
            "test_rows": len(X_te),
            "test_wrong": sum(y_te),
            "roc_auc": round(float(roc_auc_score(y_te, proba)), 3),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_te, pred)), 3),
        })

    # Honesty check: how much of the score does the single strongest feature
    # already explain on its own? If this matches the model's own AUC, the model
    # is not adding anything and we should say so rather than claim credit.
    best_name, best_auc = None, 0.0
    for i, name in enumerate(features):
        column = [row[i] for row in X_tr]
        if len(set(column)) < 2:
            continue
        auc = roc_auc_score(y_tr, column)
        auc = max(auc, 1 - auc)  # a perfectly inverted feature is just as telling
        if auc > best_auc:
            best_name, best_auc = name, auc
    metrics["best_single_feature"] = best_name
    metrics["best_single_feature_auc"] = round(float(best_auc), 3)
    print(f"\nstrongest single feature alone: {best_name} -> AUC {best_auc:.3f}")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved -> {MODEL_DIR / 'metrics.json'}  {metrics}")


if __name__ == "__main__":
    main()
