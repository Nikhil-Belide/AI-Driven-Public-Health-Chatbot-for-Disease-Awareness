"""
Train the symptom -> disease model.

Usage:
    python -m ml.train                       # synthetic data from KB profiles
    python -m ml.train --csv Training.csv    # real data (Kaggle format)

CSV format: one column per symptom (0/1) and a `prognosis` column with the
disease name. Column names are lower-cased and spaces become underscores.
For a real CSV, symptoms in the model vocabulary come from the CSV itself, so
update the LLM extraction vocabulary automatically (it reads model.symptoms).

Outputs ml/model.joblib containing the calibrated classifier, the symptom
list and a disease x symptom frequency matrix (used to pick follow-up
questions).
"""

import argparse
import os

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from ml.symptoms import DISEASE_PROFILES, SYMPTOMS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")


def make_synthetic(n_per_disease: int = 600, seed: int = 42):
    """
    Sample patients from the disease profiles. To avoid a model that is
    unrealistically sharp, we also (a) drop each present symptom with 15%
    probability (patients forget/omit things) and (b) add random unrelated
    symptoms with 3% probability.
    """
    rng = np.random.default_rng(seed)
    idx = {s: i for i, s in enumerate(SYMPTOMS)}
    X, y = [], []
    for disease, profile in DISEASE_PROFILES.items():
        for _ in range(n_per_disease):
            row = np.zeros(len(SYMPTOMS), dtype=np.int8)
            for s, p in profile.items():
                if rng.random() < p and rng.random() > 0.15:
                    row[idx[s]] = 1
            row[rng.random(len(SYMPTOMS)) < 0.03] = 1
            # Users often report only some of their symptoms. Half the rows keep
            # a random 30-100% of them, so a short list is NOT read as "all the
            # other symptoms are absent" (the main source of overconfidence).
            if rng.random() < 0.5:
                keep = rng.uniform(0.3, 1.0)
                present = np.flatnonzero(row)
                drop = present[rng.random(len(present)) > keep]
                row[drop] = 0
            if row.sum() == 0:  # nobody walks in with zero symptoms
                s = rng.choice(list(profile))
                row[idx[s]] = 1
            X.append(row)
            y.append(disease)
    return np.array(X), np.array(y), list(SYMPTOMS)


def load_csv(path: str):
    import pandas as pd

    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("unnamed")]
    y = df.pop("prognosis").astype(str).str.strip().values
    return df.values.astype(np.int8), y, list(df.columns)


def train(X, y, symptoms, save_path: str = MODEL_PATH, verbose: bool = True):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    base = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                  random_state=42, n_jobs=-1)
    clf = CalibratedClassifierCV(base, cv=3, method="isotonic")
    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)
    classes = clf.classes_
    top1 = accuracy_score(y_te, classes[proba.argmax(1)])
    top3_idx = np.argsort(proba, axis=1)[:, -3:]
    top3 = float(np.mean([y_te[i] in classes[top3_idx[i]] for i in range(len(y_te))]))

    if verbose:
        print(f"Diseases: {len(classes)} | Symptoms: {len(symptoms)} | "
              f"Train rows: {len(y_tr)} | Test rows: {len(y_te)}")
        print(f"Top-1 accuracy: {top1:.3f}")
        print(f"Top-3 accuracy: {top3:.3f}")
        print(classification_report(y_te, classes[proba.argmax(1)], zero_division=0))
        cm = confusion_matrix(y_te, classes[proba.argmax(1)], labels=classes)
        np.fill_diagonal(cm, 0)
        pairs = sorted(
            ((cm[i, j], classes[i], classes[j])
             for i in range(len(classes)) for j in range(len(classes)) if cm[i, j]),
            reverse=True,
        )[:5]
        print("Most confused pairs (true -> predicted, count):")
        for n, a, b in pairs:
            print(f"  {a} -> {b}: {n}")
        print("NOTE: scores on synthetic data only show the pipeline works; "
              "they are NOT real-world accuracy.")

    # Refit on all data for the final model
    final = CalibratedClassifierCV(base, cv=3, method="isotonic").fit(X, y)
    freq = np.array([X[y == c].mean(axis=0) for c in final.classes_])
    joblib.dump({"model": final, "symptoms": symptoms,
                 "classes": list(final.classes_), "freq": freq}, save_path)
    if verbose:
        print(f"Saved model to {save_path}")
    return top1, top3


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="Path to a real symptom dataset (Kaggle format)")
    args = ap.parse_args()
    if args.csv:
        X, y, syms = load_csv(args.csv)
    else:
        X, y, syms = make_synthetic()
    train(X, y, syms)
