"""
03_train_classifier.py

Trains a specificity classifier (exploratory=0 / mixed=1 / operational=2) on your
manually labeled training sample, and validates it against the independent holdout
sample.

Approach: sentence embeddings (all-MiniLM-L6-v2) -> logistic regression.
Embeddings generalize better than raw TF-IDF at this sample size (~200 labeled
examples), since they capture semantic similarity beyond exact wording -- important
because "deployed X, reducing handling time by 20%" and "rolled out Y, cutting
processing time by a fifth" should score similarly despite sharing few words.

A TF-IDF + logistic regression baseline is also fit and reported for comparison --
worth including in your methods section as a robustness check even if the
embeddings model is your primary specification.

Requires: pip install sentence-transformers scikit-learn joblib

Usage:
    python 03_train_classifier.py
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sentence_transformers import SentenceTransformer

TRAIN_PATH = "../../Data/specificity_annotation_train.csv"
HOLDOUT_PATH = "../../Data/specificity_annotation_holdout.csv"
MODEL_OUT_PATH = "../../Data/specificity_classifier.joblib"
TEXT_COL = "text"
LABEL_COL = "label"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
RANDOM_SEED = 42


def load_labeled(path):
    df = pd.read_csv(path)
    df = df[df[LABEL_COL].notna() & (df[LABEL_COL] != "")].copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df


def main():
    train_df = load_labeled(TRAIN_PATH)
    holdout_df = load_labeled(HOLDOUT_PATH)

    print(f"Train: {len(train_df)} labeled snippets  |  label distribution:")
    print(train_df[LABEL_COL].value_counts().sort_index())
    print(f"\nHoldout: {len(holdout_df)} labeled snippets  |  label distribution:")
    print(holdout_df[LABEL_COL].value_counts().sort_index())

    print(f"\nLoading embedding model ({EMBED_MODEL_NAME})...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    X_train_emb = embedder.encode(train_df[TEXT_COL].tolist(), show_progress_bar=True)
    y_train = train_df[LABEL_COL].values

    X_holdout_emb = embedder.encode(holdout_df[TEXT_COL].tolist(), show_progress_bar=True)
    y_holdout = holdout_df[LABEL_COL].values

    print("\n=== 5-fold cross-validation on training set (embeddings model) ===")
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_preds = cross_val_predict(clf, X_train_emb, y_train, cv=skf)
    print(classification_report(y_train, cv_preds, digits=3))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y_train, cv_preds))
    print(f"Quadratic weighted kappa (CV): {cohen_kappa_score(y_train, cv_preds, weights='quadratic'):.3f}")

    clf.fit(X_train_emb, y_train)

    print("\n=== Holdout set validation (independent, never seen during training) ===")
    holdout_preds = clf.predict(X_holdout_emb)
    print(classification_report(y_holdout, holdout_preds, digits=3))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y_holdout, holdout_preds))
    kappa = cohen_kappa_score(y_holdout, holdout_preds, weights="quadratic")
    corr = np.corrcoef(y_holdout, holdout_preds)[0, 1]
    print(f"Quadratic weighted kappa (holdout): {kappa:.3f}")
    print(f"Pearson correlation (true vs predicted, holdout): {corr:.3f}")

    holdout_df = holdout_df.copy()
    holdout_df["predicted"] = holdout_preds
    holdout_df["abs_error"] = (holdout_df[LABEL_COL] - holdout_df["predicted"]).abs()
    if "period" in holdout_df.columns:
        print("\nMean absolute error by period:")
        print(holdout_df.groupby("period")["abs_error"].mean())
    if "gics_sector" in holdout_df.columns:
        print("\nMean absolute error by sector:")
        print(holdout_df.groupby("gics_sector")["abs_error"].mean())

    print("\n=== TF-IDF + logistic regression baseline (comparison only) ===")
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = tfidf.fit_transform(train_df[TEXT_COL])
    X_holdout_tfidf = tfidf.transform(holdout_df[TEXT_COL])
    tfidf_clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
    tfidf_clf.fit(X_train_tfidf, y_train)
    tfidf_preds = tfidf_clf.predict(X_holdout_tfidf)
    print(classification_report(y_holdout, tfidf_preds, digits=3))
    tfidf_kappa = cohen_kappa_score(y_holdout, tfidf_preds, weights="quadratic")
    print(f"TF-IDF quadratic weighted kappa (holdout): {tfidf_kappa:.3f}")

    joblib.dump({
        "classifier": clf,
        "embed_model_name": EMBED_MODEL_NAME,
    }, MODEL_OUT_PATH)
    print(f"\nSaved trained model to {MODEL_OUT_PATH}")
    print("\nReport this in your methods section: holdout quadratic weighted kappa "
          f"= {kappa:.3f} (embeddings model) vs {tfidf_kappa:.3f} (TF-IDF baseline).")


if __name__ == "__main__":
    main()
