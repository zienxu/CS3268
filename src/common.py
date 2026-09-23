"""Shared helpers for the CS3268 BAF project. Everyone imports this; only Yekai edits it.

Usage in Colab:
    from google.colab import drive; drive.mount('/content/drive')
    import sys; sys.path.append('/content/drive/MyDrive/CS3268/src')
    from common import load_split, threshold_at_fpr, save_preds
"""
import numpy as np
import pandas as pd

SEED = 42
ROOT = "/content/drive/MyDrive/CS3268"
DATA = f"{ROOT}/data"
PREDS = f"{ROOT}/preds"

LABEL = "fraud_bool"
TIME = "month"
PROTECTED = "customer_age"
CATEGORICALS = ["payment_type", "employment_status", "housing_status", "source", "device_os"]

SPLITS = {"train": [0, 1, 2, 3, 4], "val": [5], "test": [6, 7]}


def load_split(split: str, variant: str = "base") -> pd.DataFrame:
    """Return the rows for 'train', 'val' or 'test'. Includes id, label, month and age_group."""
    df = pd.read_parquet(f"{DATA}/{variant}.parquet")
    return df[df[TIME].isin(SPLITS[split])].reset_index(drop=True)


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Model inputs: everything except id, label, month and the derived age_group.
    customer_age stays IN (fairness fixes act on weights/thresholds, not by dropping age)."""
    return df.drop(columns=["id", LABEL, TIME, "age_group"])


# n_estimators is FIXED here
# (best iteration from tuning) so every fix trains the same-sized model — no early stopping downstream.
BEST_PARAMS = dict(
    n_estimators=232,
    max_depth=4,
    learning_rate=0.1,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
)


def train_model(X, y, sample_weight=None, params=None):
    """Train the baseline XGBoost config. Chloe's reweighing passes sample_weight."""
    import xgboost as xgb
    p = dict(params or BEST_PARAMS)
    model = xgb.XGBClassifier(tree_method="hist", enable_categorical=True,
                              random_state=SEED, n_jobs=-1, **p)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def threshold_at_fpr(scores, y, fpr: float = 0.05) -> float:
    """Cutoff such that `fpr` of honest (y == 0) rows score at or above it."""
    scores, y = np.asarray(scores), np.asarray(y)
    return float(np.quantile(scores[y == 0], 1 - fpr))


def save_preds(model_name: str, df: pd.DataFrame, scores, cutoff) -> str:
    """Write the agreed predictions file: id, month, y_true, age_group, score, decision.
    `cutoff` is a float, or a dict {age_group_value: cutoff} for group thresholds."""
    out = pd.DataFrame({
        "id": df["id"].values,
        "month": df[TIME].values,
        "y_true": df[LABEL].values,
        "age_group": df["age_group"].values,
        "score": np.asarray(scores),
    })
    if isinstance(cutoff, dict):
        out["decision"] = (out["score"] >= out["age_group"].map(cutoff)).astype(int)
    else:
        out["decision"] = (out["score"] >= cutoff).astype(int)
    path = f"{PREDS}/preds_{model_name}.parquet"
    out.to_parquet(path, index=False)
    return path
