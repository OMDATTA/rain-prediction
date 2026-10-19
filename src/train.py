"""Train, evaluate and export the RainCast models.

Usage (locally or in Colab):
    python -m src.train --data data/weatherAUS.csv

Outputs:
    rainlib/model.json      – exported models used by the Vercel API
    reports/metrics.json    – evaluation of every model on the held-out test set
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, roc_auc_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rainlib.features import COMPASS, FEATURES  # noqa: E402
from rainlib.fuzzy import fis_from_features  # noqa: E402
from src.genetic import optimise  # noqa: E402


# ------------------------------------------------------------------ data
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.dropna(subset=["RainTomorrow"]).copy()
    df["RainTomorrow"] = (df["RainTomorrow"] == "Yes").astype(int)
    df["RainToday"] = df["RainToday"].map({"Yes": 1.0, "No": 0.0})
    df.loc[df["RainToday"].isna(), "RainToday"] = (df["Rainfall"] >= 1).astype(float)
    df["Cloud9am"] = df["Cloud9am"].clip(upper=8)
    df["Cloud3pm"] = df["Cloud3pm"].clip(upper=8)
    deg = df["WindGustDir"].map(COMPASS)
    df["GustDirSin"] = np.sin(np.radians(deg))
    df["GustDirCos"] = np.cos(np.radians(deg))
    df["HumidityDelta"] = df["Humidity3pm"] - df["Humidity9am"]
    df["PressureDelta"] = df["Pressure3pm"] - df["Pressure9am"]
    df["TempRange"] = df["MaxTemp"] - df["MinTemp"]
    if "RISK_MM" not in df.columns:  # newer Kaggle versions drop RISK_MM; rebuild from next-day Rainfall
        df = df.sort_values(["Location", "Date"])
        df["RISK_MM"] = df.groupby("Location")["Rainfall"].shift(-1)
    return df


def split_by_time(df: pd.DataFrame):
    """Chronological split – honest evaluation of a forecasting model."""
    train = df[df["Date"] < "2015-01-01"]
    val = df[(df["Date"] >= "2015-01-01") & (df["Date"] < "2016-01-01")]
    test = df[df["Date"] >= "2016-01-01"]
    return train, val, test


def impute(df: pd.DataFrame, medians: dict) -> np.ndarray:
    return df[FEATURES].fillna(medians).to_numpy(dtype=float)


def fuzzy_scores(X: np.ndarray) -> np.ndarray:
    return np.array([fis_from_features(dict(zip(FEATURES, row))) / 100.0 for row in X])


# ------------------------------------------------------------------ export
def export_trees(model, n_outputs_idx=0):
    trees = []
    for est in model.estimators_[:, n_outputs_idx]:
        t = est.tree_
        trees.append({
            "l": t.children_left.tolist(), "r": t.children_right.tolist(),
            "f": [int(v) for v in t.feature], "t": [round(float(v), 6) for v in t.threshold],
            "v": [round(float(v), 6) for v in t.value[:, 0, 0]],
        })
    return trees


def gbm_init_raw(model, X):
    """Raw (log-odds or mean) contribution of the init estimator."""
    raw = model._raw_predict_init(X[:1])
    return float(np.ravel(raw)[0])


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {"accuracy": round(accuracy_score(y, pred), 4), "precision": round(precision_score(y, pred), 4),
            "recall": round(recall_score(y, pred), 4), "f1": round(f1_score(y, pred), 4),
            "roc_auc": round(roc_auc_score(y, p), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "weatherAUS.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "rainlib", "model.json"))
    ap.add_argument("--fast", action="store_true", help="subsample for quick experiments")
    args = ap.parse_args()

    t0 = time.time()
    df = load_dataset(args.data)
    train, val, test = split_by_time(df)
    if args.fast:
        train = train.sample(20000, random_state=0)
    print(f"rows: train={len(train)} val={len(val)} test={len(test)}  rain rate={df.RainTomorrow.mean():.3f}")

    medians = train[FEATURES].median().round(4).to_dict()
    Xtr, Xva, Xte = impute(train, medians), impute(val, medians), impute(test, medians)
    ytr, yva, yte = train.RainTomorrow.values, val.RainTomorrow.values, test.RainTomorrow.values

    # ---------------- ML models
    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000, C=0.5).fit(scaler.transform(Xtr), ytr)
    gbm = GradientBoostingClassifier(n_estimators=250, max_depth=4, learning_rate=0.08,
                                     subsample=0.8, random_state=42).fit(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=200, max_depth=14, n_jobs=-1, random_state=42).fit(Xtr, ytr)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=60, early_stopping=True,
                        random_state=42).fit(scaler.transform(Xtr), ytr)
    print(f"trained models in {time.time() - t0:.0f}s")

    # Rainfall amount regressor, conditional on rain tomorrow
    rainy = train[(train.RainTomorrow == 1) & train.RISK_MM.notna()]
    reg = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.08,
                                    subsample=0.8, random_state=42)
    reg.fit(impute(rainy, medians), np.log1p(rainy.RISK_MM.values))

    # ---------------- soft computing: fuzzy + GA ensemble
    P_val = np.column_stack([lr.predict_proba(scaler.transform(Xva))[:, 1],
                             gbm.predict_proba(Xva)[:, 1], fuzzy_scores(Xva)])
    ens, ga_history = optimise(P_val, yva)
    print("GA ensemble:", ens)

    P_te = np.column_stack([lr.predict_proba(scaler.transform(Xte))[:, 1],
                            gbm.predict_proba(Xte)[:, 1], fuzzy_scores(Xte)])
    w = np.array([ens["weights"]["logistic_regression"], ens["weights"]["gbm"], ens["weights"]["fuzzy"]])
    p_ens = P_te @ w

    rainy_te = test[(test.RainTomorrow == 1) & test.RISK_MM.notna()]
    mm_pred = np.expm1(reg.predict(impute(rainy_te, medians)))
    report = {
        "test_period": f"{test.Date.min().date()} to {test.Date.max().date()}",
        "n_test": int(len(test)),
        "models": {
            "logistic_regression": metrics(yte, P_te[:, 0]),
            "random_forest": metrics(yte, rf.predict_proba(Xte)[:, 1]),
            "gradient_boosting": metrics(yte, P_te[:, 1]),
            "neural_network_mlp": metrics(yte, mlp.predict_proba(scaler.transform(Xte))[:, 1]),
            "fuzzy_inference": metrics(yte, P_te[:, 2]),
            "hybrid_ga_ensemble": metrics(yte, p_ens, ens["threshold"]),
        },
        "rain_amount_mae_mm": round(float(mean_absolute_error(rainy_te.RISK_MM.values, mm_pred)), 3),
        "ga_fitness_history": [round(h, 4) for h in ga_history],
        "feature_importance_gbm": dict(sorted(
            {f: round(float(v), 4) for f, v in zip(FEATURES, gbm.feature_importances_)}.items(),
            key=lambda kv: -kv[1])),
    }
    for k, v in report["models"].items():
        print(f"{k:22s} {v}")

    model_json = {
        "version": "1.0.0",
        "trained_on": f"Kaggle weatherAUS ({len(train)} train rows), exported {dt.date.today()}",
        "features": FEATURES,
        "medians": medians,
        "logistic_regression": {
            "mean": scaler.mean_.round(6).tolist(), "scale": scaler.scale_.round(6).tolist(),
            "coef": lr.coef_[0].round(6).tolist(), "intercept": round(float(lr.intercept_[0]), 6)},
        "gbm_classifier": {"init": gbm_init_raw(gbm, Xtr), "learning_rate": gbm.learning_rate,
                           "trees": export_trees(gbm)},
        "gbm_regressor": {"init": gbm_init_raw(reg, Xtr), "learning_rate": reg.learning_rate,
                          "trees": export_trees(reg)},
        "ensemble": {**ens, "base_rate": round(float(ytr.mean()), 4)},
        "metrics": {"test_period": report["test_period"], "n_test": report["n_test"],
                    "models": report["models"], "rain_amount_mae_mm": report["rain_amount_mae_mm"]},
    }
    with open(args.out, "w") as fh:
        json.dump(model_json, fh, separators=(",", ":"))
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "metrics.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    # parity check: pure-python inference must match scikit-learn
    from rainlib import model as pm
    pm.load_model.cache_clear()
    m = pm.load_model(args.out)
    idx = np.random.default_rng(0).choice(len(Xte), 200, replace=False)
    d1 = max(abs(pm.gbm_proba(m, list(Xte[i])) - P_te[i, 1]) for i in idx)
    d2 = max(abs(pm.lr_proba(m, list(Xte[i])) - P_te[i, 0]) for i in idx)
    print(f"parity max |diff|: gbm={d1:.2e} lr={d2:.2e}")
    assert d1 < 1e-3 and d2 < 1e-3, "exported model does not match scikit-learn"
    print(f"saved {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
