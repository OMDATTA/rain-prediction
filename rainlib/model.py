"""Dependency-free inference for models trained in Colab.

The notebook exports scikit-learn models to ``rainlib/model.json``:
  * Logistic Regression  -> scaler mean/scale + coefficients
  * Gradient Boosting classifier (RainTomorrow) -> list of decision trees
  * Gradient Boosting regressor  (tomorrow's rainfall, log1p mm) -> trees
  * GA-optimised ensemble weights + threshold
Evaluating these in plain Python keeps the Vercel function tiny and fast.
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from typing import Dict, List

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.json")


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _tree_value(tree: Dict, x: List[float]) -> float:
    node = 0
    left, right, feat, thr, val = tree["l"], tree["r"], tree["f"], tree["t"], tree["v"]
    while left[node] != -1:
        node = left[node] if x[feat[node]] <= thr[node] else right[node]
    return val[node]


def _gbm_raw(gbm: Dict, x: List[float]) -> float:
    return gbm["init"] + gbm["learning_rate"] * sum(_tree_value(t, x) for t in gbm["trees"])


@lru_cache(maxsize=1)
def load_model(path: str = MODEL_PATH) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def lr_proba(m: Dict, x: List[float]) -> float:
    lr = m["logistic_regression"]
    z = lr["intercept"]
    for xi, mu, sd, w in zip(x, lr["mean"], lr["scale"], lr["coef"]):
        z += w * ((xi - mu) / sd)
    return _sigmoid(z)


def gbm_proba(m: Dict, x: List[float]) -> float:
    return _sigmoid(_gbm_raw(m["gbm_classifier"], x))


def gbm_rain_mm(m: Dict, x: List[float]) -> float:
    """Expected rainfall tomorrow, conditional on rain (mm)."""
    return max(0.0, math.expm1(_gbm_raw(m["gbm_regressor"], x)))
