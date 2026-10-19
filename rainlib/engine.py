"""High-level prediction engine used by every API route (web, IoT, CLI)."""
from __future__ import annotations

from typing import Dict, Optional

from .features import FEATURES, RAW_INPUTS, build_vector, vector_to_dict
from .fuzzy import classify_rain_type, fis_from_features
from .model import gbm_proba, gbm_rain_mm, load_model, lr_proba


def _condition_text(p: float) -> str:
    if p < 0.15:
        return "Clear skies likely"
    if p < 0.35:
        return "Mostly dry, slight chance of showers"
    if p < 0.55:
        return "Unsettled, keep an umbrella handy"
    if p < 0.75:
        return "Rain likely"
    return "Rain almost certain"


def predict(raw: Dict, climatology: Optional[float] = None) -> Dict:
    """Run the hybrid ML + fuzzy ensemble on raw inputs.

    ``climatology`` (0-1) is the historical rain frequency for this place/date
    (from the Open-Meteo archive). When provided it acts as a weak Bayesian
    prior that nudges the probability toward local history.
    """
    m = load_model()
    x = build_vector(raw, m["medians"])
    f = vector_to_dict(x)

    p_lr = lr_proba(m, x)
    p_gbm = gbm_proba(m, x)
    fis_value, fis_details = fis_from_features(f, return_details=True)
    p_fis = fis_value / 100.0

    w = m["ensemble"]["weights"]
    p = w["logistic_regression"] * p_lr + w["gbm"] * p_gbm + w["fuzzy"] * p_fis

    prior_used = None
    if climatology is not None and 0 <= climatology <= 1:
        base_rate = m["ensemble"].get("base_rate", 0.22)
        # Shift log-odds by the difference between local and training base rates
        import math
        lo = math.log(max(p, 1e-6) / max(1 - p, 1e-6))
        c = min(max(climatology, 0.01), 0.99)
        lo += 0.5 * (math.log(c / (1 - c)) - math.log(base_rate / (1 - base_rate)))
        p = 1 / (1 + math.exp(-lo))
        prior_used = round(climatology, 3)

    threshold = m["ensemble"]["threshold"]
    mm = gbm_rain_mm(m, x)
    rain_type = classify_rain_type(mm, p, f)
    provided = [k for k in RAW_INPUTS if raw.get(k) not in (None, "")]

    return {
        "probability": round(p, 4),
        "will_rain": p >= threshold,
        "threshold": threshold,
        "condition": _condition_text(p),
        "rain_type": rain_type,
        "expected_mm_if_rain": round(mm, 2),
        "components": {
            "logistic_regression": round(p_lr, 4),
            "gradient_boosting": round(p_gbm, 4),
            "fuzzy_logic": round(p_fis, 4),
            "weights": w,
            "climatology_prior": prior_used,
        },
        "fuzzy": fis_details,
        "features": {k: round(v, 3) for k, v in f.items()},
        "inputs_provided": provided,
        "inputs_imputed": [k for k in RAW_INPUTS if k not in provided],
        "model": {"version": m.get("version"), "trained_on": m.get("trained_on")},
    }


__all__ = ["predict", "FEATURES", "RAW_INPUTS"]
