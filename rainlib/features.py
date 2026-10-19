"""Feature engineering shared by training (Colab / src) and inference (Vercel API).

Every model in this project consumes the same ordered feature vector produced
by :func:`build_vector`, so training and serving can never drift apart.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

# Raw inputs a user, an IoT node or the live-weather service can provide.
# Units follow the Kaggle "Rain in Australia" (weatherAUS) dataset.
RAW_INPUTS: Dict[str, Dict] = {
    "MinTemp":       {"unit": "°C",   "min": -10, "max": 50,   "label": "Minimum temperature"},
    "MaxTemp":       {"unit": "°C",   "min": -5,  "max": 55,   "label": "Maximum temperature"},
    "Rainfall":      {"unit": "mm",   "min": 0,   "max": 400,  "label": "Rainfall in last 24 h"},
    "WindGustSpeed": {"unit": "km/h", "min": 0,   "max": 150,  "label": "Strongest wind gust"},
    "WindGustDir":   {"unit": "°",    "min": 0,   "max": 360,  "label": "Gust direction"},
    "WindSpeed9am":  {"unit": "km/h", "min": 0,   "max": 130,  "label": "Wind speed 9 am"},
    "WindSpeed3pm":  {"unit": "km/h", "min": 0,   "max": 130,  "label": "Wind speed 3 pm"},
    "Humidity9am":   {"unit": "%",    "min": 0,   "max": 100,  "label": "Humidity 9 am"},
    "Humidity3pm":   {"unit": "%",    "min": 0,   "max": 100,  "label": "Humidity 3 pm"},
    "Pressure9am":   {"unit": "hPa",  "min": 970, "max": 1045, "label": "Pressure 9 am"},
    "Pressure3pm":   {"unit": "hPa",  "min": 970, "max": 1045, "label": "Pressure 3 pm"},
    "Cloud9am":      {"unit": "okta", "min": 0,   "max": 8,    "label": "Cloud cover 9 am"},
    "Cloud3pm":      {"unit": "okta", "min": 0,   "max": 8,    "label": "Cloud cover 3 pm"},
    "Temp9am":       {"unit": "°C",   "min": -10, "max": 50,   "label": "Temperature 9 am"},
    "Temp3pm":       {"unit": "°C",   "min": -10, "max": 55,   "label": "Temperature 3 pm"},
}

COMPASS = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5, "SE": 135,
    "SSE": 157.5, "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5, "W": 270,
    "WNW": 292.5, "NW": 315, "NNW": 337.5,
}

FEATURES: List[str] = [
    "MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm",
    "Temp9am", "Temp3pm", "RainToday",
    # engineered
    "HumidityDelta", "PressureDelta", "TempRange", "GustDirSin", "GustDirCos",
]


def _num(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def direction_to_deg(v) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, str) and v.strip().upper() in COMPASS:
        return COMPASS[v.strip().upper()]
    return _num(v)


def clamp_inputs(raw: Dict) -> Dict[str, Optional[float]]:
    """Coerce and clamp raw user / sensor input to physically valid ranges."""
    out: Dict[str, Optional[float]] = {}
    for key, meta in RAW_INPUTS.items():
        val = direction_to_deg(raw.get(key)) if key == "WindGustDir" else _num(raw.get(key))
        if val is not None:
            val = max(meta["min"], min(meta["max"], val))
        out[key] = val
    return out


def build_vector(raw: Dict, medians: Optional[Dict[str, float]] = None) -> List[float]:
    """Turn a dict of raw inputs into the ordered numeric feature vector.

    Missing values are filled with training medians so partial inputs (e.g. an
    IoT node with only temperature / humidity / pressure) still work.
    """
    medians = medians or {}
    r = clamp_inputs(raw)

    def g(k):
        v = r.get(k)
        return v if v is not None else medians.get(k, 0.0)

    rainfall = g("Rainfall")
    rain_today = raw.get("RainToday")
    if isinstance(rain_today, str):
        rain_today = 1.0 if rain_today.strip().lower() in ("yes", "1", "true") else 0.0
    elif rain_today is None:
        rain_today = 1.0 if rainfall >= 1.0 else 0.0
    else:
        rain_today = 1.0 if float(rain_today) else 0.0

    d = r.get("WindGustDir")
    if d is None:
        sin_d, cos_d = medians.get("GustDirSin", 0.0), medians.get("GustDirCos", 0.0)
    else:
        sin_d, cos_d = math.sin(math.radians(d)), math.cos(math.radians(d))

    base = {k: g(k) for k in RAW_INPUTS if k != "WindGustDir"}
    base.update(
        RainToday=rain_today,
        HumidityDelta=base["Humidity3pm"] - base["Humidity9am"],
        PressureDelta=base["Pressure3pm"] - base["Pressure9am"],
        TempRange=base["MaxTemp"] - base["MinTemp"],
        GustDirSin=sin_d,
        GustDirCos=cos_d,
    )
    return [float(base[f]) for f in FEATURES]


def vector_to_dict(vec: Iterable[float]) -> Dict[str, float]:
    return dict(zip(FEATURES, vec))
