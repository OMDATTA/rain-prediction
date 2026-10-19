"""Soft-computing layer: pure-Python Mamdani fuzzy inference systems.

1. ``RainLikelihoodFIS`` – expert rules over humidity, pressure, pressure
   tendency and cloud cover produce a 0-100 rain likelihood. It is blended with
   the ML models (weights tuned by a genetic algorithm, see src/genetic.py).
2. ``classify_rain_type`` – fuzzy memberships over expected rainfall (mm) map
   predictions onto India Meteorological Department (IMD) intensity classes,
   with a fuzzy convective index to flag thunderstorms.

No numpy / scikit-fuzzy needed, so it runs inside a tiny Vercel function.
"""
from __future__ import annotations

from typing import Dict, List, Tuple


def trimf(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 1.0 if x == b else 0.0
    return (x - a) / (b - a) if x < b else (c - x) / (c - b)


def trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b > a else 1.0
    return (d - x) / (d - c) if d > c else 1.0


# ---------------------------------------------------------------- inputs
HUMIDITY = {
    "dry":   lambda x: trapmf(x, -1, 0, 35, 55),
    "moist": lambda x: trimf(x, 40, 60, 80),
    "humid": lambda x: trapmf(x, 65, 85, 100, 101),
}
PRESSURE = {
    "low":    lambda x: trapmf(x, 960, 960, 1004, 1011),
    "normal": lambda x: trimf(x, 1006, 1013, 1020),
    "high":   lambda x: trapmf(x, 1016, 1023, 1050, 1050),
}
TENDENCY = {  # Pressure3pm - Pressure9am (hPa)
    "falling": lambda x: trapmf(x, -20, -20, -3, -0.5),
    "steady":  lambda x: trimf(x, -2, 0, 2),
    "rising":  lambda x: trapmf(x, 0.5, 3, 20, 20),
}
CLOUD = {  # oktas 0-8
    "clear":    lambda x: trapmf(x, -1, 0, 1, 3),
    "partly":   lambda x: trimf(x, 2, 4, 6),
    "overcast": lambda x: trapmf(x, 5, 7, 8, 9),
}

# ---------------------------------------------------------------- output 0-100
OUT_GRID = [float(i) for i in range(0, 101)]
LIKELIHOOD = {
    "very_low":  lambda x: trapmf(x, 0, 0, 10, 25),
    "low":       lambda x: trimf(x, 15, 30, 45),
    "medium":    lambda x: trimf(x, 35, 50, 65),
    "high":      lambda x: trimf(x, 55, 70, 85),
    "very_high": lambda x: trapmf(x, 75, 90, 100, 100),
}

# (antecedents, consequent) – antecedents combined with min (AND)
RULES: List[Tuple[Dict[str, str], str]] = [
    ({"hum": "humid", "cloud": "overcast"}, "very_high"),
    ({"hum": "humid", "pres": "low"}, "very_high"),
    ({"hum": "humid", "tend": "falling"}, "high"),
    ({"hum": "humid", "cloud": "partly"}, "high"),
    ({"hum": "moist", "cloud": "overcast"}, "medium"),
    ({"hum": "moist", "pres": "low"}, "medium"),
    ({"hum": "moist", "tend": "falling", "cloud": "partly"}, "medium"),
    ({"hum": "moist", "pres": "normal", "cloud": "partly"}, "low"),
    ({"hum": "moist", "tend": "rising"}, "low"),
    ({"hum": "dry", "cloud": "overcast"}, "low"),
    ({"hum": "moist", "cloud": "clear"}, "very_low"),
    ({"hum": "moist", "tend": "steady", "cloud": "clear"}, "very_low"),
    ({"hum": "humid", "cloud": "clear", "pres": "normal"}, "low"),
    ({"hum": "dry"}, "very_low"),
    ({"pres": "high", "cloud": "clear"}, "very_low"),
    ({"pres": "high", "tend": "rising"}, "very_low"),
    ({"rain_today": "yes", "hum": "humid"}, "very_high"),
    ({"rain_today": "yes", "hum": "moist"}, "medium"),
]


def fuzzify(humidity: float, pressure: float, tendency: float, cloud: float, rain_today: float):
    return {
        "hum": {k: f(humidity) for k, f in HUMIDITY.items()},
        "pres": {k: f(pressure) for k, f in PRESSURE.items()},
        "tend": {k: f(tendency) for k, f in TENDENCY.items()},
        "cloud": {k: f(cloud) for k, f in CLOUD.items()},
        "rain_today": {"yes": 1.0 if rain_today >= 0.5 else 0.0, "no": 1.0 if rain_today < 0.5 else 0.0},
    }


def rain_likelihood(humidity: float, pressure: float, tendency: float, cloud: float,
                    rain_today: float = 0.0, return_details: bool = False):
    """Mamdani inference with min-AND, max aggregation, centroid defuzzification."""
    mu = fuzzify(humidity, pressure, tendency, cloud, rain_today)
    strength = {k: 0.0 for k in LIKELIHOOD}
    fired = []
    for ante, cons in RULES:
        w = min(mu[var][term] for var, term in ante.items())
        if w > 0:
            strength[cons] = max(strength[cons], w)
            fired.append({"if": ante, "then": cons, "strength": round(w, 3)})

    num = den = 0.0
    for x in OUT_GRID:
        agg = max(min(strength[k], f(x)) for k, f in LIKELIHOOD.items())
        num += x * agg
        den += agg
    value = num / den if den > 0 else 22.0  # no rule fired -> climatological base rate
    if not return_details:
        return value
    return value, {
        "memberships": {v: {t: round(m, 3) for t, m in terms.items()} for v, terms in mu.items()},
        "fired_rules": sorted(fired, key=lambda r: -r["strength"])[:6],
    }


def fis_from_features(f: Dict[str, float], return_details: bool = False):
    return rain_likelihood(f["Humidity3pm"], f["Pressure3pm"], f["PressureDelta"],
                           f["Cloud3pm"], f["RainToday"], return_details)


# ---------------------------------------------------------------- rain type
# IMD 24-h rainfall categories (mm)
INTENSITY = {
    "No rain":          lambda mm: trapmf(mm, -1, -1, 0.05, 0.3),
    "Drizzle":          lambda mm: trapmf(mm, 0.05, 0.3, 1.5, 3.0),
    "Light rain":       lambda mm: trapmf(mm, 1.5, 3.0, 12.0, 17.0),
    "Moderate rain":    lambda mm: trapmf(mm, 12.0, 17.0, 60.0, 70.0),
    "Heavy rain":       lambda mm: trapmf(mm, 60.0, 70.0, 110.0, 120.0),
    "Very heavy rain":  lambda mm: trapmf(mm, 110.0, 120.0, 200.0, 210.0),
    "Extremely heavy":  lambda mm: trapmf(mm, 200.0, 210.0, 1000.0, 1000.0),
}


def convective_index(f: Dict[str, float]) -> float:
    """Fuzzy 0-1 index of thunderstorm potential: hot AND humid AND gusty AND unstable."""
    hot = trapmf(f["MaxTemp"], 24, 30, 60, 60)
    humid = trapmf(f["Humidity3pm"], 45, 65, 100, 101)
    gusty = trapmf(f["WindGustSpeed"], 35, 55, 200, 200)
    falling = trapmf(-f["PressureDelta"], 0.5, 3, 30, 30)
    return min(hot, humid, max(gusty, falling))


def classify_rain_type(expected_mm: float, probability: float, f: Dict[str, float]) -> Dict:
    # Scale the conditional amount by belief that it rains at all.
    effective_mm = expected_mm if probability >= 0.5 else expected_mm * probability
    if probability < 0.25:
        effective_mm = 0.0
    memberships = {k: round(fn(effective_mm), 3) for k, fn in INTENSITY.items()}
    label = max(memberships, key=memberships.get)
    conv = convective_index(f)
    thunder = conv >= 0.5 and probability >= 0.45
    return {
        "label": "Thunderstorm" if thunder and label != "No rain" else label,
        "intensity": label,
        "effective_mm": round(effective_mm, 2),
        "memberships": memberships,
        "convective_index": round(conv, 3),
        "thunderstorm": bool(thunder),
    }
