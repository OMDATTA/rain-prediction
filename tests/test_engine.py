import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rainlib import predict  # noqa: E402
from rainlib.features import FEATURES, build_vector  # noqa: E402
from rainlib.fuzzy import classify_rain_type, rain_likelihood  # noqa: E402
from rainlib.model import load_model  # noqa: E402

WET = {"Humidity9am": 95, "Humidity3pm": 92, "Pressure9am": 1004, "Pressure3pm": 999,
       "Cloud9am": 8, "Cloud3pm": 8, "Rainfall": 30, "WindGustSpeed": 60, "MaxTemp": 30, "MinTemp": 24}
DRY = {"Humidity9am": 40, "Humidity3pm": 22, "Pressure9am": 1024, "Pressure3pm": 1022,
       "Cloud9am": 0, "Cloud3pm": 1, "Rainfall": 0, "WindGustSpeed": 25}


def test_model_file_is_valid():
    m = load_model()
    assert m["features"] == FEATURES
    assert len(m["gbm_classifier"]["trees"]) > 10
    w = m["ensemble"]["weights"]
    assert abs(sum(w.values()) - 1) < 1e-3


def test_vector_handles_partial_input():
    v = build_vector({"Humidity3pm": 80}, load_model()["medians"])
    assert len(v) == len(FEATURES)


def test_wet_beats_dry():
    wet, dry = predict(WET), predict(DRY)
    assert wet["probability"] > 0.7 > 0.2 > dry["probability"]
    assert wet["will_rain"] and not dry["will_rain"]
    assert dry["rain_type"]["label"] == "No rain"


def test_fuzzy_monotonic_in_humidity():
    lo = rain_likelihood(30, 1012, 0, 4)
    hi = rain_likelihood(95, 1012, 0, 4)
    assert hi > lo


def test_rain_type_categories():
    f = dict(zip(FEATURES, build_vector(DRY, load_model()["medians"])))
    assert classify_rain_type(90, 0.9, f)["intensity"] == "Heavy rain"
    assert classify_rain_type(0.5, 0.1, f)["label"] == "No rain"


def test_climatology_prior_shifts_probability():
    base = predict(WET)["probability"]
    assert predict(WET, climatology=0.05)["probability"] < base
    assert predict(WET, climatology=0.9)["probability"] >= base - 1e-9


def test_iot_mapping():
    sys.path.insert(0, os.path.join(ROOT, "api"))
    from iot import sensor_to_inputs  # noqa: E402
    raw = sensor_to_inputs({"temperature": 30, "humidity": 90, "pressure": 1002, "rain_analog": 400})
    assert raw["RainToday"] == 1 and raw["Humidity3pm"] == 90
    json.dumps(predict(raw))
