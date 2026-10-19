"""POST /api/predict  – predict rain from user-supplied weather conditions.

Body (any subset; missing fields are imputed with training medians):
{
  "MinTemp": 22, "MaxTemp": 31, "Rainfall": 4.2, "WindGustSpeed": 46, "WindGustDir": "SW",
  "WindSpeed9am": 11, "WindSpeed3pm": 20, "Humidity9am": 84, "Humidity3pm": 71,
  "Pressure9am": 1008.4, "Pressure3pm": 1005.1, "Cloud9am": 7, "Cloud3pm": 6,
  "Temp9am": 25, "Temp3pm": 29, "RainToday": "Yes", "climatology": 0.38
}
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rainlib.engine import predict  # noqa: E402
from rainlib.features import RAW_INPUTS  # noqa: E402
from rainlib.http import JSONHandler  # noqa: E402


class handler(JSONHandler):
    def do_GET(self):
        self._send(200, {"usage": "POST JSON weather inputs to this endpoint", "inputs": RAW_INPUTS})

    def do_POST(self):
        try:
            data = self.body()
            clim = data.pop("climatology", None)
            clim = float(clim) if clim not in (None, "") else None
            self._send(200, predict(data, climatology=clim))
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover
            self._send(500, {"error": f"prediction failed: {exc}"})
