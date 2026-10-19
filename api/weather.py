"""GET /api/weather?lat=19.07&lon=72.88[&predict=1]
GET /api/weather?q=Mumbai          (geocode search)

Returns live conditions mapped to model features, the past 7 days of
observations, 10-year climatology and (optionally) the hybrid prediction.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rainlib.engine import predict  # noqa: E402
from rainlib.http import JSONHandler  # noqa: E402
from rainlib.weather import climatology, current_features, geocode  # noqa: E402


class handler(JSONHandler):
    def do_GET(self):
        q = self.query()
        try:
            if q.get("q"):
                self._send(200, {"results": geocode(q["q"])})
                return
            lat, lon = float(q["lat"]), float(q["lon"])
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("lat/lon out of range")
        except (KeyError, ValueError) as exc:
            self._send(400, {"error": f"provide ?q=<city> or valid ?lat=&lon= ({exc})"})
            return
        try:
            live = current_features(lat, lon)
            try:
                live["climatology"] = climatology(lat, lon)
            except Exception as exc:  # archive API is optional
                live["climatology"] = {"error": str(exc), "rain_frequency": None}
            if q.get("predict", "1") != "0":
                live["prediction"] = predict(live["features"], live["climatology"].get("rain_frequency"))
            self._send(200, live)
        except Exception as exc:
            self._send(502, {"error": f"weather service unavailable: {exc}"})
