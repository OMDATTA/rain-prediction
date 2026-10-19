"""POST /api/iot – endpoint for ESP32 / Raspberry Pi weather nodes.

Sensor payload (all optional except one reading):
{
  "device_id": "esp32-roof-01",
  "temperature": 29.4,        # DHT22 / BME280  (°C)
  "humidity": 82,             # %
  "pressure": 1006.2,         # BMP280 / BME280 (hPa)
  "pressure_3h_ago": 1008.0,  # optional, enables pressure tendency
  "rain_analog": 610,         # FC-37 / YL-83 raw ADC (0 wet … 4095 dry on ESP32)
  "rain_mm_24h": 3.5,         # tipping-bucket gauge, optional
  "wind_kmh": 14, "gust_kmh": 32, "cloud_oktas": 6,
  "lat": 19.07, "lon": 72.88  # optional: blend with live Open-Meteo data
}
Returns a compact response the micro-controller can act on (LED / buzzer).
Set IOT_API_KEY in Vercel env vars to require an X-Device-Key header.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rainlib.engine import predict  # noqa: E402
from rainlib.http import JSONHandler  # noqa: E402


def sensor_to_inputs(s: dict) -> dict:
    t, h, p = s.get("temperature"), s.get("humidity"), s.get("pressure")
    p_prev = s.get("pressure_3h_ago")
    raw = {
        "Temp3pm": t, "Temp9am": t, "MaxTemp": t, "MinTemp": (t - 6) if t is not None else None,
        "Humidity3pm": h, "Humidity9am": h,
        "Pressure3pm": p, "Pressure9am": p_prev if p_prev is not None else p,
        "WindSpeed3pm": s.get("wind_kmh"), "WindSpeed9am": s.get("wind_kmh"),
        "WindGustSpeed": s.get("gust_kmh"),
        "Cloud3pm": s.get("cloud_oktas"), "Cloud9am": s.get("cloud_oktas"),
        "Rainfall": s.get("rain_mm_24h"),
    }
    # Capacitive / resistive rain sensor: low ADC = wet plate = raining now
    adc = s.get("rain_analog")
    if adc is not None:
        adc_max = float(s.get("adc_max", 4095))
        wetness = max(0.0, min(1.0, 1 - float(adc) / adc_max))
        raw["RainToday"] = 1 if wetness > 0.5 else 0
        if raw["Rainfall"] is None:
            raw["Rainfall"] = round(wetness * 10, 1) if wetness > 0.5 else 0.0
    return raw


class handler(JSONHandler):
    def do_GET(self):
        self._send(200, {"usage": "POST sensor JSON (see docstring in api/iot.py)"})

    def do_POST(self):
        key = os.environ.get("IOT_API_KEY")
        if key and self.headers.get("X-Device-Key") != key:
            self._send(401, {"error": "invalid device key"})
            return
        try:
            s = self.body()
            raw = sensor_to_inputs(s)
            clim = None
            if s.get("lat") is not None and s.get("lon") is not None:
                try:
                    from rainlib.weather import climatology, current_features
                    live = current_features(float(s["lat"]), float(s["lon"]))["features"]
                    raw = {**live, **{k: v for k, v in raw.items() if v is not None}}
                    clim = climatology(float(s["lat"]), float(s["lon"]))["rain_frequency"]
                except Exception:
                    pass  # sensor-only prediction still works offline
            r = predict(raw, climatology=clim)
            pct = round(r["probability"] * 100)
            self._send(200, {
                "device_id": s.get("device_id", "unknown"),
                "rain_probability": pct,
                "will_rain": r["will_rain"],
                "rain_type": r["rain_type"]["label"],
                "condition": r["condition"],
                "led": "red" if pct >= 70 else "amber" if pct >= 40 else "green",
                "alert": r["will_rain"] and r["rain_type"]["label"] in
                         ("Heavy rain", "Very heavy rain", "Extremely heavy", "Thunderstorm"),
                "imputed": r["inputs_imputed"],
            })
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover
            self._send(500, {"error": f"iot prediction failed: {exc}"})
