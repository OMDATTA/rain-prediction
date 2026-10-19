"""Live + historical weather from Open-Meteo (free, no API key).

* Forecast API  (past 7 days + today, hourly) -> maps to model features
* Archive API   (last 10 years, daily)       -> local rain climatology
"""
from __future__ import annotations

import datetime as dt
import json
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

HOURLY = ["temperature_2m", "relative_humidity_2m", "pressure_msl", "cloud_cover",
          "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m", "precipitation",
          "precipitation_probability"]
DAILY = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum",
         "wind_gusts_10m_max", "wind_direction_10m_dominant", "precipitation_probability_max"]


def _get(url: str, params: Dict, timeout: float = 8.0) -> Dict:
    q = urllib.parse.urlencode(params, doseq=True, safe=",")
    req = urllib.request.Request(f"{url}?{q}", headers={"User-Agent": "RainCast/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def geocode(name: str, count: int = 5) -> List[Dict]:
    data = _get(GEOCODE_URL, {"name": name, "count": count, "language": "en", "format": "json"})
    return [{"name": r.get("name"), "country": r.get("country"), "admin1": r.get("admin1"),
             "latitude": r["latitude"], "longitude": r["longitude"]} for r in data.get("results", [])]


def _at(hourly: Dict, key: str, ts: str) -> Optional[float]:
    try:
        return hourly[key][hourly["time"].index(ts)]
    except (ValueError, KeyError, IndexError):
        return None


def current_features(lat: float, lon: float) -> Dict:
    """Fetch today's observations and map them to the weatherAUS feature space."""
    data = _get(FORECAST_URL, {
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "past_days": 7, "forecast_days": 2,
        "hourly": ",".join(HOURLY), "daily": ",".join(DAILY),
        "current": "temperature_2m,relative_humidity_2m,pressure_msl,cloud_cover,"
                   "wind_speed_10m,wind_gusts_10m,precipitation,weather_code",
    })
    h, d = data["hourly"], data["daily"]
    today = data["current"]["time"][:10]
    yday = (dt.date.fromisoformat(today) - dt.timedelta(days=1)).isoformat()
    t9, t15 = f"{today}T09:00", f"{today}T15:00"
    now_hour = data["current"]["time"][:13] + ":00"
    # If 3 pm hasn't happened yet, the forecast value for 3 pm is used (nowcast).

    # Rainfall: 24 h to 9 am today (weatherAUS convention)
    rain24 = 0.0
    for ts, p in zip(h["time"], h["precipitation"]):
        if f"{yday}T09:00" < ts <= t9 and p is not None:
            rain24 += p

    di = d["time"].index(today)
    oktas = lambda pct: None if pct is None else round(pct / 12.5, 1)
    features = {
        "MinTemp": d["temperature_2m_min"][di],
        "MaxTemp": d["temperature_2m_max"][di],
        "Rainfall": round(rain24, 2),
        "WindGustSpeed": d["wind_gusts_10m_max"][di],
        "WindGustDir": d["wind_direction_10m_dominant"][di],
        "WindSpeed9am": _at(h, "wind_speed_10m", t9),
        "WindSpeed3pm": _at(h, "wind_speed_10m", t15),
        "Humidity9am": _at(h, "relative_humidity_2m", t9),
        "Humidity3pm": _at(h, "relative_humidity_2m", t15),
        "Pressure9am": _at(h, "pressure_msl", t9),
        "Pressure3pm": _at(h, "pressure_msl", t15),
        "Cloud9am": oktas(_at(h, "cloud_cover", t9)),
        "Cloud3pm": oktas(_at(h, "cloud_cover", t15)),
        "Temp9am": _at(h, "temperature_2m", t9),
        "Temp3pm": _at(h, "temperature_2m", t15),
    }

    history = [{
        "date": d["time"][i],
        "rain_mm": d["precipitation_sum"][i],
        "tmax": d["temperature_2m_max"][i],
        "tmin": d["temperature_2m_min"][i],
    } for i in range(len(d["time"])) if d["time"][i] < today]

    tomorrow_idx = di + 1 if di + 1 < len(d["time"]) else di
    next24 = [{"time": ts, "precip_prob": pp, "precip_mm": pr}
              for ts, pp, pr in zip(h["time"], h["precipitation_probability"], h["precipitation"])
              if ts >= now_hour][:24]

    return {
        "location": {"latitude": data["latitude"], "longitude": data["longitude"],
                     "timezone": data.get("timezone"), "elevation": data.get("elevation")},
        "observed_at": data["current"]["time"],
        "current": data["current"],
        "features": features,
        "history_7d": history,
        "next_24h": next24,
        "reference_forecast": {
            "date": d["time"][tomorrow_idx],
            "precip_probability_max": d["precipitation_probability_max"][tomorrow_idx],
            "precip_sum": d["precipitation_sum"][tomorrow_idx],
        },
    }


def climatology(lat: float, lon: float, years: int = 10, window: int = 7) -> Dict:
    """Historical share of rainy days (>=1 mm) within +/- window days of tomorrow."""
    end = dt.date.today() - dt.timedelta(days=7)
    start = dt.date(end.year - years, 1, 1)
    data = _get(ARCHIVE_URL, {
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": "precipitation_sum",
    }, timeout=12.0)
    target = (dt.date.today() + dt.timedelta(days=1)).timetuple().tm_yday
    rainy = total = 0
    amounts = []
    by_month = [[0, 0] for _ in range(12)]
    for ts, p in zip(data["daily"]["time"], data["daily"]["precipitation_sum"]):
        if p is None:
            continue
        day = dt.date.fromisoformat(ts)
        by_month[day.month - 1][1] += 1
        if p >= 1.0:
            by_month[day.month - 1][0] += 1
        doy = day.timetuple().tm_yday
        if min(abs(doy - target), 366 - abs(doy - target)) <= window:
            total += 1
            if p >= 1.0:
                rainy += 1
                amounts.append(p)
    return {
        "years": years,
        "window_days": window,
        "rain_frequency": round(rainy / total, 4) if total else None,
        "mean_rain_mm_on_rainy_days": round(sum(amounts) / len(amounts), 2) if amounts else 0.0,
        "samples": total,
        "monthly_rain_frequency": [round(r / t, 3) if t else None for r, t in by_month],
    }
