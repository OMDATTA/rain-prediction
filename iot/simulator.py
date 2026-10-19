"""Simulate one or more IoT weather nodes without hardware.

    python iot/simulator.py --url http://localhost:8000/api/iot --nodes 3 --interval 5
    python iot/simulator.py --url https://your-app.vercel.app/api/iot --storm

Readings follow a slow random walk; --storm ramps humidity up and pressure down
so you can watch the probability and LED colour change.
"""
import argparse
import json
import random
import time
import urllib.request


def post(url, payload, key=None):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **({"X-Device-Key": key} if key else {})})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/api/iot")
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--interval", type=float, default=3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--storm", action="store_true")
    ap.add_argument("--key", default=None)
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    a = ap.parse_args()

    nodes = [{"device_id": f"sim-node-{i + 1}", "temperature": random.uniform(24, 32),
              "humidity": random.uniform(50, 75), "pressure": random.uniform(1008, 1014),
              "rain_analog": random.randint(3000, 4095)} for i in range(a.nodes)]
    for step in range(a.steps):
        for n in nodes:
            prev = n["pressure"]
            drift = -0.6 if a.storm else random.uniform(-0.3, 0.3)
            n["pressure"] += drift
            n["humidity"] = min(99, max(20, n["humidity"] + (3 if a.storm else random.uniform(-2, 2))))
            n["temperature"] += random.uniform(-0.3, 0.3)
            if a.storm and step > a.steps // 2:
                n["rain_analog"] = max(200, n["rain_analog"] - 600)
            payload = {**{k: round(v, 2) if isinstance(v, float) else v for k, v in n.items()},
                       "pressure_3h_ago": round(prev + 1.5 * -drift * 3, 2)}
            if a.lat is not None and a.lon is not None:
                payload.update(lat=a.lat, lon=a.lon)
            try:
                r = post(a.url, payload, a.key)
                print(f"[{n['device_id']}] T={payload['temperature']}°C RH={payload['humidity']}% "
                      f"P={payload['pressure']}hPa -> {r['rain_probability']:3d}% {r['rain_type']:<15} LED={r['led']}")
            except Exception as exc:
                print(f"[{n['device_id']}] error: {exc}")
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
