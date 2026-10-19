"""GET /api/health – liveness + model metadata."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rainlib.http import JSONHandler  # noqa: E402
from rainlib.model import load_model  # noqa: E402


class handler(JSONHandler):
    def do_GET(self):
        m = load_model()
        self._send(200, {"status": "ok", "model_version": m["version"], "trained_on": m["trained_on"],
                         "ensemble": m["ensemble"], "metrics": m["metrics"]})
