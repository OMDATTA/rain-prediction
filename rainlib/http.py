"""Tiny helpers shared by the Vercel Python functions in /api."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse


class JSONHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Device-Key")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):  # CORS pre-flight (browsers, ESP32 web clients)
        self._send(204, {})

    def query(self) -> dict:
        return {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}

    def body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            data = json.loads(self.rfile.read(n).decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except (ValueError, UnicodeDecodeError):
            raise ValueError("Request body must be a JSON object")

    def log_message(self, *args):  # keep Vercel logs clean
        pass
