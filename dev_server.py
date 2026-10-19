"""Local development server that mimics Vercel: serves /public and routes /api/<name>
to api/<name>.py handlers.  Run:  python dev_server.py  ->  http://localhost:8000
(or use `vercel dev` if you have the Vercel CLI)."""
import importlib.util
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rainlib.http import JSONHandler  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
PUBLIC = os.path.join(ROOT, "public")
_cache = {}


def load(name):
    if name not in _cache:
        path = os.path.join(ROOT, "api", f"{name}.py")
        if not os.path.exists(path):
            return None
        spec = importlib.util.spec_from_file_location(f"api_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _cache[name] = mod.handler
    return _cache[name]


class Dev(JSONHandler, SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=PUBLIC, **kw)

    def _route(self, method):
        if self.path.startswith("/api/"):
            name = self.path[5:].split("?")[0].strip("/")
            h = load(name)
            if h is None:
                self.send_error(404)
                return True
            fn = getattr(h, f"do_{method}", None)
            if fn is None:
                self.send_error(405)
            else:
                fn(self)  # run the Vercel handler method against this request
            return True
        return False

    def do_GET(self):
        if not self._route("GET"):
            super().do_GET()

    def do_POST(self):
        if not self._route("POST"):
            self.send_error(405)

    def do_OPTIONS(self):
        self._route("OPTIONS")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", sys.argv[1] if len(sys.argv) > 1 else 8000))
    print(f"RainCast dev server on http://localhost:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), Dev).serve_forever()
