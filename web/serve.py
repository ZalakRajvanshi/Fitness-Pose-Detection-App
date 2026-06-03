"""
Tiny static server for the FormCoach web app.

The app uses ES modules + getUserMedia, which browsers only allow over
http://localhost (not file://). Run this, then open the printed URL.

    python web/serve.py
"""
import http.server
import os
import socketserver
import webbrowser

PORT = 8000
ROOT = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self):
        # Avoid caching during development.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    url = f"http://localhost:{PORT}/index.html"
    print(f"FormCoach running at {url}  (Ctrl+C to stop)")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()
