import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import DATA

PORT = 8766
PROFILE = DATA / "chrome_profile"
KEEP_AWAKE = [
    "--disable-features=CalculateNativeWinOcclusion",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
    "--disable-background-timer-throttling",
]
PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Hands Free Computer - listening</title>
<style>
 body{font-family:Segoe UI,Arial,sans-serif;background:#16181d;color:#e8e8e8;margin:0;padding:14px}
 #status{font-size:15px;font-weight:600} #text{margin-top:8px;font-size:14px;color:#9fd89f;min-height:40px}
 #note{margin-top:8px;font-size:12px;color:#888}
</style></head><body>
<div id="status">Starting...</div><div id="text"></div>
<div id="note">Keep this window open. Your speech is recognised by Google while it is.</div>
<script>
const status = document.getElementById('status'), text = document.getElementById('text');
function post(path, body){ return fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})}); }
const R = window.SpeechRecognition || window.webkitSpeechRecognition;
let blocked = false, running = false, startedAt = 0, askedAt = 0, lastResult = Date.now(), rec = null;
function report(what){ post('/event', {error: what}).catch(() => {}); }
function begin(){
  if (blocked) return;
  try { if (rec) { rec.onend = null; rec.onerror = null; rec.onresult = null; rec.abort(); } } catch (x) {}
  rec = new R();
  rec.lang = '__LANG__'; rec.continuous = true; rec.interimResults = true; rec.maxAlternatives = 1;
  rec.onstart = () => { running = true; startedAt = Date.now(); status.textContent = 'Listening'; };
  rec.onresult = e => {
    lastResult = Date.now();
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const res = e.results[i], said = res[0].transcript.trim();
      text.textContent = said;
      post('/heard', {text: said, final: res.isFinal, confidence: res[0].confidence});
    }
  };
  rec.onerror = e => {
    if (e.error === 'not-allowed' || e.error === 'service-not-allowed') {
      blocked = true; status.textContent = 'Microphone blocked. Allow the microphone for this window, then restart the app.';
    } else if (e.error !== 'no-speech' && e.error !== 'aborted') { status.textContent = 'Problem: ' + e.error; }
    if (e.error !== 'no-speech') report(e.error);
  };
  rec.onend = () => { running = false; setTimeout(begin, 150); };
  askedAt = Date.now();
  try { rec.start(); } catch (x) { running = false; report('start-failed: ' + x.name); }
}
if (!R) { status.textContent = 'This browser has no speech recognition.'; report('unsupported'); }
else {
  document.addEventListener('visibilitychange', () => { report('page ' + document.visibilityState); if (document.visibilityState === 'visible') begin(); });
  begin();
  setInterval(() => {
    post('/ping').catch(() => {});
    const now = Date.now();
    if (!blocked && !running && now - askedAt > 20000) { report('watchdog restart'); begin(); }
    else if (running && now - startedAt > 55000 && now - lastResult > 8000) { try { rec.stop(); } catch (x) { begin(); } }
  }, 2000);
}
</script></body></html>"""


def find_chrome():
    for base in (os.environ.get("PROGRAMFILES"), os.environ.get("PROGRAMFILES(X86)"), os.environ.get("LOCALAPPDATA")):
        if base:
            path = os.path.join(base, "Google", "Chrome", "Application", "chrome.exe")
            if os.path.exists(path):
                return path
    return None


def close_stale_windows():
    query = ('Get-CimInstance Win32_Process -Filter "Name=\'chrome.exe\'" | '
             "Where-Object { $_.CommandLine -like '*chrome_profile*' } | "
             "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }")
    subprocess.run(["powershell", "-NoProfile", "-Command", query], capture_output=True,
                   creationflags=subprocess.CREATE_NO_WINDOW)


class WebSpeech:
    def __init__(self, on_result, on_event=None, lang="en-IN", port=PORT):
        self.on_result = on_result
        self.on_event = on_event or (lambda error: None)
        self.lang = lang
        self.port = port
        self.last_ping = 0.0
        self.flags = list(KEEP_AWAKE)
        self._server = None
        self._proc = None

    @property
    def connected(self):
        return time.time() - self.last_ping < 6

    def start(self, open_window=True, screen=(1920, 1080)):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self, body=b"{}", kind="application/json"):
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._reply(PAGE.replace("__LANG__", outer.lang).encode("utf-8"), "text/html; charset=utf-8")

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                try:
                    data = json.loads(raw or b"{}")
                except ValueError:
                    data = {}
                outer.last_ping = time.time()
                if self.path == "/heard" and data.get("text"):
                    outer.on_result(str(data["text"]), bool(data.get("final")))
                elif self.path == "/event" and data.get("error"):
                    outer.on_event(str(data["error"]))
                self._reply()

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        if open_window:
            self.open_window(screen)

    def open_window(self, screen=(1920, 1080)):
        chrome = find_chrome()
        if chrome is None:
            raise RuntimeError("Google Chrome was not found")
        close_stale_windows()
        PROFILE.mkdir(parents=True, exist_ok=True)
        w, h = 430, 190
        self._proc = subprocess.Popen([
            chrome, f"--user-data-dir={PROFILE}", "--no-first-run", "--no-default-browser-check", *self.flags,
            f"--app=http://127.0.0.1:{self.port}/", f"--window-size={w},{h}",
            f"--window-position={screen[0] - w - 20},{20}"])

    def close(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._proc is not None:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self._proc.pid)], capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        close_stale_windows()
