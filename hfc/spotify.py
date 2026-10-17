import base64
import re
import hashlib
import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

from .config import DATA

TOKEN_FILE = DATA / "spotify_token.json"
REDIRECT = "http://127.0.0.1:8765/callback"
SCOPES = "user-modify-playback-state user-read-playback-state"
AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API = "https://api.spotify.com/v1"


GENERIC = re.compile(r"\b(?:songs?|music|tracks?|playlists?|hits|tunes|beats|vibes|mix)\b")
VAGUE_START = re.compile(r"^(?:a|an|some|any|my|me|good|nice|the best|best|popular|latest|new|top|random)\b")


def describe(query):
    q = re.sub(r"\s+", " ", query.lower()).strip()
    if not GENERIC.search(q):
        return None
    if not VAGUE_START.match(q) and len(q.split()) > 4:
        return None
    words = [w for w in re.sub(r"[^a-z0-9 ]", " ", q).split()
             if w not in {"a", "an", "some", "any", "me", "the", "for", "please", "on", "of"}]
    words = [w for w in words if not GENERIC.fullmatch(w)]
    return " ".join(words) or "top hits"


class SpotifyError(Exception):
    pass


class Spotify:
    def __init__(self, client_id=None, token_file=TOKEN_FILE):
        self.client_id = client_id or os.environ.get("SPOTIFY_CLIENT_ID", "")
        self.token_file = token_file
        self._tokens = None

    @property
    def ready(self):
        return bool(self.client_id) and self.token_file.exists()

    def _http(self, method, url, body=None, headers=None, form=False):
        data = None
        headers = dict(headers or {})
        if body is not None:
            if form:
                data = urllib.parse.urlencode(body).encode()
                headers["Content-Type"] = "application/x-www-form-urlencoded"
            else:
                data = json.dumps(body).encode()
                headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read()
                try:
                    data = json.loads(raw) if raw else {}
                except ValueError:
                    data = {}
                return data if isinstance(data, dict) else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")[:200]
            raise SpotifyError(f"{e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise SpotifyError(f"network: {e.reason}") from e

    def _api(self, method, path, body=None):
        return self._http(method, API + path, body, {"Authorization": "Bearer " + self._access_token()})

    def login(self, open_browser=webbrowser.open):
        if not self.client_id:
            raise SpotifyError("SPOTIFY_CLIENT_ID is not set in .env")
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        state = secrets.token_urlsafe(16)
        result = {}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                if query.get("state", [""])[0] == state:
                    result.update({k: v[0] for k, v in query.items()})
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<h2>Spotify is connected. You can close this tab.</h2>")

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 8765), Handler)
        server.timeout = 1
        open_browser(AUTH_URL + "?" + urllib.parse.urlencode({
            "client_id": self.client_id, "response_type": "code", "redirect_uri": REDIRECT,
            "scope": SCOPES, "state": state, "code_challenge_method": "S256", "code_challenge": challenge}))
        deadline = time.time() + 180
        while not result and time.time() < deadline:
            server.handle_request()
        server.server_close()
        if "code" not in result:
            raise SpotifyError(result.get("error", "no approval received within three minutes"))
        self._store(self._http("POST", TOKEN_URL, form=True, body={
            "grant_type": "authorization_code", "code": result["code"], "redirect_uri": REDIRECT,
            "client_id": self.client_id, "code_verifier": verifier}))

    def _store(self, tokens):
        old = self._load() or {}
        tokens.setdefault("refresh_token", old.get("refresh_token"))
        tokens["expires_at"] = time.time() + tokens.get("expires_in", 3600) - 60
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        self.token_file.write_text(json.dumps(tokens), encoding="utf-8")
        self._tokens = tokens

    def _load(self):
        if self._tokens is None and self.token_file.exists():
            self._tokens = json.loads(self.token_file.read_text(encoding="utf-8"))
        return self._tokens

    def _access_token(self):
        tokens = self._load()
        if not tokens:
            raise SpotifyError("not logged in")
        if time.time() >= tokens.get("expires_at", 0):
            self._store(self._http("POST", TOKEN_URL, form=True, body={
                "grant_type": "refresh_token", "refresh_token": tokens["refresh_token"],
                "client_id": self.client_id}))
        return self._tokens["access_token"]

    def _device(self, wake=None):
        deadline = None
        while True:
            devices = self._api("GET", "/me/player/devices").get("devices", [])
            if devices:
                active = [d for d in devices if d.get("is_active")]
                return (active or devices)[0]["id"]
            if deadline is None:
                if wake is None:
                    break
                wake()
                deadline = time.time() + 15
            elif time.time() > deadline:
                break
            time.sleep(1.5)
        raise SpotifyError("no Spotify device is open")

    @staticmethod
    def _pick(query, items):
        import difflib
        q = query.lower()

        def score(pair):
            index, track = pair
            title = track["name"].lower()
            names = title + " " + " ".join(a["name"].lower() for a in track.get("artists", []))
            return max(difflib.SequenceMatcher(None, q, title).ratio(),
                       difflib.SequenceMatcher(None, q, names).ratio()) - 0.01 * index

        return max(enumerate(items), key=score)[1]

    def play(self, query, wake=None):
        found = self._api("GET", "/search?" + urllib.parse.urlencode({"q": query, "type": "track", "limit": 10}))
        items = found.get("tracks", {}).get("items", [])
        if not items:
            raise SpotifyError(f"no track found for {query}")
        track = self._pick(query, items)
        device = urllib.parse.quote(self._device(wake))
        self._api("PUT", "/me/player/play?device_id=" + device,
                  {"context_uri": track["album"]["uri"], "offset": {"uri": track["uri"]}, "position_ms": 0})
        return track["name"], ", ".join(a["name"] for a in track.get("artists", []))

    def play_playlist(self, mood, wake=None):
        found = self._api("GET", "/search?" + urllib.parse.urlencode({"q": f"{mood} songs", "type": "playlist", "limit": 10}))
        items = [p for p in found.get("playlists", {}).get("items", []) if p and p.get("uri")]
        if not items:
            raise SpotifyError(f"no playlist found for {mood}")
        words = set(mood.split())
        best = max(enumerate(items), key=lambda pair: (len(words & set(re.sub(r"[^a-z0-9 ]", " ", pair[1]["name"].lower()).split())), -pair[0]))[1]
        device = urllib.parse.quote(self._device(wake))
        try:
            self._api("PUT", "/me/player/shuffle?state=true&device_id=" + device)
        except SpotifyError:
            pass
        self._api("PUT", "/me/player/play?device_id=" + device, {"context_uri": best["uri"]})
        return best["name"]
