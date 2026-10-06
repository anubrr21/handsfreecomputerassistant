import ctypes
import datetime
import difflib
import json
import os
import re
import subprocess
import threading
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

from pynput import keyboard as kb
from pynput import mouse as ms

HOME = Path.home()
KNOWN_FOLDERS = {
    "desktop": HOME / "Desktop", "documents": HOME / "Documents", "downloads": HOME / "Downloads",
    "pictures": HOME / "Pictures", "photos": HOME / "Pictures", "music": HOME / "Music",
    "videos": HOME / "Videos", "home": HOME, "user": HOME, "screenshots": HOME / "Pictures" / "Screenshots",
}
SEARCH_ROOTS = ["Desktop", "Documents", "Downloads", "Pictures", "Music", "Videos", "OneDrive"]
SKIP_DIRS = {"node_modules", ".git", ".venv", "venv", "__pycache__", "AppData", "site-packages", "$RECYCLE.BIN"}

APP_ALIASES = {
    "chrome": "Google Chrome", "google chrome": "Google Chrome", "browser": "Google Chrome",
    "edge": "Microsoft Edge", "vs code": "Visual Studio Code", "vscode": "Visual Studio Code",
    "code": "Visual Studio Code", "word": "Word", "excel": "Excel", "powerpoint": "PowerPoint",
    "whatsapp": "WhatsApp", "files": "File Explorer", "explorer": "File Explorer",
    "file manager": "File Explorer", "terminal": "Terminal", "command prompt": "Command Prompt",
    "cmd": "Command Prompt", "calc": "Calculator", "paint": "Paint", "camera": "Camera",
    "store": "Microsoft Store", "task manager": "Task Manager",
}
APP_COMMANDS = {
    "File Explorer": "explorer.exe", "Task Manager": "taskmgr.exe", "Settings": "ms-settings:",
    "Command Prompt": "cmd.exe", "Notepad": "notepad.exe", "Calculator": "calc.exe",
}
SITES = {
    "youtube": "https://www.youtube.com", "gmail": "https://mail.google.com", "google": "https://www.google.com",
    "github": "https://github.com", "whatsapp web": "https://web.whatsapp.com", "maps": "https://maps.google.com",
    "chatgpt": "https://chatgpt.com", "linkedin": "https://www.linkedin.com",
    "instagram": "https://www.instagram.com", "netflix": "https://www.netflix.com", "amazon": "https://www.amazon.in",
    "twitter": "https://x.com", "facebook": "https://www.facebook.com", "drive": "https://drive.google.com",
}
KEY_NAMES = {
    "ctrl": kb.Key.ctrl, "control": kb.Key.ctrl, "alt": kb.Key.alt, "shift": kb.Key.shift, "win": kb.Key.cmd,
    "windows": kb.Key.cmd, "enter": kb.Key.enter, "return": kb.Key.enter, "esc": kb.Key.esc,
    "escape": kb.Key.esc, "tab": kb.Key.tab, "space": kb.Key.space, "backspace": kb.Key.backspace,
    "delete": kb.Key.delete, "up": kb.Key.up, "down": kb.Key.down, "left": kb.Key.left, "right": kb.Key.right,
    "home": kb.Key.home, "end": kb.Key.end, "pageup": kb.Key.page_up, "pagedown": kb.Key.page_down,
    "printscreen": kb.Key.print_screen, **{f"f{i}": getattr(kb.Key, f"f{i}") for i in range(1, 13)},
}
SHORTCUTS = {
    "copy": "ctrl+c", "paste": "ctrl+v", "cut": "ctrl+x", "undo": "ctrl+z", "redo": "ctrl+y",
    "select all": "ctrl+a", "save": "ctrl+s", "find": "ctrl+f", "new tab": "ctrl+t", "close tab": "ctrl+w",
    "reopen tab": "ctrl+shift+t", "next tab": "ctrl+tab", "previous tab": "ctrl+shift+tab",
    "refresh": "f5", "go back": "alt+left", "go forward": "alt+right", "new window": "ctrl+n",
    "close window": "alt+f4", "switch window": "alt+tab", "show desktop": "win+d", "task view": "win+tab",
    "maximize": "win+up", "minimize": "win+down", "snap left": "win+left", "snap right": "win+right",
    "zoom in": "ctrl+=", "zoom out": "ctrl+-", "full screen": "f11", "screenshot": "win+printscreen",
    "enter": "enter", "escape": "esc", "start menu": "win", "notifications": "win+n", "emoji": "win+.",
}


def _norm(text):
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def _best(query, names, cutoff):
    q = _norm(query)
    table = {_norm(n): n for n in names}
    if q in table:
        return table[q]
    contained = [n for n in table if q and (q in n or n in q)]
    if contained:
        return table[min(contained, key=lambda n: abs(len(n) - len(q)))]
    close = difflib.get_close_matches(q, list(table), n=1, cutoff=cutoff)
    return table[close[0]] if close else None


class Laptop:
    def __init__(self, dry_run=False, spotify_api=None):
        self.dry_run = dry_run
        self.spotify_api = spotify_api
        self.log = []
        self._kb = kb.Controller()
        self._mouse = ms.Controller()
        self.apps = {}
        if not dry_run:
            threading.Thread(target=self._index_apps, daemon=True).start()

    def _do(self, kind, detail, fn):
        self.log.append((kind, detail))
        if not self.dry_run:
            fn()

    def _index_apps(self):
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "Get-StartApps | ConvertTo-Json -Compress"],
                capture_output=True, text=True, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW,
            ).stdout
            self.apps = {a["Name"]: a["AppID"] for a in json.loads(out) if a.get("Name")}
        except Exception:
            self.apps = {}

    def confirm(self, question):
        if self.dry_run:
            self.log.append(("confirm", question))
            return False
        flags = 0x4 | 0x20 | 0x1000 | 0x40000 | 0x100
        return ctypes.windll.user32.MessageBoxW(0, question, "Hands Free Computer", flags) == 6

    def open_app(self, name):
        wanted = APP_ALIASES.get(_norm(name), name)
        if wanted in APP_COMMANDS:
            self._do("app", wanted, lambda: os.startfile(APP_COMMANDS[wanted]))
            return f"Opening {wanted}."
        match = _best(wanted, list(self.apps) + list(APP_COMMANDS), cutoff=0.7)
        if match is None:
            return None
        if match in APP_COMMANDS:
            self._do("app", match, lambda: os.startfile(APP_COMMANDS[match]))
        else:
            target = f"shell:AppsFolder\\{self.apps[match]}"
            self._do("app", match, lambda: subprocess.Popen(["explorer.exe", target]))
        return f"Opening {match}."

    def find_paths(self, query, want="any", limit=5):
        q = _norm(query)
        if not q:
            return []
        scored = []
        for root in SEARCH_ROOTS:
            base = HOME / root
            if not base.exists():
                continue
            for dirpath, dirnames, filenames in os.walk(base):
                depth = len(Path(dirpath).relative_to(base).parts)
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
                if depth >= 4:
                    dirnames[:] = []
                names = ([(d, True) for d in dirnames] if want != "file" else []) + \
                        ([(f, False) for f in filenames] if want != "folder" else [])
                for name, is_dir in names:
                    stem = _norm(name if is_dir else Path(name).stem)
                    if not stem:
                        continue
                    score = 1.0 if stem == q else 0.9 if q in stem else difflib.SequenceMatcher(None, q, stem).ratio()
                    if score >= 0.72:
                        scored.append((score - 0.01 * depth, str(Path(dirpath) / name)))
        scored.sort(reverse=True)
        return [p for _, p in scored[:limit]]

    def open_path(self, query, want="any"):
        key = _norm(query).removesuffix(" folder").strip()
        if want != "file" and key in KNOWN_FOLDERS:
            self._do("path", str(KNOWN_FOLDERS[key]), lambda: os.startfile(KNOWN_FOLDERS[key]))
            return f"Opening {key}."
        found = self.find_paths(query, want)
        if not found:
            return None
        self._do("path", found[0], lambda: os.startfile(found[0]))
        return f"Opening {Path(found[0]).name}."

    def open_url(self, url):
        if not re.match(r"^[a-z][a-z0-9+.-]*:", url):
            url = "https://" + url
        self._do("url", url, lambda: webbrowser.open(url))
        return "Opening it in your browser."

    def open_site(self, name):
        key = _norm(name)
        if key in SITES:
            return self.open_url(SITES[key])
        if re.fullmatch(r"[a-z0-9-]+(\.[a-z0-9-]+)+(/\S*)?", name.strip().lower().replace(" dot ", ".")):
            return self.open_url(name.strip().lower().replace(" dot ", "."))
        return None

    def web_search(self, query):
        self.open_url("https://www.google.com/search?q=" + urllib.parse.quote_plus(query))
        return f"Searching the web for {query}."

    def youtube(self, query, play=True):
        results = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(query)
        if play and not self.dry_run:
            try:
                req = urllib.request.Request(results, headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en"})
                page = urllib.request.urlopen(req, timeout=6).read().decode("utf-8", "ignore")
                video = re.search(r'"videoId":"([\w-]{11})"', page)
                if video:
                    self.open_url("https://www.youtube.com/watch?v=" + video.group(1))
                    return f"Playing {query} on YouTube."
            except Exception:
                pass
        self.open_url(results)
        return f"Here are YouTube results for {query}."

    def spotify(self, query):
        failed = ""
        if self.spotify_api is not None and self.spotify_api.ready and not self.dry_run:
            try:
                name, artist = self.spotify_api.play(query, wake=lambda: os.startfile("spotify:"))
                self.log.append(("spotify-play", query))
                return f"Playing {name} by {artist} on Spotify."
            except Exception as e:
                self.log.append(("spotify-error", str(e)))
                failed = str(e)
        uri = "spotify:search:" + urllib.parse.quote(query)
        if _best("Spotify", list(self.apps), cutoff=0.9) or self.dry_run:
            self._do("spotify", query, lambda: os.startfile(uri))
            if failed:
                return f"I couldn't start it ({failed[:60]}), so Spotify is showing {query}."
            return f"Spotify is showing {query}. Pick the track to start it."
        self.open_url("https://open.spotify.com/search/" + urllib.parse.quote(query))
        return f"Spotify web is showing {query}."

    def press(self, combo):
        keys = [KEY_NAMES.get(k.strip().lower(), k.strip().lower()) for k in combo.split("+") if k.strip()]

        def go():
            for k in keys:
                self._kb.press(k)
            for k in reversed(keys):
                self._kb.release(k)

        self._do("keys", combo, go)
        return f"Pressed {combo}."

    def shortcut(self, name):
        if name not in SHORTCUTS:
            return None
        self.press(SHORTCUTS[name])
        return f"{name.capitalize()}."

    def type_text(self, text):
        self._do("type", text, lambda: self._kb.type(text))
        return "Typed it."

    def click(self, kind="left"):
        button = ms.Button.right if kind == "right" else ms.Button.left
        self._do("click", kind, lambda: self._mouse.click(button, 2 if kind == "double" else 1))
        return f"{kind.capitalize()} click."

    def scroll(self, direction, amount=5):
        self._do("scroll", direction, lambda: self._mouse.scroll(0, amount if direction == "up" else -amount))
        return f"Scrolling {direction}."

    def media(self, action):
        key = {"play": kb.Key.media_play_pause, "pause": kb.Key.media_play_pause,
               "next": kb.Key.media_next, "previous": kb.Key.media_previous,
               "mute": kb.Key.media_volume_mute}[action]
        self._do("media", action, lambda: self._kb.tap(key))
        return {"play": "Playing.", "pause": "Paused.", "next": "Next track.",
                "previous": "Previous track.", "mute": "Toggled mute."}[action]

    def volume(self, change=None, level=None):
        def taps(key, n):
            for _ in range(n):
                self._kb.tap(key)

        if level is not None:
            level = max(0, min(100, int(level)))
            self._do("volume", f"={level}", lambda: (taps(kb.Key.media_volume_down, 50),
                                                      taps(kb.Key.media_volume_up, level // 2)))
            return f"Volume set to {level} percent."
        key = kb.Key.media_volume_up if change > 0 else kb.Key.media_volume_down
        self._do("volume", f"{change:+d}", lambda: taps(key, abs(change)))
        return "Volume up." if change > 0 else "Volume down."

    def lock(self):
        self._do("system", "lock", ctypes.windll.user32.LockWorkStation)
        return "Locking."

    def power(self, action):
        if not self.confirm(f"Really {action} the computer now?"):
            return f"Okay, not doing the {action}."
        command = {"sleep": "rundll32.exe powrprof.dll,SetSuspendState 0,1,0",
                   "restart": "shutdown /r /t 5", "shutdown": "shutdown /s /t 5"}[action]
        self._do("system", action, lambda: subprocess.Popen(command, shell=True))
        return f"Okay, {action} now."

    def time(self):
        now = datetime.datetime.now()
        return f"It is {now.strftime('%I:%M %p').lstrip('0')} on {now.strftime('%A, %d %B')}."

    def battery(self):
        class Status(ctypes.Structure):
            _fields_ = [("ac", ctypes.c_byte), ("flag", ctypes.c_byte), ("percent", ctypes.c_byte),
                        ("saver", ctypes.c_byte), ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]

        s = Status()
        ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))
        if s.percent < 0 or s.percent > 100:
            return "I can't read the battery level."
        return f"Battery is at {s.percent} percent" + (", plugged in." if s.ac == 1 else ", on battery.")

    def run_powershell(self, command):
        if not self.confirm(f"The assistant wants to run this command:\n\n{command}\n\nAllow it?"):
            return "The user did not allow that command."
        self.log.append(("powershell", command))
        try:
            done = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True,
                                  text=True, timeout=60, creationflags=subprocess.CREATE_NO_WINDOW)
        except subprocess.TimeoutExpired:
            return "The command took longer than 60 seconds and was stopped."
        out = (done.stdout or "") + (("\nERROR: " + done.stderr) if done.stderr else "")
        return out.strip()[:4000] or f"Finished with exit code {done.returncode}."
