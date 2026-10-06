import re
import time

from .skills import APP_ALIASES, KNOWN_FOLDERS, SHORTCUTS, SITES

SAFE_UNADDRESSED = {"open", "youtube", "spotify", "web_search", "time", "battery", "volume", "media", "app"}

POLITE = re.compile(r"^(?:hey |ok |okay |so |um |uh )*(?:please |can you |could you |would you |will you |"
                    r"i want you to |i want to |i need you to |just |kindly )*", re.I)
NUMBER_WORDS = {"zero": 0, "ten": 10, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
                "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "full": 100, "max": 100,
                "maximum": 100, "half": 50}
SPOKEN_KEYS = {"control": "ctrl", "windows": "win", "window": "win", "escape": "esc", "return": "enter",
               "page up": "pageup", "page down": "pagedown", "print screen": "printscreen", "del": "delete"}
PHRASES = {
    "close this window": "close window", "close the window": "close window", "close it": "close window",
    "close this": "close window", "close this tab": "close tab", "close the tab": "close tab",
    "open a new tab": "new tab", "open new tab": "new tab", "minimise": "minimize",
    "minimize this": "minimize", "minimize window": "minimize", "minimize the window": "minimize",
    "maximise": "maximize", "maximize this": "maximize", "maximize window": "maximize",
    "maximize the window": "maximize", "take a screenshot": "screenshot", "take screenshot": "screenshot",
    "capture the screen": "screenshot", "switch windows": "switch window", "switch app": "switch window",
    "switch apps": "switch window", "go to desktop": "show desktop", "show the desktop": "show desktop",
    "back": "go back", "forward": "go forward", "reload": "refresh", "refresh the page": "refresh",
    "reload the page": "refresh", "save this": "save", "save the file": "save", "save it": "save",
    "copy this": "copy", "copy that": "copy", "paste it": "paste", "paste here": "paste",
    "undo that": "undo", "select everything": "select all", "fullscreen": "full screen",
    "press enter": "enter", "hit enter": "enter", "open start menu": "start menu", "open start": "start menu",
    "show all windows": "task view", "show windows": "task view",
}


def clean(text):
    original = text.strip().strip(".!?,").strip()
    original = POLITE.sub("", original)
    original = re.sub(r"[ ,]*(?:please|for me|right now|now)$", "", original, flags=re.I).strip()
    return original.lower(), original


def parse(text):
    found = _parse(text)
    words = text.split()
    for skip in range(1, min(4, len(words))):
        if found is not None:
            break
        found = _parse(" ".join(words[skip:]))
        if found is not None and found[0] in ("open", "youtube", "web_search", "type_text"):
            found = None
    return found


def split_steps(text):
    parts = [p.strip() for p in re.split(r"\b(?:and then|after that|then|and)\b|[,;]", text, flags=re.I) if p.strip()]
    if len(parts) < 2 or len(parts) > 4:
        return None
    steps = [_parse(p) for p in parts]
    if not all(steps):
        return None
    merged = []
    for step in steps:
        prev = merged[-1] if merged else None
        if prev and prev[0] == "open" and step[0] in ("youtube", "web_search") and prev[1][0] in ("spotify", "youtube"):
            query = step[1][0]
            merged[-1] = ("spotify", (query,)) if prev[1][0] == "spotify" else ("youtube", (query, True))
        else:
            merged.append(step)
    return merged


def _parse(text):
    t, original = clean(text)
    if not t:
        return None
    t = PHRASES.get(t, t)

    if re.fullmatch(r"(?:pause|stop|disable) (?:the )?(?:hand|hands|gesture|cursor)(?:s)? ?(?:control|tracking)?", t):
        return "app", ("pause",)
    if re.fullmatch(r"(?:resume|start|enable|continue) (?:the )?(?:hand|hands|gesture|cursor)(?:s)? ?(?:control|tracking)?", t):
        return "app", ("resume",)
    if re.fullmatch(r"(?:show|open) (?:me )?(?:the )?(?:gestures?|help|gesture (?:sheet|list))|what can you do|help", t):
        return "app", ("help",)

    if re.search(r"\bwhat(?:'s| is)? (?:the )?(?:time|date|day)\b|\bwhat time is it\b|\bwhat day is it\b|"
                 r"\btoday'?s date\b", t):
        return "time", ()
    if re.search(r"\bbattery\b", t) and not t.startswith(("search", "google")):
        return "battery", ()

    m = re.fullmatch(r"(?:set |turn |change |make )?(?:the )?volume (?:to |at )?(\d{1,3}|[a-z]+)(?: ?percent| ?%)?", t)
    if m and (m.group(1).isdigit() or m.group(1) in NUMBER_WORDS):
        return "volume", (None, int(m.group(1)) if m.group(1).isdigit() else NUMBER_WORDS[m.group(1)])
    if re.fullmatch(r"(?:turn (?:the )?)?volume up|(?:turn it |make it )?louder|(?:increase|raise) (?:the )?volume|turn up (?:the )?volume|turn (?:the )?volume up", t):
        return "volume", (5, None)
    if re.fullmatch(r"(?:turn (?:the )?)?volume down|(?:turn it |make it )?quieter|(?:decrease|lower|reduce) (?:the )?volume|turn down (?:the )?volume|turn (?:the )?volume down", t):
        return "volume", (-5, None)
    if re.fullmatch(r"(?:un)?mute(?: (?:the )?(?:sound|volume|audio|laptop|computer))?", t):
        return "media", ("mute",)
    if re.fullmatch(r"(?:pause|stop)(?: (?:the |this )?(?:music|song|video|track|playback))?", t):
        return "media", ("pause",)
    if re.fullmatch(r"(?:play|resume)(?: (?:the |this )?(?:music|song|video|track|playback))?", t):
        return "media", ("play",)
    if re.fullmatch(r"(?:play (?:the )?)?next(?: (?:song|track|video|one))?|skip(?: (?:this )?(?:song|track))?", t):
        return "media", ("next",)
    if re.fullmatch(r"(?:play (?:the )?)?previous(?: (?:song|track|video|one))?|last (?:song|track)", t):
        return "media", ("previous",)

    m = re.fullmatch(r"(?:(?:open|launch|go to|start|on|in) )?(youtube|spotify|chrome|google|(?:the |my )?browser)"
                     r"(?: and| then|,)+ (?:search(?: for)?|play|find|look up|look for) (.+)", t)
    if m:
        where, what = m.group(1), re.sub(r"^(?:the song |the video |the |song |some )", "", m.group(2))
        if where == "youtube":
            return "youtube", (what, False)
        if where == "spotify":
            return "spotify", (what,)
        return "web_search", (what,)

    m = re.fullmatch(r"(?:play|search(?: for)?|find|look up|put on) (.+?) (?:on|in|from) youtube", t) or \
        re.fullmatch(r"(?:search )?youtube (?:for )?(.+)", t)
    if m:
        return "youtube", (m.group(1), t.startswith(("play", "put on")))
    m = re.fullmatch(r"(?:play|search(?: for)?|find|look up|put on) (.+?) (?:on|in|from) spotify", t) or \
        re.fullmatch(r"(?:search )?spotify (?:for )?(.+)", t)
    if m:
        return "spotify", (re.sub(r"^(?:and |then |to |play |the song |song )+", "", m.group(1)),)
    m = re.fullmatch(r"(?:play|put on) (?:the song |the video |some )?(.+)", t)
    if m:
        return "youtube", (m.group(1), True)

    m = re.match(r"(?:type|write|dictate|enter the text)[ :]+(.+)", original, flags=re.I | re.S)
    if m:
        return "type_text", (m.group(1).strip(),)
    if t in SHORTCUTS:
        return "shortcut", (t,)
    m = re.fullmatch(r"(?:press|hit|push) (.+)", t)
    if m:
        spoken = m.group(1)
        for words, key in SPOKEN_KEYS.items():
            spoken = spoken.replace(words, key)
        keys = [k for k in re.split(r"[ +]+|\band\b|\bplus\b", spoken) if k and k not in ("key", "the", "button")]
        if keys and len(keys) <= 4:
            return "press", ("+".join(keys),)
    m = re.fullmatch(r"(double|right|left)?[ -]?click(?: (?:here|it|this|that))?", t)
    if m:
        return "click", (m.group(1) or "left",)
    m = re.fullmatch(r"scroll (up|down)(?: (?:a )?(?:bit|little|more|page))?|page (up|down)", t)
    if m:
        return "scroll", (m.group(1) or m.group(2),)

    if re.fullmatch(r"lock(?: (?:the|my|this))?(?: (?:computer|laptop|screen|pc))?", t):
        return "lock", ()
    if re.fullmatch(r"(?:shut ?down|turn off|power off|switch off)(?: (?:the|my|this))?(?: (?:computer|laptop|pc|system))?", t):
        return "power", ("shutdown",)
    if re.fullmatch(r"(?:restart|reboot)(?: (?:the|my|this))?(?: (?:computer|laptop|pc|system))?", t):
        return "power", ("restart",)
    if re.fullmatch(r"(?:go to sleep|sleep|put (?:the |my )?(?:computer|laptop|pc) to sleep)", t):
        return "power", ("sleep",)

    m = re.fullmatch(r"(?:search(?: the)?(?: web| internet| online| google)?(?: for)?|google|look up|"
                     r"look for|find out|search up) (.+)", t)
    if m:
        return "web_search", (m.group(1),)

    m = re.fullmatch(r"(?:open|launch|start|run|go to|show(?: me)?|bring up|take me to) (?:up )?(.+)", t)
    if m:
        target = re.sub(r"^(?:the|my|a|an) ", "", m.group(1)).strip()
        kind = "any"
        if re.search(r"\b(?:folder|directory)\b", target):
            kind = "folder"
        elif re.search(r"\b(?:file|document|pdf|photo|picture|image|video)\b", target):
            kind = "file"
        target = re.sub(r"\b(?:app|application|program|website|site|web page|page|folder|directory|file)\b", "", target)
        target = re.sub(r"\s+", " ", target).strip()
        if target:
            return "open", (target, kind)
    return None


class Assistant:

    def __init__(self, laptop, brain=None, app_control=None):
        self.laptop = laptop
        self.brain = brain
        self.app_control = app_control or (lambda action: None)

    def is_clear_command(self, text):
        parsed = parse(text)
        if parsed is None or parsed[0] not in SAFE_UNADDRESSED:
            return False
        if parsed[0] == "open":
            target, kind = parsed[1]
            return kind == "any" and self._known_name(target) is not None
        return True

    def _known_name(self, target):
        if target in SITES or target in KNOWN_FOLDERS or target in APP_ALIASES:
            return target
        for name in self.laptop.apps:
            if target == name.lower():
                return name
        return None

    def _keyword_guess(self, text):
        words = set(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())
        names = {n for n in list(SITES) + list(APP_ALIASES) if " " not in n and len(n) > 3 and n in words}
        names |= {n.lower() for n in self.laptop.apps if " " not in n and len(n) > 3 and n.lower() in words}
        return names.pop() if len(names) == 1 else None

    def handle(self, text):
        steps = split_steps(text)
        if steps:
            replies = []
            for i, step in enumerate(steps):
                if i:
                    time.sleep(1.5)
                replies.append(self._run(*step) or "I couldn't do one step.")
            return " ".join(replies)
        parsed = parse(text)
        reply = self._run(*parsed) if parsed else None
        if reply is not None:
            return reply
        if self.brain is not None:
            return self.brain.ask(text)
        guess = self._keyword_guess(text)
        if guess is not None:
            return self._open(guess, "any")
        if parsed and parsed[0] == "open":
            return f"I couldn't find {parsed[1][0]} on this laptop."
        return "Sorry, I didn't catch a command in that."

    def _run(self, action, args):
        lap = self.laptop
        if action == "app":
            self.app_control(args[0])
            return {"pause": "Hand control paused.", "resume": "Hand control is on.",
                    "help": "Here are the gestures."}[args[0]]
        if action == "open":
            return self._open(*args)
        if action == "volume":
            return lap.volume(change=args[0], level=args[1])
        simple = {
            "time": lap.time, "battery": lap.battery, "media": lap.media, "youtube": lap.youtube,
            "spotify": lap.spotify, "web_search": lap.web_search, "type_text": lap.type_text,
            "shortcut": lap.shortcut, "press": lap.press, "click": lap.click, "scroll": lap.scroll,
            "lock": lap.lock, "power": lap.power,
        }
        return simple[action](*args)

    def _open(self, target, kind):
        lap = self.laptop
        if kind == "any":
            exact_app = any(target == name.lower() for name in lap.apps)
            if target in SITES and not exact_app:
                return lap.open_site(target)
            return lap.open_app(target) or lap.open_site(target) or lap.open_path(target)
        return lap.open_path(target, kind)
