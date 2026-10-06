import time

import cv2
import numpy as np

from .config import FINGER_DATA_PATH
from .gestures import TRAIN_POSES as CLASSES
from .gestures import FingerModel

WINDOW = "Hands Free Computer - gesture training"
BLOCK_S = 1.0
MIN_SAMPLES = 60
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
    (18, 19), (19, 20), (0, 17),
]
WHITE, GREEN, AMBER, GREY = (235, 235, 235), (60, 220, 60), (60, 170, 255), (150, 150, 150)

GUIDE = {
    "open": ("OPEN HAND",
             ["Spread all five fingers wide.", "Palm facing the screen."],
             "Used for: moving the cursor"),
    "rest": ("LAZY HAND",
             ["Let your hand go loose: fingers slightly bent,", "not touching each other."],
             "Used for: moving the cursor (this is how a tired hand looks)"),
    "two": ("TWO FINGERS",
            ["Index and middle fingers straight up.", "Ring and little finger folded down, thumb tucked in."],
            "Used for: scrolling"),
    "fist": ("FIST",
             ["Close all your fingers into your palm."],
             "Used for: click, drag, right click"),
    "thumb": ("THUMBS UP",
              ["Make a fist, then point your thumb straight up."],
              "Used for: starting voice typing"),
}

CHEAT_SHEET = [
    ("open", "OPEN HAND", ["Move the cursor"]),
    ("point", "POINT", ["Also moves the cursor"]),
    ("fist", "FIST", ["Close + open = click", "Close + move = drag", "Close + hold = right click"]),
    ("two", "TWO FINGERS", ["Move hand up / down", "to scroll"]),
    ("pinky", "LITTLE FINGER", ["Move hand up / down", "to change volume"]),
    ("rock", "ROCK SIGN", ["Hold: switch windows"]),
    ("thumb", "THUMBS UP", ["Hold: voice typing"]),
]

current_pose = None


_MCP = {"index": (0.37, 0.56), "middle": (0.48, 0.53), "ring": (0.58, 0.55), "pinky": (0.67, 0.61)}
_DIR = {"index": (-0.12, -1.0), "middle": (0.0, -1.0), "ring": (0.10, -1.0), "pinky": (0.24, -1.0)}
_LEN = {"index": 0.34, "middle": 0.38, "ring": 0.34, "pinky": 0.27}
_WRIST = (0.50, 0.95)


def _finger(name, shape):
    m = np.array(_MCP[name])
    d = np.array(_DIR[name])
    d = d / np.linalg.norm(d)
    length = _LEN[name]
    if shape == "straight":
        return [m + d * length * k for k in (0.45, 0.75, 1.0)]
    if shape == "bent":
        side = np.array([0.03, 0.0])
        return [m + d * length * 0.42, m + d * length * 0.62 + side, m + d * length * 0.66 + 2 * side]
    return [m + d * length * 0.22, m + d * length * 0.10 + (0.02, 0.03), m + (0.01, 0.07)]


_THUMBS = {
    "out": [(0.37, 0.83), (0.25, 0.73), (0.17, 0.63), (0.11, 0.53)],
    "loose": [(0.37, 0.83), (0.27, 0.74), (0.21, 0.66), (0.18, 0.58)],
    "tucked": [(0.37, 0.83), (0.33, 0.73), (0.40, 0.68), (0.47, 0.66)],
    "up": [(0.37, 0.83), (0.30, 0.68), (0.27, 0.52), (0.25, 0.36)],
    "pinch": [(0.37, 0.83), (0.27, 0.71), (0.22, 0.57), (0.235, 0.44)],
}
_POSES = {
    "open": ("out", "straight", "straight", "straight", "straight"),
    "rest": ("loose", "bent", "bent", "bent", "bent"),
    "point": ("tucked", "straight", "folded", "folded", "folded"),
    "pinky": ("tucked", "folded", "folded", "folded", "straight"),
    "rock": ("tucked", "straight", "folded", "folded", "straight"),
    "two": ("tucked", "straight", "straight", "folded", "folded"),
    "fist": ("tucked", "folded", "folded", "folded", "folded"),
    "thumb": ("up", "folded", "folded", "folded", "folded"),
}


def illustration(name, size=420):
    img = np.full((size, size, 3), 38, np.uint8)
    skin, edge = (150, 185, 235), (70, 95, 140)
    thumb, *fingers = _POSES[name]

    def px(p):
        return int(p[0] * size), int(p[1] * size)

    chains = [[np.array(p) for p in _THUMBS[thumb]]]
    for fname, shape in zip(_MCP, fingers):
        pts = _finger(fname, shape)
        chains.append([np.array(_MCP[fname])] + [np.array(p) for p in pts])

    palm = np.array([px(p) for p in (_WRIST, (0.37, 0.83), *_MCP.values(), (0.66, 0.85))])
    for colour, extra in ((edge, 6), (skin, 0)):
        if extra:
            cv2.polylines(img, [palm], True, colour, extra * 2, cv2.LINE_AA)
        cv2.fillPoly(img, [palm], colour, cv2.LINE_AA)
        for i, chain in enumerate(chains):
            thick = int(size * (0.075 if i == 0 else 0.062)) + extra
            for a, b in zip(chain, chain[1:]):
                cv2.line(img, px(a), px(b), colour, thick, cv2.LINE_AA)
    return img


def cheat_sheet(tile=230):
    cols = len(CHEAT_SHEET)
    sheet = np.full((tile + 120, tile * cols, 3), 38, np.uint8)
    for i, (pose, name, lines) in enumerate(CHEAT_SHEET):
        x = i * tile
        sheet[:tile, x:x + tile] = illustration(pose, tile)
        _put(sheet, name, x + 10, tile + 26, 0.6, WHITE)
        for j, line in enumerate(lines):
            _put(sheet, line, x + 10, tile + 52 + j * 22, 0.45, GREY, 1)
    return sheet


def _put(img, msg, x, y, scale=0.8, colour=WHITE, thick=2):
    cv2.putText(img, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def _centre(img, msg, y, scale=1.0, colour=WHITE):
    w = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0]
    _put(img, msg, (img.shape[1] - w) // 2, y, scale, colour)


def _camera_view(percept, height):
    view = percept.frame.copy()
    h, w = view.shape[:2]
    if percept.hand is not None:
        pts = (percept.hand.pts * (w, h)).astype(int)
        for a, b in HAND_EDGES:
            cv2.line(view, tuple(pts[a]), tuple(pts[b]), (80, 220, 80), 2, cv2.LINE_AA)
    return cv2.resize(view, (int(w * height / h), height))


def _pose_screen(canvas, percept, name, step, footer, footer_colour, progress=None):
    canvas[:] = 0
    sh, sw = canvas.shape[:2]
    title, how, use = GUIDE[name]
    _centre(canvas, f"Pose {step} of {len(CLASSES)}:  {title}", int(sh * 0.09), 1.4)
    for i, line in enumerate(how):
        _centre(canvas, line, int(sh * 0.16) + i * 40, 0.9)
    _centre(canvas, use, int(sh * 0.16) + len(how) * 40 + 8, 0.75, GREY)

    size = int(sh * 0.46)
    top = int(sh * 0.34)
    pic = illustration(name, size)
    cam = _camera_view(percept, size)
    gap = 60
    x0 = (sw - size - gap - cam.shape[1]) // 2
    canvas[top:top + size, x0:x0 + size] = pic
    cx = x0 + size + gap
    canvas[top:top + size, cx:cx + cam.shape[1]] = cam
    _put(canvas, "Make it look like this", x0, top - 12, 0.7, GREY)
    _put(canvas, "Your hand (camera)", cx, top - 12, 0.7, GREY)

    if progress is not None:
        y = top + size + 26
        cv2.rectangle(canvas, (x0, y), (cx + cam.shape[1], y + 18), (70, 70, 70), -1)
        cv2.rectangle(canvas, (x0, y), (x0 + int((cx + cam.shape[1] - x0) * progress), y + 18), GREEN, -1)
    _centre(canvas, footer, int(sh * 0.93), 1.0, footer_colour)
    cv2.imshow(WINDOW, canvas)


def _message(canvas, lines):
    canvas[:] = 0
    sh = canvas.shape[0]
    for i, (msg, scale, colour) in enumerate(lines):
        _centre(canvas, msg, int(sh * 0.40) + i * 56, scale, colour)
    cv2.imshow(WINDOW, canvas)


def run(cam, perception, cfg, screen_w, screen_h):
    global current_pose
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    canvas = np.zeros((screen_h, screen_w, 3), np.uint8)
    seq = -1

    def frame():
        nonlocal seq
        seq, raw = cam.read(seq)
        return perception.process(raw, want_face=False)

    def key():
        return cv2.pollKey() & 0xFF

    try:
        while True:
            frame()
            _message(canvas, [
                ("Teach the computer your hand", 1.5, WHITE),
                (f"You will be shown {len(CLASSES)} hand poses, one at a time, each with a picture.", 0.9, WHITE),
                ("Nothing records until you press SPACE, so take your time.", 0.9, WHITE),
                ("Use the same hand for all of them.", 0.9, WHITE),
                ("SPACE = begin      ESC = cancel", 1.0, AMBER),
            ])
            k = key()
            if k == 27:
                return None
            if k == 32:
                break

        data = {}
        ci = 0
        while ci < len(CLASSES):
            name = current_pose = CLASSES[ci]
            while True:
                p = frame()
                ready = p.hand is not None
                _pose_screen(canvas, p, name, ci + 1,
                             "Hold the pose, then press SPACE to record" if ready
                             else "Show your hand to the camera", AMBER)
                k = key()
                if k == 27:
                    return None
                if k == 32 and ready:
                    break

            samples = []
            start = time.perf_counter()
            while (el := time.perf_counter() - start) < cfg.train_record_s:
                p = frame()
                if p.hand is not None:
                    samples.append((p.hand.world.copy(), int(el / BLOCK_S)))
                _pose_screen(canvas, p, name, ci + 1,
                             "RECORDING: keep the pose, move your hand around slowly, tilt it a little"
                             if p.hand is not None else "Hand not visible: bring it back into view",
                             GREEN if p.hand is not None else AMBER, progress=el / cfg.train_record_s)
                if key() == 27:
                    return None

            enough = len(samples) >= MIN_SAMPLES
            while True:
                p = frame()
                _pose_screen(canvas, p, name, ci + 1,
                             "Recorded.   SPACE = next pose      R = redo this pose" if enough
                             else "Hand was out of view too much.   R = redo this pose", GREEN if enough else AMBER)
                k = key()
                if k == 27:
                    return None
                if k in (ord("r"), ord("R")):
                    break
                if k == 32 and enough:
                    data[name] = samples
                    ci += 1
                    break

        current_pose = None
        X, y, groups = [], [], []
        for ci, name in enumerate(CLASSES):
            for feat, block in data[name]:
                X.append(feat)
                y.append(name)
                groups.append(ci * 1000 + block)
        X, y, groups = np.array(X), np.array(y), np.array(groups)
        _message(canvas, [("Training the network...", 1.2, WHITE)])
        cv2.pollKey()
        np.savez(FINGER_DATA_PATH, L=X, y=y, groups=groups)
        model = FingerModel().fit(X, y, groups)

        worst = min(model.accuracy.values())
        weak = [f for f, a in model.accuracy.items() if a < 0.95]
        lines = [(f"Done. It can tell straight from curled fingers {worst:.0%} of the time or better.", 1.1, GREEN)]
        if weak:
            lines.append(("Less reliable fingers: " + ", ".join(f"{f} {model.accuracy[f]:.0%}" for f in weak), 0.9, AMBER))
            lines.append(("Ctrl+Alt+T trains again.", 0.9, WHITE))
        lines.append(("Press SPACE to start using it.", 1.0, AMBER))
        while True:
            frame()
            _message(canvas, lines)
            if key() == 32:
                return model
    finally:
        current_pose = None
        cv2.destroyWindow(WINDOW)
