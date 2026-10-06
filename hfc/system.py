import ctypes
import math
import threading
import time
import winsound
from ctypes import wintypes

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass
ctypes.windll.winmm.timeBeginPeriod(1)

from pynput import keyboard as kb
from pynput.mouse import Button, Controller

BUTTONS = {"left": Button.left, "right": Button.right}
KEYS = {"alt": kb.Key.alt, "ctrl": kb.Key.ctrl, "win": kb.Key.cmd, "shift": kb.Key.shift,
        "left": kb.Key.left, "right": kb.Key.right, "tab": kb.Key.tab}
_user32 = ctypes.windll.user32


def screen_size():
    return _user32.GetSystemMetrics(0), _user32.GetSystemMetrics(1)


def beep(freq, ms=25):
    threading.Thread(target=winsound.Beep, args=(freq, ms), daemon=True).start()


class Mouse:

    LEAD = 1.0
    TAU = 0.02

    RATE = 120

    def __init__(self, sound=True):
        self._m = Controller()
        self._kb = kb.Controller()
        self._sound = sound
        self._target = None
        self._pos = None
        self._last_move = None
        self._interval = 1 / 30
        self._vel = (0.0, 0.0)
        self._lock = threading.Lock()
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        dt = 1.0 / self.RATE
        while self._running:
            with self._lock:
                if self._target is not None:
                    alpha = 1.0 - math.exp(-dt / self.TAU)
                    if self._pos is None:
                        self._pos = list(self.position())
                    age = time.perf_counter() - self._last_move
                    lead = self.LEAD * max(0.0, min(age, 2 * self._interval - age))
                    self._pos[0] += (self._target[0] + self._vel[0] * lead - self._pos[0]) * alpha
                    self._pos[1] += (self._target[1] + self._vel[1] * lead - self._pos[1]) * alpha
                    _user32.SetCursorPos(round(self._pos[0]), round(self._pos[1]))
            time.sleep(dt)

    def _snap(self):
        with self._lock:
            if self._target is not None:
                self._vel = (0.0, 0.0)
                self._pos = list(self._target)
                _user32.SetCursorPos(round(self._pos[0]), round(self._pos[1]))

    def position(self):
        pt = wintypes.POINT()
        _user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y

    def move(self, x, y):
        now = time.perf_counter()
        with self._lock:
            new = (float(x), float(y))
            gap = None if self._last_move is None else now - self._last_move
            if gap is not None and gap < 0.15:
                self._interval = gap
                self._vel = ((new[0] - self._target[0]) / gap, (new[1] - self._target[1]) / gap)
            else:
                self._vel = (0.0, 0.0)
            self._target = new
            self._last_move = now

    def idle(self):
        with self._lock:
            self._target = None
            self._pos = None
            self._last_move = None

    def press(self, button):
        self._snap()
        self._m.press(BUTTONS[button])

    def release(self, button):
        self._snap()
        self._m.release(BUTTONS[button])

    def click(self, button):
        self._snap()
        self._m.click(BUTTONS[button])
        if self._sound:
            beep(1200 if button == "left" else 700)

    def scroll(self, notches):
        self._m.scroll(0, notches)

    def volume(self, steps):
        key = kb.Key.media_volume_up if steps > 0 else kb.Key.media_volume_down
        for _ in range(abs(steps)):
            self._kb.tap(key)

    def hotkey(self, combo):
        keys = [KEYS.get(k, k) for k in combo.split("+")]
        for k in keys:
            self._kb.press(k)
        for k in reversed(keys):
            self._kb.release(k)
        if self._sound:
            beep(900, 40)

    def close(self):
        self._running = False


class DryMouse:

    def __init__(self, x=0, y=0):
        self.pos = (x, y)
        self.log = []

    def position(self):
        return self.pos

    def move(self, x, y):
        self.pos = (round(x), round(y))

    def idle(self):
        pass

    def press(self, button):
        self.log.append(("press", button))

    def release(self, button):
        self.log.append(("release", button))

    def click(self, button):
        self.log.append(("click", button))

    def scroll(self, notches):
        self.log.append(("scroll", notches))

    def volume(self, steps):
        self.log.append(("volume", steps))

    def hotkey(self, combo):
        self.log.append(("hotkey", combo))

    def close(self):
        pass
