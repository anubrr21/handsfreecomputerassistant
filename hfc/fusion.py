from collections import deque

import numpy as np

from .filters import OneEuro
from .gestures import HandReader

PALM = [0, 5, 9, 13, 17]
SLIDERS = ("two", "pinky")
HOLDS = {"thumb": ("win+h", "VOICE TYPING"), "rock": ("win+tab", "SWITCH WINDOWS")}


class CursorFusion:
    def __init__(self, cfg, screen_w, screen_h, mouse, reader=None):
        self.cfg = cfg
        self.reader = reader or HandReader(cfg)
        self.reading = None
        self.screen = np.array([screen_w, screen_h], dtype=float)
        self.diag = float(np.linalg.norm(self.screen))
        self.mouse = mouse
        self.mode = "hand"
        self.paused = False
        self.dwell = False
        self.status = ""
        self.grabbing = False
        self.gaze = None
        self.box = None
        self.on_thumb = None

        self._palm_filter = OneEuro(min_cutoff=1.2, beta=12.0)
        self._gaze_filter = OneEuro(min_cutoff=0.4, beta=0.002)
        self._reset_hand()
        self._brow = False
        self._dwell_anchor = None
        self._dwell_since = None

    def _reset_hand(self):
        self._palm_filter.reset()
        self._palm_prev = None
        self._state = "idle"
        self._press_t = 0.0
        self._press_palm = None
        self._sticky_until = 0.0
        self._label = None
        self._label_t = 0.0
        self._fired = False
        self._acc = 0.0
        self._unfist_t = None
        self._aim = None
        self._history = deque()
        self.grabbing = False
        self.reader.reset()

    def set_reader(self, reader):
        self.release_all()
        self.reader = reader
        self._reset_hand()

    def release_all(self):
        if self._state == "drag":
            self.mouse.release("left")
        self._state = "idle"

    def toggle_pause(self):
        self.paused = not self.paused
        self.release_all()
        self.mouse.idle()

    def _place_box(self, palm):
        bw, bh = self.cfg.box_size
        lx0, ly0, lx1, ly1 = self.cfg.box_limits
        u = np.clip(np.array(self.mouse.position(), dtype=float) / (self.screen - 1), 0.0, 1.0)
        x0 = float(np.clip(palm[0] - u[0] * bw, lx0, lx1 - bw))
        y0 = float(np.clip(palm[1] - u[1] * bh, ly0, ly1 - bh))
        self.box = (x0, y0, x0 + bw, y0 + bh)

    def _to_screen(self, palm):
        x0, y0, x1, y1 = self.box
        u = (palm - (x0, y0)) / (x1 - x0, y1 - y0)
        return np.clip(u, 0.0, 1.0) * (self.screen - 1)

    def _steer(self, target):
        self.mouse.move(*target)
        self._aim = target

    def _aim_at(self, t):
        best = None
        for when, aim in self._history:
            if when > t and best is not None:
                break
            best = aim
        return best

    def update(self, t, gaze_px, hand, blend=None):
        if gaze_px is not None:
            self.gaze = self._gaze_filter(t, gaze_px)

        if self.paused:
            self.status = "PAUSED"
            return

        if hand is None:
            self.release_all()
            self._reset_hand()
            self.reading = None
            if self.mode == "hand":
                self.mouse.idle()
                self.status = "show your hand"
        else:
            self._hand_step(t, hand)

        if self.mode == "eyes":
            self._eyes_step(t, blend or {})

    def _hand_step(self, t, hand):
        cfg = self.cfg
        r = self.reading = self.reader.read(hand)
        palm = self._palm_filter(t, hand.pts[PALM].mean(axis=0))
        if self._palm_prev is None:
            self._place_box(palm)
        delta = np.zeros(2) if self._palm_prev is None else palm - self._palm_prev
        self._palm_prev = palm.copy()
        target = self._to_screen(palm)
        steer = self.mode == "hand"
        while self._history and t - self._history[0][0] > 0.6:
            self._history.popleft()

        if r.label != self._label:
            self._label, self._label_t, self._fired, self._acc = r.label, t, False, 0.0
        held = t - self._label_t

        if r.label == "fist":
            self._unfist_t = None
            if held >= cfg.grab_hold_s:
                self.grabbing = True
        elif self.grabbing:
            if self._unfist_t is None:
                self._unfist_t = t
            elif t - self._unfist_t >= cfg.grab_release_s:
                self.grabbing = False

        if self._state == "idle":
            if self.grabbing:
                self._state = "pending"
                self._press_t = t
                self._press_palm = palm.copy()
                before = self._aim_at(t - cfg.rewind_s)
                if steer and before is not None:
                    self._steer(before)
                self.status = "GRAB"
            elif r.label in SLIDERS and held >= cfg.slider_hold_s:
                self._slide(r.label, delta[1])
            elif r.label in HOLDS:
                combo, name = HOLDS[r.label]
                if not self._fired and held >= cfg.action_hold_s:
                    if r.label == "thumb" and self.on_thumb is not None:
                        self.on_thumb()
                        name = "LISTENING"
                    else:
                        self.mouse.hotkey(combo)
                    self._fired = True
                    self.status = name
                elif not self._fired:
                    self.status = "HOLD..."
            elif r.label != "move":
                pass
            elif t < self._sticky_until and np.linalg.norm(palm - self._press_palm) < cfg.drag_start:
                pass
            else:
                self._sticky_until = 0.0
                if steer:
                    self._steer(target)
                self.status = "MOVE"

        elif self._state == "pending":
            if not self.grabbing:
                self.mouse.click("left")
                self._state = "idle"
                self._sticky_until = t + cfg.sticky_s
                self.status = "CLICK"
            elif (
                t - self._press_t >= cfg.drag_delay_s
                and np.linalg.norm(palm - self._press_palm) > cfg.drag_start
            ):
                self.mouse.press("left")
                self._state = "drag"
                self.status = "DRAG"
            elif t - self._press_t >= cfg.long_press_s:
                self.mouse.click("right")
                self._state = "spent"
                self.status = "RIGHT CLICK"

        elif self._state == "drag":
            if not self.grabbing:
                self.mouse.release("left")
                self._state = "idle"
                self.status = "MOVE"
            else:
                if steer:
                    self._steer(target)
                self.status = "DRAG"

        elif self._state == "spent" and not self.grabbing:
            self._state = "idle"

        if self._aim is not None:
            self._history.append((t, self._aim))

    def _slide(self, label, dy):
        if label == "two":
            self._acc += dy * self.cfg.scroll_gain
            self.status = "SCROLL"
        else:
            self._acc -= dy * self.cfg.volume_gain
            self.status = "VOLUME"
        steps = int(self._acc)
        if steps:
            (self.mouse.scroll if label == "two" else self.mouse.volume)(steps)
            self._acc -= steps

    def _eyes_step(self, t, blend):
        cfg = self.cfg
        if self.gaze is None:
            self.status = "eyes: calibrate first (Ctrl+Alt+C)"
            return
        if self._state in ("idle", "drag") and t >= self._sticky_until:
            self.mouse.move(*np.clip(self.gaze, 0, self.screen - 1))
        if self._state == "idle":
            self.status = "EYES"

        brow = blend.get("browInnerUp", 0.0)
        if not self._brow and brow > cfg.brow_on:
            self._brow = True
            self.mouse.click("left")
        elif self._brow and brow < cfg.brow_off:
            self._brow = False

        if not self.dwell:
            return
        if (
            self._dwell_anchor is None
            or np.linalg.norm(self.gaze - self._dwell_anchor) > cfg.dwell_radius * self.diag
        ):
            self._dwell_anchor, self._dwell_since = self.gaze.copy(), t
        elif self._dwell_since is not None and t - self._dwell_since >= cfg.dwell_s:
            self.mouse.click("left")
            self._dwell_since = None
