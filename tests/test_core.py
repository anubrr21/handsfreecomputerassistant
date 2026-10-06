import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hfc.config import Config
from hfc.filters import OneEuro
from hfc.fusion import CursorFusion
from hfc.gaze import GazeModel
from hfc.gestures import TRAIN_POSES, FingerModel, HandReader, finger_features, rule_straight
from hfc.perception import Hand
from hfc.system import DryMouse

W, H = 1920, 1080
CENTRE = (W // 2, H // 2)
FINGER_X = {"index": -0.02, "middle": 0.0, "ring": 0.02, "pinky": 0.04}
JOINTS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
POSES = {
    "open": (("index", "middle", "ring", "pinky"), True),
    "rest": (("index", "middle", "ring", "pinky"), True),
    "point": (("index",), False),
    "two": (("index", "middle"), False),
    "fist": ((), False),
    "thumb": ((), True),
    "pinky": (("pinky",), False),
    "rock": (("index", "pinky"), False),
}


def make_hand(x=0.5, y=0.46, pose="open", noise=0.0, rng=None):
    straight, thumb_out = POSES[pose]
    bend = 0.012 if pose == "rest" else 0.0
    world = np.zeros((21, 3))
    for name, (mcp, pip, dip, tip) in JOINTS.items():
        fx = FINGER_X[name]
        world[mcp] = (fx, -0.09, 0.0)
        if name in straight:
            world[pip] = (fx, -0.13, bend)
            world[dip] = (fx, -0.155, 2.2 * bend)
            world[tip] = (fx, -0.175, 3.6 * bend)
        else:
            world[pip] = (fx, -0.12, 0.025)
            world[dip] = (fx, -0.10, 0.045)
            world[tip] = (fx, -0.075, 0.035)
    if thumb_out:
        world[1:5] = [(-0.02, -0.02, 0), (-0.04, -0.04, 0), (-0.06, -0.06, 0), (-0.08, -0.08, 0)]
    else:
        world[1:5] = [(-0.02, -0.02, 0), (-0.03, -0.04, 0.01), (-0.01, -0.06, 0.02), (0.02, -0.07, 0.02)]
    if noise:
        world[1:] += rng.normal(scale=noise, size=(20, 3))

    pts = np.full((21, 2), (x, y), dtype=float)
    pts[0] = (x, y + 0.10)
    pts[9] = (x, y - 0.10)
    if pose == "thumb":
        pts[4] = (x, y - 0.25)
    return Hand(pts, world)


def run(fusion, frames, t0=0.0):
    t = t0
    for hand in frames:
        fusion.update(t, None, hand)
        t += 1 / 30
    return t


def trained_model():
    rng = np.random.default_rng(0)
    L, y, groups = [], [], []
    for ci, name in enumerate(TRAIN_POSES):
        for i in range(120):
            L.append(make_hand(pose=name, noise=0.003, rng=rng).world)
            y.append(name)
            groups.append(ci * 100 + i // 15)
    return FingerModel().fit(np.array(L), y, groups)


class GestureBehaviour:

    model = None

    def setUp(self):
        self.mouse = DryMouse(*CENTRE)
        cfg = Config()
        self.fusion = CursorFusion(cfg, W, H, self.mouse, HandReader(cfg, self.model))

    def test_open_hand_steers_cursor_across_the_whole_screen(self):
        t = run(self.fusion, [make_hand()] * 5)
        x0, y0, x1, y1 = self.fusion.box
        t = run(self.fusion, [make_hand(x0, y0)] * 60, t0=t)
        self.assertEqual(self.mouse.pos, (0, 0))
        run(self.fusion, [make_hand(x1, y1)] * 60, t0=t)
        self.assertGreaterEqual(self.mouse.pos, (W - 3, H - 3))

    def test_raising_the_hand_anywhere_does_not_move_the_cursor(self):
        for x, y in ((0.70, 0.60), (0.30, 0.30), (0.5, 0.5)):
            self.setUp()
            run(self.fusion, [make_hand(x, y)] * 20)
            self.assertLess(np.linalg.norm(np.array(self.mouse.pos) - CENTRE), 3, (x, y))

    def test_whole_screen_is_reachable_from_a_hand_held_low_and_right(self):
        t = run(self.fusion, [make_hand(0.80, 0.85)] * 5)
        x0, y0, x1, y1 = self.fusion.box
        self.assertGreaterEqual(x0, 0.06)
        self.assertLessEqual(y1, 0.92 + 1e-9)
        run(self.fusion, [make_hand(x0, y0)] * 60, t0=t)
        self.assertEqual(self.mouse.pos, (0, 0))

    def test_two_fingers_still_scroll_when_the_little_finger_sticks_out(self):
        hand = make_hand(pose="two")
        pinky_out = make_hand(pose="open").world[17:21]
        def frame(y):
            h = make_hand(y=y, pose="two")
            h.world[17:21] = pinky_out
            return h
        t = run(self.fusion, [make_hand()] * 10 + [frame(0.46)] * 10)
        run(self.fusion, [frame(0.46 - 0.01 * i) for i in range(20)], t0=t)
        self.assertTrue(self.mouse.log and all(kind == "scroll" for kind, _ in self.mouse.log))

    def test_pointing_and_relaxed_hands_also_steer_and_trigger_nothing(self):
        for pose in ("point", "rest"):
            self.setUp()
            run(self.fusion, [make_hand(x=0.3 + 0.005 * i, pose=pose) for i in range(60)])
            self.assertEqual(self.mouse.log, [], pose)
            self.assertNotEqual(self.mouse.pos, CENTRE, pose)

    def test_closing_and_opening_the_hand_is_one_left_click(self):
        run(self.fusion, [make_hand()] * 8 + [make_hand(pose="fist")] * 6 + [make_hand()] * 8)
        self.assertEqual(self.mouse.log, [("click", "left")])

    def test_single_misread_fist_frame_does_not_click(self):
        run(self.fusion, [make_hand()] * 8 + [make_hand(pose="fist")] + [make_hand()] * 8)
        self.assertEqual(self.mouse.log, [])

    def test_click_lands_where_the_cursor_was_before_the_hand_closed(self):
        t = run(self.fusion, [make_hand(x=0.4)] * 40)
        aimed = self.mouse.pos
        closing = [make_hand(x=0.4 + 0.004 * i, pose="fist") for i in range(6)]
        run(self.fusion, closing + [make_hand(x=0.424)] * 3 + [make_hand(x=0.4)] * 6, t0=t)
        self.assertEqual(self.mouse.log, [("click", "left")])
        self.assertLess(abs(self.mouse.pos[0] - aimed[0]), 12)

    def test_quick_fist_is_a_click_even_if_the_hand_slips_a_lot(self):
        t = run(self.fusion, [make_hand(x=0.4)] * 40)
        aimed = self.mouse.pos
        slip = [make_hand(x=0.4 + 0.02 * i, pose="fist") for i in range(6)]
        for hand in slip + [make_hand(x=0.52)] * 8:
            t = run(self.fusion, [hand], t0=t)
            if self.mouse.log:
                break
        self.assertEqual(self.mouse.log, [("click", "left")])
        self.assertLess(abs(self.mouse.pos[0] - aimed[0]), 12)

    def test_two_quick_fists_click_twice_at_the_same_spot(self):
        t = run(self.fusion, [make_hand()] * 30)
        before = self.mouse.pos
        tap = [make_hand(pose="fist")] * 5 + [make_hand(x=0.505)] * 5
        run(self.fusion, tap + tap, t0=t)
        self.assertEqual(self.mouse.log, [("click", "left")] * 2)
        self.assertEqual(self.mouse.pos, before)

    def test_fist_and_move_is_a_drag(self):
        frames = [make_hand()] * 8 + [make_hand(x=0.5 + 0.01 * i, pose="fist") for i in range(18)]
        run(self.fusion, frames + [make_hand(x=0.68)] * 6)
        self.assertEqual(self.mouse.log, [("press", "left"), ("release", "left")])
        self.assertGreater(self.mouse.pos[0], CENTRE[0] + 100)

    def test_one_misread_frame_does_not_drop_a_drag(self):
        drag = [make_hand(x=0.5 + 0.01 * i, pose="fist") for i in range(18)]
        drag[12] = make_hand(x=0.62, pose="open")
        run(self.fusion, [make_hand()] * 8 + drag + [make_hand(x=0.68)] * 6)
        self.assertEqual(self.mouse.log, [("press", "left"), ("release", "left")])

    def test_fist_held_still_is_one_right_click(self):
        run(self.fusion, [make_hand()] * 8 + [make_hand(pose="fist")] * 50 + [make_hand()] * 8)
        self.assertEqual(self.mouse.log, [("click", "right")])

    def test_slow_click_under_a_second_is_still_a_left_click(self):
        run(self.fusion, [make_hand()] * 8 + [make_hand(pose="fist")] * 24 + [make_hand()] * 8)
        self.assertEqual(self.mouse.log, [("click", "left")])

    def test_thumbs_up_calls_the_voice_assistant_when_one_is_attached(self):
        calls = []
        self.fusion.on_thumb = lambda: calls.append(1)
        run(self.fusion, [make_hand()] * 8 + [make_hand(pose="thumb")] * 60)
        self.assertEqual((calls, self.mouse.log), ([1], []))

    def test_losing_the_hand_mid_drag_releases_the_button(self):
        frames = [make_hand()] * 8 + [make_hand(x=0.5 + 0.01 * i, pose="fist") for i in range(14)]
        run(self.fusion, frames + [None])
        self.assertEqual(self.mouse.log, [("press", "left"), ("release", "left")])

    def test_two_fingers_scroll_and_do_not_move_cursor(self):
        t = run(self.fusion, [make_hand()] * 30 + [make_hand(pose="two")] * 10)
        before = self.mouse.pos
        run(self.fusion, [make_hand(y=0.46 - 0.01 * i, pose="two") for i in range(20)], t0=t)
        self.assertTrue(self.mouse.log and all(kind == "scroll" and n < 0 for kind, n in self.mouse.log))
        self.assertEqual(self.mouse.pos, before)

    def test_little_finger_up_changes_volume_with_hand_height(self):
        t = run(self.fusion, [make_hand()] * 10 + [make_hand(pose="pinky")] * 10)
        run(self.fusion, [make_hand(y=0.46 - 0.01 * i, pose="pinky") for i in range(20)], t0=t)
        self.assertTrue(self.mouse.log and all(kind == "volume" and n > 0 for kind, n in self.mouse.log))

    def test_brief_two_finger_misread_does_not_scroll(self):
        frames = [make_hand(y=0.46 - 0.01 * i, pose="two" if i % 6 == 0 else "open") for i in range(40)]
        run(self.fusion, frames)
        self.assertEqual(self.mouse.log, [])

    def test_thumbs_up_held_starts_voice_typing_once(self):
        run(self.fusion, [make_hand()] * 8 + [make_hand(pose="thumb")] * 60)
        self.assertEqual(self.mouse.log, [("hotkey", "win+h")])

    def test_rock_sign_held_switches_windows_once(self):
        run(self.fusion, [make_hand()] * 8 + [make_hand(pose="rock")] * 60)
        self.assertEqual(self.mouse.log, [("hotkey", "win+tab")])

    def test_brief_thumbs_up_or_rock_does_nothing(self):
        for pose in ("thumb", "rock"):
            self.setUp()
            run(self.fusion, [make_hand()] * 8 + [make_hand(pose=pose)] * 8 + [make_hand()] * 8)
            self.assertEqual(self.mouse.log, [], pose)

    def test_fist_with_thumb_resting_out_is_still_a_click(self):
        hand = make_hand(pose="thumb")
        hand.pts[4] = hand.pts[5]
        run(self.fusion, [make_hand()] * 8 + [hand] * 6 + [make_hand()] * 8)
        self.assertEqual(self.mouse.log, [("click", "left")])

    def test_paused_does_nothing(self):
        self.fusion.toggle_pause()
        run(self.fusion, [make_hand(x=0.3)] * 8 + [make_hand(x=0.3, pose="fist")] * 6 + [make_hand()] * 8)
        self.assertEqual(self.mouse.log, [])
        self.assertEqual(self.mouse.pos, CENTRE)

    def test_eyes_mode_follows_gaze_and_brow_clicks(self):
        self.fusion.mode = "eyes"
        gaze = np.array([400.0, 300.0])
        for i in range(20):
            self.fusion.update(i / 30, gaze, None, {"browInnerUp": 0.1})
        self.fusion.update(1.0, gaze, None, {"browInnerUp": 0.9})
        self.fusion.update(1.1, gaze, None, {"browInnerUp": 0.9})
        self.assertLess(np.linalg.norm(np.array(self.mouse.pos) - gaze), 5)
        self.assertEqual(self.mouse.log, [("click", "left")])


class FixedThresholdTests(GestureBehaviour, unittest.TestCase):
    pass


class TrainedClassifierTests(GestureBehaviour, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = trained_model()

    def test_every_finger_is_learned_accurately(self):
        self.assertGreater(min(self.model.accuracy.values()), 0.97)

    def test_gestures_never_recorded_are_still_read(self):
        cfg = Config()
        for pose in ("pinky", "rock", "point"):
            reader = HandReader(cfg, self.model)
            for _ in range(5):
                label = reader.read(make_hand(pose=pose)).label
            self.assertEqual(label, "move" if pose == "point" else pose)


class FingerFeatureTests(unittest.TestCase):
    def test_features_ignore_hand_rotation_scale_and_position(self):
        hand = make_hand(pose="two")
        a = np.radians(35)
        rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        moved = (hand.world @ rot.T) * 1.3 + 0.05
        fa, fb = finger_features(hand.world), finger_features(moved)
        for part in fa:
            np.testing.assert_allclose(fa[part], fb[part], atol=1e-6)

    def test_fixed_thresholds_read_straight_and_curled(self):
        p = rule_straight(finger_features(make_hand(pose="two").world))
        self.assertGreater(min(p["index"], p["middle"]), 0.9)
        self.assertLess(max(p["ring"], p["pinky"], p["thumb"]), 0.1)


class GazeModelTests(unittest.TestCase):
    def test_recovers_a_nonlinear_mapping_on_unseen_targets(self):
        rng = np.random.default_rng(1)
        targets = np.array([(x, y) for x in (0.05, 0.35, 0.65, 0.95) for y in (0.06, 0.5, 0.94)])
        X, Y, groups = [], [], []
        mix = rng.normal(size=(2, 20))
        for gi, target in enumerate(targets):
            for _ in range(25):
                z = target - 0.5
                feat = z @ mix + 0.3 * (z**2) @ mix + rng.normal(scale=0.01, size=20)
                X.append(feat)
                Y.append(target * (W, H))
                groups.append(gi)
        model = GazeModel(W, H).fit(np.array(X), np.array(Y), np.array(groups))
        self.assertLess(model.cv_error_px, 120)
        pred = model.predict(np.array(X[0]))
        self.assertLess(np.linalg.norm(pred - Y[0]), 120)


class FilterTests(unittest.TestCase):
    def test_one_euro_reduces_jitter(self):
        rng = np.random.default_rng(0)
        f = OneEuro(min_cutoff=0.4, beta=0.002)
        out = [f(i / 30, np.array([100.0, 100.0]) + rng.normal(scale=20, size=2)) for i in range(120)]
        self.assertLess(np.std(out[30:], axis=0).max(), 8)


if __name__ == "__main__":
    unittest.main()
