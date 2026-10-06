import pickle
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WRIST, MIDDLE_MCP = 0, 9
CHAINS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
PARTS = ["thumb", *CHAINS]
TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}

TRAIN_POSES = ["open", "rest", "two", "fist", "thumb"]
POSE_FINGERS = {
    "open": dict(thumb=1, index=1, middle=1, ring=1, pinky=1),
    "rest": dict(thumb=1, index=1, middle=1, ring=1, pinky=1),
    "two": dict(thumb=0, index=1, middle=1, ring=0, pinky=0),
    "fist": dict(thumb=0, index=0, middle=0, ring=0, pinky=0),
    "thumb": dict(thumb=1, index=0, middle=0, ring=0, pinky=0),
    "pinch": dict(thumb=1, middle=1, ring=1, pinky=1),
}


@dataclass
class Reading:
    label: str
    conf: float
    fingers: str


def _angle(a, b):
    c = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def finger_features(world):
    w = world - world[WRIST]
    w = w / (np.linalg.norm(w[MIDDLE_MCP]) + 1e-9)
    out = {}
    for name, (m, p, d, t) in CHAINS.items():
        out[name] = np.array([
            _angle(w[m], w[p] - w[m]),
            _angle(w[p] - w[m], w[d] - w[p]),
            _angle(w[d] - w[p], w[t] - w[d]),
            np.linalg.norm(w[t]) / (np.linalg.norm(w[p]) + 1e-9) * 100,
        ])
    out["thumb"] = np.array([
        _angle(w[2] - w[1], w[3] - w[2]),
        _angle(w[3] - w[2], w[4] - w[3]),
        np.linalg.norm(w[4] - w[5]) * 100,
        np.linalg.norm(w[4] - w[17]) * 100,
    ])
    return out


def rule_straight(feats):
    def sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    p = {name: float(sig((115.0 - feats[name][:3].sum()) / 8.0)) for name in CHAINS}
    p["thumb"] = float(sig((feats["thumb"][3] - 80.0) / 8.0))
    return p


class FingerModel:

    def __init__(self):
        self.clf = {}
        self.accuracy = {}

    @staticmethod
    def _make():
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))

    def fit(self, landmarks, poses, groups):
        feats = [finger_features(lm) for lm in landmarks]
        poses, groups = np.asarray(poses), np.asarray(groups)
        for part in PARTS:
            keep = np.array([part in POSE_FINGERS[p] for p in poses])
            X = np.array([f[part] for f, k in zip(feats, keep) if k])
            y = np.array([POSE_FINGERS[p][part] for p in poses[keep]])
            hits = np.zeros(len(y), dtype=bool)
            for tr, te in StratifiedGroupKFold(n_splits=4).split(X, y, groups[keep]):
                hits[te] = self._make().fit(X[tr], y[tr]).predict(X[te]) == y[te]
            self.accuracy[part] = float(hits.mean())
            self.clf[part] = self._make().fit(X, y)
        return self

    def straight(self, feats):
        return {part: float(self.clf[part].predict_proba(feats[part][None])[0, 1]) for part in PARTS}

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path):
        if not path.exists():
            return None
        with open(path, "rb") as fh:
            return pickle.load(fh)


class HandReader:

    def __init__(self, cfg, model=None):
        self.cfg = cfg
        self.model = model
        self.trained = model is not None
        self.reset()

    def reset(self):
        self._p = None
        self.straight = dict.fromkeys(PARTS, True)

    def read(self, hand):
        feats = finger_features(hand.world)
        p = self.model.straight(feats) if self.model else rule_straight(feats)
        if self._p is not None:
            k = self.cfg.finger_smoothing
            p = {part: k * p[part] + (1 - k) * self._p[part] for part in PARTS}
        self._p = p
        for part in PARTS:
            if self.straight[part]:
                self.straight[part] = p[part] > self.cfg.finger_off
            else:
                self.straight[part] = p[part] > self.cfg.finger_on
        s = self.straight
        conf = min(max(v, 1 - v) for v in p.values())
        letters = "".join(c if s[part] else "-" for c, part in zip("TIMRP", PARTS))
        return Reading(self._label(s, hand), conf, letters)

    @staticmethod
    def _label(s, hand):
        i, m, r, pk = s["index"], s["middle"], s["ring"], s["pinky"]
        if not (i or m or r or pk):
            size = np.linalg.norm(hand.pts[WRIST] - hand.pts[MIDDLE_MCP])
            up = hand.pts[4][1] < hand.pts[5][1] - 0.5 * size
            return "thumb" if s["thumb"] and up else "fist"
        if i and m and not r:
            return "two"
        if pk and not i and not m and not r:
            return "pinky"
        if i and pk and not m and not r:
            return "rock"
        return "move"
