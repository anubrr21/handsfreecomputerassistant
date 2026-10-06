import math
import pickle

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

EYES = ((33, 133, 468), (263, 362, 473))
NOSE_TIP = 1
LOOK_SHAPES = [
    "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft",
    "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeLookDownRight",
]


def _eye(pts, outer, inner, iris):
    axis = pts[inner] - pts[outer]
    width = np.linalg.norm(axis) + 1e-9
    u = axis / width
    n = np.array([-u[1], u[0]])
    d = pts[iris] - pts[outer]
    return d @ u / width, d @ n / width


def _euler(m):
    r = m[:3, :3]
    sy = math.hypot(r[0, 0], r[1, 0])
    return (
        math.atan2(-r[2, 0], sy),
        math.atan2(r[2, 1], r[2, 2]),
        math.atan2(r[1, 0], r[0, 0]),
    )


def features(face, frame_w, frame_h):
    f = []
    for outer, inner, iris in EYES:
        f.extend(_eye(face.pts, outer, inner, iris))
    f.extend(face.blend.get(name, 0.0) for name in LOOK_SHAPES)
    f.extend(_euler(face.pose))
    f.extend(face.pose[:3, 3] / 50.0)
    nose = face.pts[NOSE_TIP]
    f.extend((nose[0] / frame_w, nose[1] / frame_h))
    return np.array(f, dtype=float)


def is_blinking(face, threshold):
    b = face.blend
    return max(b.get("eyeBlinkLeft", 0.0), b.get("eyeBlinkRight", 0.0)) > threshold


class GazeModel:

    CANDIDATES = [(d, a) for d in (1, 2) for a in (1.0, 10.0, 100.0, 1000.0)]

    def __init__(self, screen_w, screen_h):
        self.screen = np.array([screen_w, screen_h], dtype=float)
        self.pipe = None
        self.cv_error_px = None
        self.choice = None

    @staticmethod
    def _make(degree, alpha):
        return make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree, include_bias=False),
            StandardScaler(),
            Ridge(alpha=alpha),
        )

    def fit(self, X, Y_px, groups):
        Y = Y_px / self.screen
        best = None
        for degree, alpha in self.CANDIDATES:
            errs = []
            for tr, te in LeaveOneGroupOut().split(X, Y, groups):
                pred = self._make(degree, alpha).fit(X[tr], Y[tr]).predict(X[te])
                errs.append(np.linalg.norm((pred - Y[te]) * self.screen, axis=1).mean())
            err = float(np.mean(errs))
            if best is None or err < best[0]:
                best = (err, degree, alpha)
        self.cv_error_px, degree, alpha = best
        self.choice = (degree, alpha)
        self.pipe = self._make(degree, alpha).fit(X, Y)
        return self

    def predict(self, feat):
        xy = self.pipe.predict(feat[None])[0] * self.screen
        return np.clip(xy, 0, self.screen - 1)

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path, screen_w, screen_h):
        if not path.exists():
            return None
        with open(path, "rb") as fh:
            model = pickle.load(fh)
        if tuple(model.screen) != (screen_w, screen_h):
            return None
        return model
