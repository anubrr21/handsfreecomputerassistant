import time

import cv2
import numpy as np

from .gaze import GazeModel, features, is_blinking

WINDOW = "Hands Free Computer - calibration"


def _targets(w, h):
    xs, ys = (0.05, 0.5, 0.95), (0.06, 0.5, 0.94)
    grid = [(x, y) for y in ys for x in xs]
    inner = [(0.275, 0.28), (0.725, 0.28), (0.275, 0.72), (0.725, 0.72)]
    pts = np.array(grid + inner) * (w, h)
    rng = np.random.default_rng(0)
    return pts[rng.permutation(len(pts))]


def _text(img, msg, y):
    size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    x = (img.shape[1] - size[0]) // 2
    cv2.putText(img, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)


def run(cam, perception, cfg, screen_w, screen_h):
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    canvas = np.zeros((screen_h, screen_w, 3), np.uint8)

    try:
        _text(canvas, "Look at each dot until it disappears. Keep your head roughly still.", screen_h // 2 - 30)
        _text(canvas, "Press SPACE to start, ESC to cancel.", screen_h // 2 + 30)
        cv2.imshow(WINDOW, canvas)
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                return None
            if key == 32:
                break

        X, Y, groups = [], [], []
        seq = -1
        targets = _targets(screen_w, screen_h)
        for gi, target in enumerate(targets):
            centre = tuple(int(v) for v in target)
            start = time.perf_counter()
            total = cfg.calib_settle_s + cfg.calib_sample_s
            while (elapsed := time.perf_counter() - start) < total:
                seq, frame = cam.read(seq)
                p = perception.process(frame)
                sampling = elapsed >= cfg.calib_settle_s
                if sampling and p.face is not None and not is_blinking(p.face, cfg.blink_threshold):
                    X.append(features(p.face, frame.shape[1], frame.shape[0]))
                    Y.append(target)
                    groups.append(gi)

                canvas[:] = 0
                radius = int(28 - 20 * min(elapsed / total, 1.0))
                colour = (60, 220, 60) if sampling else (60, 160, 255)
                cv2.circle(canvas, centre, radius, colour, -1, cv2.LINE_AA)
                cv2.circle(canvas, centre, 3, (255, 255, 255), -1, cv2.LINE_AA)
                if p.face is None:
                    _text(canvas, "Face not visible", screen_h - 60)
                cv2.imshow(WINDOW, canvas)
                if cv2.waitKey(1) & 0xFF == 27:
                    return None

        if len(set(groups)) < len(targets) * 0.8:
            canvas[:] = 0
            _text(canvas, "Too few samples: face was not visible. Press any key.", screen_h // 2)
            cv2.imshow(WINDOW, canvas)
            cv2.waitKey(0)
            return None

        model = GazeModel(screen_w, screen_h).fit(np.array(X), np.array(Y), np.array(groups))
        canvas[:] = 0
        _text(canvas, f"Calibrated. Expected gaze error: about {model.cv_error_px:.0f} px.", screen_h // 2 - 30)
        _text(canvas, "Press any key to continue.", screen_h // 2 + 30)
        cv2.imshow(WINDOW, canvas)
        cv2.waitKey(0)
        return model
    finally:
        cv2.destroyWindow(WINDOW)
