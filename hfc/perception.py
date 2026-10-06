import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .config import MODELS


class Camera:

    def __init__(self, index, width, height):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}")
        self._frame = None
        self._seq = 0
        self._cond = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def read(self, last_seq=-1, timeout=2.0):
        with self._cond:
            if not self._cond.wait_for(lambda: self._frame is not None and self._seq != last_seq, timeout):
                raise RuntimeError("Camera stopped delivering frames")
            return self._seq, self._frame

    def close(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


@dataclass
class Face:
    pts: np.ndarray
    blend: dict
    pose: np.ndarray


@dataclass
class Hand:
    pts: np.ndarray
    world: np.ndarray
    right: bool = True


@dataclass
class Percept:
    t: float
    frame: np.ndarray
    face: Face | None
    hand: Hand | None


def _base(name):
    return mp_python.BaseOptions(model_asset_buffer=(MODELS / name).read_bytes())


class Perception:
    def __init__(self):
        self.face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=_base("face_landmarker.task"),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
        )
        self.hand = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=_base("hand_landmarker.task"),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self._pool = ThreadPoolExecutor(max_workers=2)
        self._t0 = time.perf_counter()
        self._last_ms = -1

    def process(self, frame, want_face=True):
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        t = time.perf_counter()
        ms = max(int((t - self._t0) * 1000), self._last_ms + 1)
        self._last_ms = ms
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
        )

        hand_job = self._pool.submit(self.hand.detect_for_video, image, ms)
        face = None
        fr = self.face.detect_for_video(image, ms) if want_face else None
        if fr is not None and fr.face_landmarks:
            pts = np.array([(p.x * w, p.y * h) for p in fr.face_landmarks[0]])
            blend = {c.category_name: c.score for c in fr.face_blendshapes[0]}
            face = Face(pts, blend, np.array(fr.facial_transformation_matrixes[0]))

        hand = None
        hr = hand_job.result()
        if hr.hand_landmarks:
            pts = np.array([(p.x, p.y) for p in hr.hand_landmarks[0]])
            world = np.array([(p.x, p.y, p.z) for p in hr.hand_world_landmarks[0]])
            right = hr.handedness[0][0].category_name == "Right"
            hand = Hand(pts, world, right)

        return Percept(t, frame, face, hand)

    def close(self):
        self._pool.shutdown()
        self.face.close()
        self.hand.close()
