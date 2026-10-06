from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATA = ROOT / "data"
GAZE_MODEL_PATH = DATA / "gaze_model.pkl"
FINGER_MODEL_PATH = DATA / "finger_model.pkl"
FINGER_DATA_PATH = DATA / "finger_data.npz"
SESSION_LOG = DATA / "session.csv"
STOP_FILE = DATA / "stop"


@dataclass
class Config:
    camera_index: int = 0
    cam_width: int = 640
    cam_height: int = 480

    box_size: tuple = (0.44, 0.38)
    box_limits: tuple = (0.06, 0.08, 0.94, 0.92)

    finger_smoothing: float = 0.6
    finger_on: float = 0.65
    finger_off: float = 0.35

    grab_hold_s: float = 0.06
    grab_release_s: float = 0.06
    rewind_s: float = 0.18
    drag_start: float = 0.06
    drag_delay_s: float = 0.25
    long_press_s: float = 1.2
    sticky_s: float = 0.4

    slider_hold_s: float = 0.15
    scroll_gain: float = 90.0
    volume_gain: float = 60.0

    action_hold_s: float = 0.6

    train_record_s: float = 8.0

    blink_threshold: float = 0.5
    brow_on: float = 0.60
    brow_off: float = 0.35
    dwell_s: float = 1.0
    dwell_radius: float = 0.035

    calib_settle_s: float = 0.9
    calib_sample_s: float = 1.0
