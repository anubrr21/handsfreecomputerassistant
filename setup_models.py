import urllib.request

from hfc.config import MODELS

BASE = "https://storage.googleapis.com/mediapipe-models"
FILES = {
    "face_landmarker.task": f"{BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task": f"{BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}

if __name__ == "__main__":
    MODELS.mkdir(exist_ok=True)
    for name, url in FILES.items():
        target = MODELS / name
        if target.exists():
            print(f"{name}: already present")
            continue
        print(f"{name}: downloading...")
        urllib.request.urlretrieve(url, target)
    print("Models ready.")
