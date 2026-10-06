import os
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download

from hfc.config import MODELS

REPO = "Systran/faster-whisper-small.en"

if __name__ == "__main__":
    while True:
        try:
            path = snapshot_download(REPO, cache_dir=str(MODELS / "whisper"))
            print("Done:", path)
            break
        except Exception as e:
            print(f"Interrupted ({type(e).__name__}); retrying in 10 seconds...")
            time.sleep(10)
