import argparse
import csv
import subprocess
import time

from . import system

import cv2
import numpy as np
from pynput import keyboard

from . import brain as brain_module
from . import calibrate, train
from .assistant import Assistant
from .config import DATA, FINGER_MODEL_PATH, GAZE_MODEL_PATH, SESSION_LOG, STOP_FILE, Config
from .fusion import CursorFusion
from .gaze import GazeModel, features, is_blinking
from .gestures import TIP, FingerModel, HandReader
from .perception import Camera, Perception
from .skills import Laptop
from .spotify import Spotify
from .voice import Speaker, Voice
from .webspeech import find_chrome

HUD = "Hands Free Computer"
HELP = "Hands Free Computer - gestures (Ctrl+Alt+H closes)"
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
    (18, 19), (19, 20), (0, 17),
]
IRIS = (468, 473)
STATUS_COLOURS = {"CLICK": (0, 255, 0), "RIGHT CLICK": (0, 165, 255), "DRAG": (0, 0, 255),
                  "GRAB": (0, 255, 255), "SCROLL": (255, 200, 0), "VOLUME": (255, 200, 0),
                  "PAUSED": (120, 120, 120), "HOLD...": (255, 120, 255),
                  "VOICE TYPING": (0, 255, 0), "SWITCH WINDOWS": (0, 255, 0)}


VOICE_LABELS = {"loading": "voice: loading...", "idle": "just say a command", "armed": "LISTENING: speak now",
                "hearing": "hearing you...", "thinking": "thinking...", "speaking": "speaking", "off": "voice off"}


def draw_voice(img, voice):
    h, w = img.shape[:2]
    busy = voice.state in ("armed", "hearing", "thinking", "speaking")
    lines = [(VOICE_LABELS[voice.state] if voice.wake or voice.state != "idle" else "thumbs up to talk",
              (0, 255, 255) if busy else (200, 200, 200))]
    if voice.interim:
        lines.append(("... " + voice.interim[-54:], (0, 255, 255)))
    elif voice.overheard:
        lines.append(("ignored: " + voice.overheard[:48], (150, 150, 150)))
    if voice.heard:
        lines.append(("you: " + voice.heard[:52], (255, 255, 255)))
    if voice.reply:
        lines.append(("> " + voice.reply[:54], (160, 255, 160)))
    if voice.state == "off":
        lines.append((voice.error[:56], (80, 80, 255)))
    for i, (text, colour) in enumerate(lines):
        y = 84 + i * 20
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1, cv2.LINE_AA)
    for i, mic in enumerate(voice.mics):
        y = h - 92 + i * 16
        name = mic.name.replace("Microphone (", "").replace("Headset (", "")[:14]
        cv2.putText(img, name, (w - 230, y + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, name, (w - 230, y + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        length = int(100 * min(1.0, max(0.0, (np.log10(mic.level + 1e-5) + 4) / 4)))
        cv2.rectangle(img, (w - 112, y), (w - 12, y + 10), (60, 60, 60), -1)
        cv2.rectangle(img, (w - 112, y), (w - 112 + length, y + 10),
                      (60, 220, 60) if mic.speech else (160, 160, 160), -1)


def draw_hud(p, fusion, cfg, fps, screen, voice=None):
    img = p.frame.copy()
    h, w = img.shape[:2]
    if fusion.mode == "hand" and fusion.box is not None and p.hand is not None:
        x0, y0, x1, y1 = fusion.box
        cv2.rectangle(img, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (255, 255, 255), 1)
    if p.face is not None:
        for i in IRIS:
            cv2.circle(img, tuple(p.face.pts[i].astype(int)), 3, (0, 255, 255), -1, cv2.LINE_AA)
    if p.hand is not None:
        pts = (p.hand.pts * (w, h)).astype(int)
        for a, b in HAND_EDGES:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), (80, 220, 80), 1, cv2.LINE_AA)
        for part, tip in TIP.items():
            colour = (0, 220, 0) if fusion.reader.straight[part] else (0, 0, 255)
            cv2.circle(img, tuple(pts[tip]), 7, colour, -1, cv2.LINE_AA)
    if fusion.mode == "eyes" and fusion.gaze is not None:
        mx, my, mw, mh = w - 132, 8, 124, 70
        cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (255, 255, 255), 1)
        g = fusion.gaze / screen
        cv2.circle(img, (int(mx + g[0] * mw), int(my + g[1] * mh)), 4, (0, 255, 255), -1, cv2.LINE_AA)

    if fps < 20:
        cv2.rectangle(img, (0, h - 62), (w, h - 30), (0, 0, 200), -1)
        cv2.putText(img, "LOW LIGHT: predicting motion; more light = more precise", (8, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    colour = STATUS_COLOURS.get(fusion.status, (255, 255, 255))
    brain = "trained" if fusion.reader.trained else "untrained: Ctrl+Alt+T"
    pose = f"{fusion.reading.label}  {fusion.reading.fingers}" if fusion.reading else ""
    for text, pos, scale, col in (
        (fusion.status, (10, 34), 0.9, colour),
        (pose, (10, 58), 0.5, (255, 255, 255)),
        (f"{fusion.mode} mode  {fps:.0f} fps  {brain}", (10, h - 12), 0.5, (255, 255, 255)),
    ):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2 if scale > 0.6 else 1, cv2.LINE_AA)
    if voice is not None:
        draw_voice(img, voice)
    return cv2.resize(img, (400, 300))


def main():
    ap = argparse.ArgumentParser(description="Control the computer with your hand and eyes.")
    ap.add_argument("--dry-run", action="store_true", help="track and show the HUD but never touch the mouse")
    ap.add_argument("--headless", action="store_true", help="no windows (for smoke tests)")
    ap.add_argument("--seconds", type=float, default=0, help="quit after this many seconds")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--dwell", action="store_true", help="eyes mode: click by staring")
    ap.add_argument("--quiet", action="store_true", help="no click sounds")
    ap.add_argument("--train", action="store_true", help="start with the gesture training screen")
    ap.add_argument("--no-help", action="store_true", help="do not show the gesture sheet on start")
    ap.add_argument("--no-voice", action="store_true", help="run without the voice assistant")
    ap.add_argument("--no-wake", action="store_true",
                    help="do not listen for the wake word; talk only after a thumbs-up or Ctrl+Alt+V")
    ap.add_argument("--silent", action="store_true", help="the assistant shows its replies but does not speak")
    ap.add_argument("--mic", default=None, help="listen only on microphones whose name contains this text")
    ap.add_argument("--speech", choices=["web", "local"], default="web",
                    help="web: Chrome's recogniser (sharp; audio goes to Google). local: Whisper on this laptop")
    ap.add_argument("--strict-wake", action="store_true",
                    help="act only on sentences that start with the wake word or follow a thumbs-up")
    ap.add_argument("--model", default=None, help="speech model: base.en (default) or small.en if downloaded")
    ap.add_argument("--rules", action="store_true", help="ignore the trained finger classifiers")
    args = ap.parse_args()

    cfg = Config(camera_index=args.camera)
    sw, sh = system.screen_size()
    screen = np.array([sw, sh], dtype=float)
    mouse = system.DryMouse(sw // 2, sh // 2) if args.dry_run else system.Mouse(sound=not args.quiet)
    fingers = None if args.rules else FingerModel.load(FINGER_MODEL_PATH)
    fusion = CursorFusion(cfg, sw, sh, mouse, HandReader(cfg, fingers))
    fusion.dwell = args.dwell
    model = GazeModel.load(GAZE_MODEL_PATH, sw, sh)

    cam = Camera(cfg.camera_index, cfg.cam_width, cfg.cam_height)
    perception = Perception()

    requests = set()
    hotkeys = keyboard.GlobalHotKeys({
        "<ctrl>+<alt>+p": lambda: requests.add("pause"),
        "<ctrl>+<alt>+c": lambda: requests.add("calibrate"),
        "<ctrl>+<alt>+t": lambda: requests.add("train"),
        "<ctrl>+<alt>+h": lambda: requests.add("help"),
        "<ctrl>+<alt>+v": lambda: requests.add("listen"),
        "<ctrl>+<alt>+m": lambda: requests.add("mode"),
        "<ctrl>+<alt>+k": lambda: requests.add("keyboard"),
        "<ctrl>+<alt>+q": lambda: requests.add("quit"),
    })
    hotkeys.start()

    if not args.headless:
        cv2.namedWindow(HUD, cv2.WINDOW_AUTOSIZE)
        cv2.setWindowProperty(HUD, cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow(HUD, sw - 430, sh - 400)

    DATA.mkdir(exist_ok=True)
    STOP_FILE.unlink(missing_ok=True)
    log_file = open(SESSION_LOG, "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["t", "dt_ms", "infer_ms", "hand", "status", "pose", "fingers", "conf", "palm_x", "palm_y"])

    voice = speaker = None
    if not args.no_voice and not args.headless:
        def app_control(action):
            if action == "help":
                requests.add("help")
            elif (action == "pause") != fusion.paused:
                requests.add("pause")

        has_brain = brain_module.available()
        spotify = Spotify()
        laptop = Laptop(dry_run=args.dry_run, spotify_api=spotify)
        brain = brain_module.Brain(laptop, app_control) if has_brain else None
        speaker = Speaker(enabled=not args.silent)
        assistant = Assistant(laptop, brain, app_control)
        engine = args.speech if args.speech == "local" or find_chrome() else "local"
        voice = Voice(assistant.handle, speaker, wake=not args.no_wake, prefer_mic=args.mic,
                      model_name=args.model, accepts=None if args.strict_wake else assistant.is_clear_command,
                      engine=engine, screen=(sw, sh))
        voice.start()
        fusion.on_thumb = voice.arm
        print("Voice assistant: " + ("offline commands + LLM backend" if brain else
                                     "offline commands only (no API key in .env)")
              + ("; speech: Chrome (Google, online)" if engine == "web" else f"; speech: Whisper {voice.model_name}")
              + ("; Spotify connected" if spotify.ready else "; Spotify not connected (search only)"))

    print(f"Screen {sw}x{sh}. Ctrl+Alt+Q quits, Ctrl+Alt+P pauses.")
    if fingers:
        print(f"Finger classifiers loaded (worst held-out accuracy {min(fingers.accuracy.values()):.0%}).")
    else:
        print("No trained finger classifiers: using fixed thresholds. Ctrl+Alt+T trains them.")
    if args.train:
        requests.add("train")
    help_open = False
    if not args.headless and not args.no_help:
        requests.add("help")
    seq, fps, frames, start = -1, 0.0, 0, time.perf_counter()
    last = start
    try:
        while "quit" not in requests:
            seq, frame = cam.read(seq)
            eyes = fusion.mode == "eyes"
            t_in = time.perf_counter()
            p = perception.process(frame, want_face=eyes)
            infer_ms = (time.perf_counter() - t_in) * 1000

            gaze_px = None
            if eyes and model and p.face is not None and not is_blinking(p.face, cfg.blink_threshold):
                gaze_px = model.predict(features(p.face, frame.shape[1], frame.shape[0]))
            blend = p.face.blend if p.face is not None else None
            fusion.update(p.t, gaze_px, p.hand, blend)

            frames += 1
            now = time.perf_counter()
            dt = max(now - last, 1e-6)
            fps = 0.9 * fps + 0.1 / dt if fps else 1 / dt
            last = now
            r = fusion.reading
            palm = p.hand.pts[[0, 5, 9, 13, 17]].mean(axis=0) if p.hand is not None else None
            log.writerow([f"{now - start:.3f}", f"{dt * 1000:.1f}", f"{infer_ms:.1f}",
                          int(p.hand is not None), fusion.status,
                          r.label if r else "", r.fingers if r else "", f"{r.conf:.2f}" if r else "",
                          f"{palm[0]:.4f}" if palm is not None else "",
                          f"{palm[1]:.4f}" if palm is not None else ""])

            if not args.headless:
                cv2.imshow(HUD, draw_hud(p, fusion, cfg, fps, screen, voice))
                key = cv2.pollKey() & 0xFF
                if key in (ord("q"), 27):
                    requests.add("quit")
                elif key == ord("c"):
                    requests.add("calibrate")

            if "pause" in requests:
                requests.discard("pause")
                fusion.toggle_pause()
            if "mode" in requests:
                requests.discard("mode")
                fusion.release_all()
                fusion.mode = "eyes" if fusion.mode == "hand" else "hand"
            if "keyboard" in requests:
                requests.discard("keyboard")
                subprocess.Popen("osk.exe", shell=True)
            if "listen" in requests:
                requests.discard("listen")
                if voice is not None:
                    voice.arm()
            if "help" in requests:
                requests.discard("help")
                if not args.headless:
                    if help_open:
                        cv2.destroyWindow(HELP)
                    else:
                        cv2.namedWindow(HELP, cv2.WINDOW_AUTOSIZE)
                        cv2.setWindowProperty(HELP, cv2.WND_PROP_TOPMOST, 1)
                        sheet = train.cheat_sheet()
                        cv2.imshow(HELP, sheet)
                        cv2.moveWindow(HELP, (sw - sheet.shape[1]) // 2, 40)
                    help_open = not help_open
            if "train" in requests:
                requests.discard("train")
                if not args.headless:
                    was_paused = fusion.paused
                    if not was_paused:
                        fusion.toggle_pause()
                    new = train.run(cam, perception, cfg, sw, sh)
                    if new is not None:
                        new.save(FINGER_MODEL_PATH)
                        fusion.set_reader(HandReader(cfg, new))
                        print("Finger classifiers trained: "
                              + ", ".join(f"{c} {v:.0%}" for c, v in new.accuracy.items()))
                    fusion.paused = was_paused
            if "calibrate" in requests:
                requests.discard("calibrate")
                if not args.headless:
                    was_paused = fusion.paused
                    if not was_paused:
                        fusion.toggle_pause()
                    new = calibrate.run(cam, perception, cfg, sw, sh)
                    if new is not None:
                        model = new
                        model.save(GAZE_MODEL_PATH)
                        print(f"Calibrated: {model.cv_error_px:.0f} px error, model {model.choice}")
                    fusion.paused = was_paused

            if args.seconds and now - start > args.seconds:
                break
            if frames % 8 == 0 and STOP_FILE.exists():
                STOP_FILE.unlink(missing_ok=True)
                break
    finally:
        fusion.release_all()
        mouse.close()
        if voice is not None:
            voice.close()
            speaker.close()
        hotkeys.stop()
        cam.close()
        perception.close()
        log_file.close()
        cv2.destroyAllWindows()
        elapsed = time.perf_counter() - start
        print(f"{frames} frames in {elapsed:.1f}s ({frames / max(elapsed, 1e-6):.1f} fps)")
