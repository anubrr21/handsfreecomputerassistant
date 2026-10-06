# Hands Free Computer

Control a Windows laptop without touching it: move the cursor and click with hand gestures in
front of the webcam, and open apps, search the web, play music and control the system by voice.

Everything runs on an ordinary laptop with a webcam and a microphone. No extra hardware and no
GPU are needed.

## Features

- **Gesture mouse**: cursor movement, click, double-click, drag, right-click, scroll and volume,
  all from one hand in the air
- **Per-finger gesture recognition**: a small classifier per finger, trainable on your own hand
  in about a minute, so gestures are defined simply by which fingers are up
- **Voice assistant**: open apps, folders and websites, search, play music, control media and
  windows, type, and chain several commands in one sentence
- **Spotify playback**: "play Kesariya on Spotify" starts the actual track through the Spotify Web API
- **Optional LLM backend**: free-form requests and questions, with tool use over the same actions
- **Experimental eye mode**: a per-user gaze model that moves the cursor with your eyes
- **Low-light handling**: cursor motion prediction when the webcam drops to 15 fps

## Requirements

- Windows 10 or 11
- Python 3.10
- A webcam and a microphone
- Google Chrome (used for speech recognition)

## Setup

```bash
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe setup_models.py
```

Then start it:

```bash
.venv\Scripts\python.exe main.py
```

A small camera preview opens in the bottom-right corner, and a Chrome "listening" window opens
for speech. Allow the microphone the first time it asks.

## Gestures

Every gesture is defined by which fingers are straight and which are curled.

| Hand | Action |
|---|---|
| Open hand, or pointing | Move the cursor |
| Fist | Mouse button: close and open to click, twice to double-click, close and move to drag, close and hold to right-click |
| Two fingers up | Scroll (move the hand up or down) |
| Little finger up | Volume (move the hand up or down) |
| Index and little finger up, held | Switch windows |
| Thumbs up, held | Make the assistant listen for one command |

Lower your hand and the normal mouse works again. In the preview, a green dot on a fingertip
means that finger is read as straight, and a red dot means curled.

To train the finger classifiers on your own hand, press `Ctrl+Alt+T` and follow the on-screen
poses. Without training, fixed joint-angle thresholds are used.

## Voice commands

Clear, harmless commands run as soon as they are heard. Actions that type, click, close windows
or change power state need the wake word first ("Computer, ..." or "Jarvis, ...") or a thumbs-up.

| Say | Action |
|---|---|
| "open Chrome", "open Spotify", "launch VS Code" | Start any app from the Start menu |
| "open my downloads folder", "open the file budget report" | Open a folder or find a file by name |
| "go to YouTube", "open github.com" | Open a website |
| "search for the weather in Kolkata" | Web search |
| "play Shape of You on YouTube" | Find the video and play it |
| "play Kesariya on Spotify" | Play the track on Spotify |
| "open Chrome and search for cricket score" | Several commands in one sentence |
| "pause", "next song", "volume to 40 percent", "mute" | Media and volume |
| "Computer, close the tab", "Computer, take a screenshot" | Window and keyboard shortcuts |
| "Computer, type hello world" | Typing |
| "what time is it", "how much battery is left" | Spoken answer |
| "Computer, lock my laptop", "Computer, shut down" | Power (shut down, restart and sleep ask for confirmation) |

## Hotkeys

| Hotkey | Action |
|---|---|
| Ctrl+Alt+H | Show or hide the gesture sheet |
| Ctrl+Alt+T | Train the finger classifiers |
| Ctrl+Alt+V | Make the assistant listen for one command |
| Ctrl+Alt+P | Pause or resume hand control |
| Ctrl+Alt+M | Switch between hand mode and eye mode |
| Ctrl+Alt+C | Calibrate gaze for eye mode |
| Ctrl+Alt+K | Open the on-screen keyboard |
| Ctrl+Alt+Q | Quit |

## Optional integrations

Copy `.env.example` to `.env` and fill in what you want to use.

**Spotify playback** (Spotify Premium required)

1. Create an app at https://developer.spotify.com/dashboard with the redirect URI
   `http://127.0.0.1:8765/callback`.
2. Put its Client ID in `.env` as `SPOTIFY_CLIENT_ID`.
3. Run `.venv\Scripts\python.exe spotify_login.py` once and approve access in the browser.

**LLM backend**

Put an Anthropic API key in `.env` as `ANTHROPIC_API_KEY`. Requests that the built-in command
patterns do not cover are then sent to the model, which can call the same actions as tools.
Shell commands it proposes are shown in a confirmation box and run only if you accept.

## Speech recognition

By default speech is recognised by Chrome's built-in recogniser through a small local page,
set to Indian English. Microphone audio is sent to Google while that window is open.

To keep speech on the device, start with `--speech local`. This uses Whisper with a Silero
voice-activity detector. It is slower and less accurate on accented speech.

## Command-line options

| Option | Effect |
|---|---|
| `--dry-run` | Track and show the preview without moving the mouse |
| `--train` | Open the training screen on start |
| `--rules` | Ignore trained finger classifiers and use fixed thresholds |
| `--speech web\|local` | Choose the speech engine |
| `--model small.en` | Larger Whisper model for the local engine (run `download_model.py` first) |
| `--mic NAME` | Use only microphones whose name contains NAME (local engine) |
| `--strict-wake` | Require the wake word for every command |
| `--no-voice` | Run without the voice assistant |
| `--silent` | Show replies without speaking them |
| `--no-help` | Do not show the gesture sheet on start |

## How it works

- **Landmarks**: MediaPipe hand (21 points) and face (478 points) models run on every frame.
- **Finger states**: joint angles, which do not change with hand position, size or rotation,
  feed one logistic-regression classifier per finger. Gestures are lookups on the five states.
- **Cursor**: the palm centre is mapped through a control box placed around the hand when it is
  raised, smoothed with a One Euro filter, and driven at 120 Hz with velocity prediction between
  camera frames.
- **Clicks**: a click lands where the cursor pointed just before the hand began to close, and a
  single misread frame cannot click or drop a drag.
- **Voice**: recognised text is matched against command patterns; multi-step sentences are split
  and run in order.
- **Gaze**: iris position, eye blendshapes and head pose feed a ridge regression fitted per user,
  with model selection by leave-one-target-out cross-validation.

## Project layout

```
main.py              entry point
setup_models.py      downloads the MediaPipe models
spotify_login.py     one-time Spotify authorisation
mic_check.py         records and plays back each microphone
download_model.py    downloads the larger Whisper model
hfc/
  app.py             main loop, preview, hotkeys
  perception.py      camera capture and landmark models
  gestures.py        finger features, classifiers, gesture naming
  fusion.py          gestures to cursor, clicks, scroll and shortcuts
  system.py          mouse and keyboard output
  train.py           training screen and gesture sheet
  gaze.py            gaze features and model
  calibrate.py       gaze calibration screen
  voice.py           listening loop, wake word, spoken replies
  webspeech.py       Chrome speech engine
  assistant.py       command parsing
  skills.py          actions: apps, files, web, keys, media, power
  spotify.py         Spotify Web API client
  brain.py           LLM backend with tool use
tests/               unit tests (no camera, microphone or network needed)
```

## Tests

```bash
.venv\Scripts\python.exe tests\test_core.py
.venv\Scripts\python.exe tests\test_assistant.py
```
