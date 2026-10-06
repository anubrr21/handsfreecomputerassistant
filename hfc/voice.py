import datetime
import queue
import re
import subprocess
import threading
import time
import traceback
import winsound

import numpy as np

from .config import DATA, MODELS

RATE = 16000
BLOCK = 512
VAD_CONTEXT = 64
WAKE_WORDS = ("jarvis", "jervis", "jarvus", "javis", "jarvice", "jarves", "jarvi", "jarviz", "garvis", "charvis",
              "computer")
NOISE_PHRASES = {"", "you", "thank you", "thanks", "thanks for watching", "bye", "okay", "so"}
VOICE_LOG = DATA / "voice.log"
UTTERANCES = DATA / "utterances"
KEEP_UTTERANCES = 40
WHISPER_DIR = MODELS / "whisper"


def downloaded(name):
    return bool(list(WHISPER_DIR.glob(f"models--Systran--faster-whisper-{name}/snapshots/*/model.bin")))


class Speaker:

    SCRIPT = ("Add-Type -AssemblyName System.Speech; "
              "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate = 2; "
              "[Console]::Out.WriteLine('ready'); "
              "while (($l = [Console]::In.ReadLine()) -ne $null) { "
              "if ($l -eq '') { continue }; $s.Speak($l); [Console]::Out.WriteLine('done') }")

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._proc = None
        self._lock = threading.Lock()
        if enabled:
            threading.Thread(target=self._ensure, daemon=True).start()

    def _ensure(self):
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._proc = subprocess.Popen(
                    ["powershell", "-NoProfile", "-Command", self.SCRIPT],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    text=True, encoding="utf-8", creationflags=subprocess.CREATE_NO_WINDOW)
                self._proc.stdout.readline()

    def say(self, text):
        text = re.sub(r"\s+", " ", text).strip()
        if not self.enabled or not text:
            return
        try:
            self._ensure()
            with self._lock:
                self._proc.stdin.write(text + "\n")
                self._proc.stdin.flush()
                self._proc.stdout.readline()
        except OSError:
            self._proc = None

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()


def split_wake_word(text):
    words = text.strip().split()
    for i, word in enumerate(words[:3]):
        if re.sub(r"[^a-z]", "", word.lower()) in WAKE_WORDS:
            return " ".join(words[i + 1:]).strip(" ,.!?")
    return None


class SpeechDetector:

    def __init__(self):
        from faster_whisper.vad import get_vad_model
        self._session = get_vad_model().session
        self.reset()

    def reset(self):
        self._h = np.zeros((1, 1, 128), dtype="float32")
        self._c = np.zeros((1, 1, 128), dtype="float32")
        self._context = np.zeros(VAD_CONTEXT, dtype="float32")

    def __call__(self, block):
        frame = np.concatenate([self._context, block])[None].astype("float32")
        out, self._h, self._c = self._session.run(None, {"input": frame, "h": self._h, "c": self._c})
        self._context = block[-VAD_CONTEXT:]
        return float(np.ravel(out)[0])


class Segmenter:

    def __init__(self, start_blocks=3, end_silence_s=0.65, pre_roll_s=0.3, min_s=0.25, max_s=15.0):
        self.start_blocks = start_blocks
        self.end_blocks = int(end_silence_s * RATE / BLOCK)
        self.pre_roll = int(pre_roll_s * RATE / BLOCK)
        self.min_blocks = int(min_s * RATE / BLOCK)
        self.max_blocks = int(max_s * RATE / BLOCK)
        self.reset()

    def reset(self):
        self._recent = []
        self._speech = None
        self._run = 0
        self._quiet = 0
        self._voiced = 0

    @property
    def speaking(self):
        return self._speech is not None

    def take(self):
        if self._speech is None or self._voiced < self.min_blocks:
            return None
        return np.concatenate(self._speech)

    def feed(self, block, is_speech):
        if self._speech is None:
            self._recent.append(block)
            self._recent = self._recent[-(self.pre_roll + self.start_blocks):]
            self._run = self._run + 1 if is_speech else 0
            if self._run >= self.start_blocks:
                self._speech, self._quiet, self._voiced = list(self._recent), 0, self._run
            return None
        self._speech.append(block)
        self._quiet = 0 if is_speech else self._quiet + 1
        self._voiced += bool(is_speech)
        if self._quiet >= self.end_blocks or len(self._speech) >= self.max_blocks:
            speech, voiced = self._speech, self._voiced
            self.reset()
            if voiced >= self.min_blocks:
                return np.concatenate(speech)
        return None


class Mic:

    def __init__(self, index, name):
        self.index = index
        self.name = name
        self.level = 0.0
        self.speech = False
        self.blocks = queue.Queue()
        self.detector = SpeechDetector()
        self.segmenter = Segmenter()
        self.stream = None

    def open(self):
        import sounddevice as sd
        self.stream = sd.InputStream(device=self.index, samplerate=RATE, channels=1, dtype="float32",
                                     blocksize=BLOCK, callback=lambda data, *_: self.blocks.put(data[:, 0].copy()))
        self.stream.start()

    def step(self, block):
        self.level = max(float(np.sqrt(np.mean(block * block))), self.level * 0.85)
        self.speech = self.detector(block) > 0.5
        return self.segmenter.feed(block, self.speech)

    def clear(self):
        with self.blocks.mutex:
            self.blocks.queue.clear()
        self.segmenter.reset()
        self.detector.reset()
        self.speech = False

    def close(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()


def find_microphones(prefer=None):
    import sounddevice as sd
    devices = sd.query_devices()
    default = sd.default.device[0]
    api = devices[default]["hostapi"] if default is not None and default >= 0 else 0
    inputs = [(i, d["name"]) for i, d in enumerate(devices)
              if d["max_input_channels"] > 0 and d["hostapi"] == api and "Sound Mapper" not in d["name"]]
    if prefer:
        return [(i, n) for i, n in inputs if prefer.lower() in n.lower()]
    return sorted(inputs, key=lambda item: item[0] != default)[:3]


class Voice:
    def __init__(self, on_command, speaker, wake=True, model_name=None, prefer_mic=None, accepts=None,
                 engine="local", screen=(1920, 1080)):
        self.on_command = on_command
        self.engine = engine
        self.screen = screen
        self.interim = ""
        self._web = None
        self._finals = queue.Queue()
        self._interim_at = 0.0
        self._tainted = False
        self.accepts = accepts
        self.speaker = speaker
        self.wake = wake
        self.model_name = model_name if model_name and downloaded(model_name) else "base.en"
        self.prefer_mic = prefer_mic
        self.state = "loading"
        self.heard = ""
        self.reply = ""
        self.error = ""
        self.overheard = ""
        self.mics = []
        self._armed_until = 0.0
        self._running = True
        self._model = None
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def log(self, message):
        try:
            with open(VOICE_LOG, "a", encoding="utf-8") as fh:
                fh.write(f"{datetime.datetime.now():%H:%M:%S} {message}\n")
        except OSError:
            pass

    def arm(self, seconds=25.0, beep=True):
        if self.state in ("loading", "off", "thinking", "speaking"):
            return
        self._armed_until = time.perf_counter() + seconds
        if beep:
            threading.Thread(target=winsound.Beep, args=(880, 120), daemon=True).start()
            self.log("armed (thumbs-up or hotkey)")

    @property
    def armed(self):
        return time.perf_counter() < self._armed_until

    def transcribe_scored(self, audio):
        segments, _ = self._model.transcribe(
            audio, language="en", beam_size=5, temperature=0.0, vad_filter=True,
            condition_on_previous_text=False, hotwords="Computer Jarvis")
        segments = list(segments)
        text = " ".join(s.text.strip() for s in segments).strip()
        conf = float(np.mean([s.avg_logprob for s in segments])) if segments else -9.0
        return text, conf

    def transcribe(self, audio):
        return self.transcribe_scored(audio)[0]

    def load_model(self):
        from faster_whisper import WhisperModel
        self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8", cpu_threads=4,
                                   download_root=str(WHISPER_DIR))

    def _web_result(self, text, final):
        if self.state in ("thinking", "speaking"):
            self._tainted = True
            return
        self._interim_at = time.perf_counter()
        if not final:
            self.interim = text
            return
        self.interim = ""
        if self._tainted:
            self._tainted = False
            self.log(f"[chrome] dropped {text!r}: it overlapped the spoken reply")
            return
        self._finals.put(text)

    def _web_event(self, error):
        if error in ("not-allowed", "service-not-allowed"):
            self.error = "Chrome: microphone blocked. Allow it in the listening window."
        elif error != "aborted":
            self.log(f"[chrome] recogniser reported: {error}")

    def _loop_web(self):
        from .webspeech import WebSpeech
        self._web = WebSpeech(self._web_result, self._web_event)
        self._web.start(screen=self.screen)
        self.log("speech engine: Chrome (Google), language en-IN")
        announced = False
        while self._running:
            if not self._web.connected:
                self.state = "loading" if not announced else "off"
                if announced and not self.error:
                    self.error = "The Chrome listening window is closed."
                time.sleep(0.2)
                continue
            if not announced or self.state == "off":
                announced = True
                self.error = ""
                self.log("[chrome] listening window connected")
            try:
                text = self._finals.get(timeout=0.15)
            except queue.Empty:
                hearing = bool(self.interim) and time.perf_counter() - self._interim_at < 1.5
                self.state = "hearing" if hearing else ("armed" if self.armed else "idle")
                continue
            self._dispatch(text, "chrome", self.armed)

    def _loop(self):
        try:
            DATA.mkdir(exist_ok=True)
            VOICE_LOG.write_text("", encoding="utf-8")
            if self.engine == "web":
                self._loop_web()
                return
            self.load_model()
            for index, name in find_microphones(self.prefer_mic):
                mic = Mic(index, name)
                try:
                    mic.open()
                    self.mics.append(mic)
                    self.log(f"listening on microphone [{index}] {name}")
                except Exception as e:
                    self.log(f"could not open microphone [{index}] {name}: {e}")
            if not self.mics:
                raise RuntimeError("no microphone could be opened")
            self.state = "idle"
            while self._running:
                self._listen_once()
        except Exception as e:
            self.state, self.error = "off", str(e)
            self.log("STOPPED: " + traceback.format_exc())
        finally:
            for mic in self.mics:
                mic.close()

    def _listen_once(self):
        got_audio = False
        for mic in self.mics:
            while True:
                try:
                    block = mic.blocks.get_nowait()
                except queue.Empty:
                    break
                got_audio = True
                utterance = mic.step(block)
                if utterance is not None and (self.wake or self.armed):
                    heard = [(mic, utterance)]
                    for other in self.mics:
                        partial = other.segmenter.take() if other is not mic else None
                        if partial is not None:
                            heard.append((other, partial))
                    self._handle(heard)
                    for other in self.mics:
                        other.clear()
                    return
        if not got_audio:
            time.sleep(0.01)
        if self.state not in ("thinking", "speaking"):
            hearing = any(m.segmenter.speaking for m in self.mics)
            self.state = "hearing" if hearing else ("armed" if self.armed else "idle")

    def _handle(self, heard):
        was_armed = self.armed
        self.state = "thinking"
        t0 = time.perf_counter()
        versions = [(*self.transcribe_scored(audio), mic, audio) for mic, audio in heard]
        took = time.perf_counter() - t0
        if len(versions) > 1:
            self.log("candidates: " + "; ".join(f"[{m.name}] {t!r} ({c:.2f})" for t, c, m, _ in versions))
        text, conf, mic, utterance = max(versions, key=lambda v: v[1] + (split_wake_word(v[0]) is not None))
        self._save(utterance, text)
        self._dispatch(text, mic.name, was_armed, f"{len(utterance) / RATE:.1f}s, transcribed in {took:.1f}s")

    def _dispatch(self, text, source, was_armed, detail=""):
        self.state = "thinking"
        after_wake = split_wake_word(text)
        if after_wake is not None:
            command = after_wake
        elif was_armed:
            command = text
        elif self.accepts is not None and self.accepts(text):
            command = text
        else:
            self.log(f"[{source}] heard {text!r} ({detail}): not a clear command, ignored")
            self.overheard = text
            self.state = "idle"
            return
        if not command and after_wake is not None:
            self.log(f"[{source}] heard only the wake word; listening for the command")
            self.state = "idle"
            self.arm()
            return
        if re.sub(r"[^a-z ]", "", command.lower()).strip() in NOISE_PHRASES:
            self.log(f"[{source}] heard {text!r}: looks like noise, ignored")
            self.state = "idle"
            return
        self._armed_until = 0.0
        self.heard, self.reply, self.overheard = command, "", ""
        t0 = time.perf_counter()
        try:
            reply = self.on_command(command)
        except Exception as e:
            reply = f"Something went wrong: {e}"
            self.log("command failed: " + traceback.format_exc())
        self.reply = reply or ""
        self.log(f"[{source}] heard {text!r} ({detail}) -> command {command!r} "
                 f"-> {self.reply!r} ({time.perf_counter() - t0:.1f}s)")
        self.state = "speaking"
        self.speaker.say(self.reply)
        self.state = "idle"
        self.arm(seconds=10.0, beep=False)

    def _save(self, audio, text):
        try:
            import wave
            UTTERANCES.mkdir(parents=True, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%H%M%S")
            with wave.open(str(UTTERANCES / f"{stamp}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
            (UTTERANCES / f"{stamp}.txt").write_text(text, encoding="utf-8")
            for old in sorted(UTTERANCES.glob("*.wav"))[:-KEEP_UTTERANCES]:
                old.unlink()
                old.with_suffix(".txt").unlink(missing_ok=True)
        except OSError:
            pass

    def close(self):
        self._running = False
        if self._web is not None:
            self._web.close()
