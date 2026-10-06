import time
import winsound

import numpy as np
import sounddevice as sd

from hfc.voice import BLOCK, RATE, Speaker, Voice, find_microphones

SECONDS = 5
SENTENCE = "Computer, what time is it"


def main():
    speaker = Speaker()
    voice = Voice(on_command=None, speaker=None)
    voice.load_model()
    mics = find_microphones()

    speaker.say(f"Microphone check. After the beep, say: {SENTENCE}.")
    buffers = {i: [] for i, _ in mics}
    streams = []
    for i, _ in mics:
        s = sd.InputStream(device=i, samplerate=RATE, channels=1, dtype="float32", blocksize=BLOCK,
                           callback=lambda data, *_, i=i: buffers[i].append(data[:, 0].copy()))
        s.start()
        streams.append(s)
    time.sleep(0.2)
    winsound.Beep(880, 200)
    time.sleep(SECONDS)
    for s in streams:
        s.stop()
        s.close()
    winsound.Beep(600, 150)

    for i, name in mics:
        x = np.concatenate(buffers[i])[int(0.5 * RATE):]
        text = voice.transcribe(x)
        spec = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(len(x), 1 / RATE)
        treble = spec[(freqs > 1000) & (freqs < 4000)].sum() / (spec[(freqs > 100) & (freqs < 4000)].sum() + 1e-12)
        print(f"{name}: peak {np.abs(x).max():.2f}, treble share {treble:.3f}, heard {text!r}")
        short = "the laptop microphone" if "Realtek" in name else "the " + name.split("(")[-1].rstrip(")")
        speaker.say(f"This is what {short} recorded.")
        loud = x / (np.abs(x).max() + 1e-9) * 0.8
        sd.play(loud, RATE)
        sd.wait()
        time.sleep(0.3)
        speaker.say("The speech model heard: " + (text or "nothing") + ".")
    speaker.say("Microphone check finished.")
    speaker.close()


if __name__ == "__main__":
    main()
