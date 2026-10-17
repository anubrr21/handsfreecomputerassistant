import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hfc.assistant import Assistant, parse
from hfc.brain import MODEL, TOOLS, Brain
from hfc.skills import Laptop
from hfc.voice import BLOCK, RATE, Segmenter, split_wake_word


class ParseTests(unittest.TestCase):
    CASES = {
        "open chrome": ("open", ("chrome", "any")),
        "Could you please open Chrome?": ("open", ("chrome", "any")),
        "launch the Spotify app": ("open", ("spotify", "any")),
        "open my downloads folder": ("open", ("downloads", "folder")),
        "open the file budget report": ("open", ("budget report", "file")),
        "go to youtube": ("open", ("youtube", "any")),
        "play Shape of You on YouTube": ("youtube", ("shape of you", True)),
        "search youtube for lofi beats": ("youtube", ("lofi beats", False)),
        "open youtube and search for python tutorials": ("youtube", ("python tutorials", False)),
        "play Kesariya on Spotify": ("spotify", ("kesariya",)),
        "open spotify and play arijit singh": ("spotify", ("arijit singh",)),
        "spotify and play shape of you": ("spotify", ("shape of you",)),
        "on spotify play kesariya": ("spotify", ("kesariya",)),
        "spotify and play the song believer": ("spotify", ("believer",)),
        "play believer": ("youtube", ("believer", True)),
        "search for the weather in Kolkata": ("web_search", ("the weather in kolkata",)),
        "google best laptops under 50000": ("web_search", ("best laptops under 50000",)),
        "open chrome and search for cricket score": ("web_search", ("cricket score",)),
        "set volume to 40%": ("volume", (None, 40)),
        "volume to fifty percent": ("volume", (None, 50)),
        "volume up": ("volume", (5, None)),
        "make it quieter": ("volume", (-5, None)),
        "mute": ("media", ("mute",)),
        "pause": ("media", ("pause",)),
        "pause the music": ("media", ("pause",)),
        "play": ("media", ("play",)),
        "next song": ("media", ("next",)),
        "what time is it": ("time", ()),
        "what's the date": ("time", ()),
        "how much battery is left": ("battery", ()),
        "take a screenshot": ("shortcut", ("screenshot",)),
        "close this window": ("shortcut", ("close window",)),
        "new tab": ("shortcut", ("new tab",)),
        "copy": ("shortcut", ("copy",)),
        "press control shift t": ("press", ("ctrl+shift+t",)),
        "press enter": ("shortcut", ("enter",)),
        "double click": ("click", ("double",)),
        "right click": ("click", ("right",)),
        "scroll down": ("scroll", ("down",)),
        "lock my laptop": ("lock", ()),
        "shut down the computer": ("power", ("shutdown",)),
        "restart": ("power", ("restart",)),
        "pause hand control": ("app", ("pause",)),
        "resume hand tracking": ("app", ("resume",)),
        "show me the gestures": ("app", ("help",)),
    }

    def test_commands(self):
        for said, expected in self.CASES.items():
            self.assertEqual(parse(said), expected, said)

    def test_stutters_and_stray_leading_words_are_tolerated(self):
        self.assertEqual(parse("Close, close the tab"), ("shortcut", ("close tab",)))
        self.assertEqual(parse("uh so volume up"), ("volume", (5, None)))
        self.assertEqual(parse("and what time is it"), ("time", ()))
        self.assertIsNone(parse("we will open the meeting"))

    def test_typing_keeps_the_original_wording(self):
        self.assertEqual(parse("Type Hello World, how are you?"), ("type_text", ("Hello World, how are you",)))

    def test_open_ended_requests_are_left_for_the_brain(self):
        for said in ("summarise my last email", "why is the sky blue", "rename the report file to final", ""):
            self.assertIsNone(parse(said), said)

    def test_natural_phrasing(self):
        cases = {
            "lower the volume to 24": ("volume", (None, 24)),
            "volume low": ("volume", (None, 20)),
            "sound to 60 percent": ("volume", (None, 60)),
            "reduce the volume by 10": ("volume", (-5, None)),
            "turn the volume up": ("volume", (5, None)),
            "set brightness to 40": ("brightness", (None, 40)),
            "dim the screen": ("brightness", (-10, None)),
            "close the Microsoft Edge": ("close_app", ("microsoft edge",)),
            "quit spotify": ("close_app", ("spotify",)),
            "close the tab": ("shortcut", ("close tab",)),
            "switch to chrome": ("focus", ("chrome",)),
            "fire up notepad": ("open", ("notepad", "any")),
            "open google chrome browser": ("open", ("chrome", "any")),
            "what's the weather in Kolkata": ("weather", ("kolkata",)),
            "is it going to rain today": ("weather", ("",)),
            "what is 23 times 7": ("calc", ("23 times 7",)),
            "who is Sachin Tendulkar": ("wiki", ("sachin tendulkar",)),
            "tell me about the Taj Mahal": ("wiki", ("taj mahal",)),
            "set a timer for 5 minutes": ("timer", (300, "")),
            "remind me in 10 minutes to drink water": ("timer", (600, "drink water")),
            "shut down": ("power", ("shutdown",)),
        }
        for said, expected in cases.items():
            self.assertEqual(parse(said), expected, said)


class AssistantTests(unittest.TestCase):
    def setUp(self):
        self.laptop = Laptop(dry_run=True)
        self.laptop.apps = {"Google Chrome": "Chrome", "Spotify": "SpotifyAB.SpotifyMusic", "Word": "Word.App"}
        self.controls = []
        self.assistant = Assistant(self.laptop, app_control=self.controls.append)

    def test_open_app_by_spoken_alias(self):
        self.assertEqual(self.assistant.handle("open chrome"), "Opening Google Chrome.")
        self.assertEqual(self.laptop.log, [("app", "Google Chrome")])

    def test_open_misheard_app_name(self):
        self.assertEqual(self.assistant.handle("open spotifi"), "Opening Spotify.")

    def test_site_when_no_such_app(self):
        self.assistant.handle("open youtube")
        self.assertEqual(self.laptop.log, [("url", "https://www.youtube.com")])

    def test_known_folder(self):
        self.assertEqual(self.assistant.handle("open my downloads folder"), "Opening downloads.")
        self.assertEqual(self.laptop.log[0][0], "path")

    def test_web_search_builds_a_google_url(self):
        self.assistant.handle("search for the weather in Kolkata")
        self.assertEqual(self.laptop.log, [("url", "https://www.google.com/search?q=the+weather+in+kolkata")])

    def test_volume_and_media_and_keys(self):
        self.assistant.handle("set volume to 40%")
        self.assistant.handle("next song")
        self.assistant.handle("press control shift t")
        self.assertEqual(self.laptop.log, [("volume", "=40"), ("media", "next"), ("keys", "ctrl+shift+t")])

    def test_shutdown_asks_first_and_does_nothing_without_a_yes(self):
        reply = self.assistant.handle("shut down the computer")
        self.assertEqual(self.laptop.log, [("confirm", "Really shutdown the computer now?")])
        self.assertIn("not doing", reply)

    def test_several_commands_in_one_sentence_run_in_order(self):
        import hfc.assistant as module
        module.time.sleep = lambda s: None
        reply = self.assistant.handle("open chrome and search for cricket score then volume up")
        self.assertEqual(self.laptop.log, [("app", "Google Chrome"),
                                           ("url", "https://www.google.com/search?q=cricket+score"),
                                           ("volume", "+5")])
        self.assertIn("Opening Google Chrome.", reply)

    def test_open_an_app_and_play_there_stays_in_that_app(self):
        self.assistant.handle("open spotify and play kesariya")
        self.assertEqual(self.laptop.log, [("spotify", "kesariya")])
        self.laptop.log.clear()
        self.assistant.handle("open youtube and play believer")
        self.assertEqual([kind for kind, _ in self.laptop.log], ["url"])

    def test_a_sentence_with_and_in_a_name_is_not_split_wrongly(self):
        self.assistant.handle("play rock and roll on youtube")
        self.assertEqual(self.laptop.log[0][0], "url")
        self.assertEqual(len(self.laptop.log), 1)

    def test_app_control_is_forwarded(self):
        self.assistant.handle("pause hand control")
        self.assertEqual(self.controls, ["pause"])

    def test_folders_are_found_by_spoken_name(self):
        import tempfile
        import hfc.skills as skills
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            (home / "iTantra").mkdir()
            (home / "Downloads" / "iTantra-notes").mkdir(parents=True)
            (home / "Downloads" / "budget report.pdf").write_text("x")
            (home / "AppData" / "iTantra").mkdir(parents=True)
            old_home, old_known = skills.HOME, dict(skills.KNOWN_FOLDERS)
            skills.HOME = home
            skills.KNOWN_FOLDERS.update(downloads=home / "Downloads", download=home / "Downloads")
            try:
                found = self.laptop.find_paths("i tantra", "folder")
                self.assertEqual(Path(found[0]).name, "iTantra")
                self.assertFalse(any("AppData" in Path(f).relative_to(home).parts for f in found))
                self.assertEqual(self.assistant.handle("open I Tantra folder for me"), "Opening iTantra.")
                self.assertTrue(self.assistant.is_clear_command("open I Tantra folder"))
                self.assertFalse(self.assistant.is_clear_command("open the door"))
                self.assertEqual(parse("open itra from download folder for me"), ("open_in", ("itra", "download")))
                self.assertEqual(self.assistant.handle("open notes from download folder for me"), "Opening iTantra-notes.")
                self.assertEqual(self.assistant.handle("open itra from download folder for me"), "Opening iTantra.")
                (home / "Downloads" / "tr.pak").write_text("x")
                self.assertEqual(self.assistant.handle("open itra from download folder"), "Opening iTantra.")
                self.assertEqual(self.assistant.handle("open budget report from downloads"), "Opening budget report.pdf.")
            finally:
                skills.HOME = old_home
                skills.KNOWN_FOLDERS.clear()
                skills.KNOWN_FOLDERS.update(old_known)

    def test_arithmetic_is_computed_safely(self):
        lap = self.laptop
        self.assertEqual(lap.calculate("23 times 7"), "That is 161.")
        self.assertEqual(lap.calculate("15 percent of 200"), "That is 30.")
        self.assertEqual(lap.calculate("12 plus 5 divided by 2"), "That is 14.5.")
        self.assertEqual(lap.calculate("square root of 144"), "That is 12.")
        self.assertIsNone(lap.calculate("__import__('os').system('dir')"))
        self.assertIsNone(lap.calculate("9 to the power of 9999"))

    def test_split_sentences_are_joined(self):
        join = self.assistant.join_fragments
        self.assertEqual(join("lower the volume", "to 24"), "lower the volume to 24")
        self.assertEqual(join("play", "believer on spotify"), "play believer on spotify")
        self.assertIsNone(join("open chrome", "what time is it"))
        self.assertIsNone(join("hello there", "shut down the computer"))

    def test_closing_an_app_needs_the_wake_word(self):
        self.assertFalse(self.assistant.is_clear_command("close spotify"))
        self.assertEqual(self.assistant.handle("close spotify"), "Closed Spotify.")
        self.assertEqual(self.laptop.log, [("close", "Spotify")])

    def test_new_harmless_commands_need_no_wake_word(self):
        for said in ("lower the volume to 24", "what's the weather in Kolkata", "what is 23 times 7",
                     "who is Sachin Tendulkar", "set a timer for 5 minutes", "scroll down", "set brightness to 40"):
            self.assertTrue(self.assistant.is_clear_command(said), said)

    def test_timer_and_brightness_and_weather_run(self):
        self.assertEqual(self.assistant.handle("set a timer for 5 minutes"), "Timer set for 5 minutes.")
        self.assertEqual(self.assistant.handle("remind me in 10 minutes to drink water"), "I'll remind you in 10 minutes.")
        self.assistant.handle("set brightness to 40")
        self.assistant.handle("what's the weather in Kolkata")
        self.assertEqual(self.laptop.log, [("timer", "300s"), ("timer", "600s drink water"),
                                           ("brightness", "=40"), ("weather", "kolkata")])

    def test_factual_questions_go_to_the_brain_when_there_is_one(self):
        self.assistant.brain = SimpleNamespace(ask=lambda text: f"brain: {text}")
        self.assertEqual(self.assistant.handle("who is Sachin Tendulkar"), "brain: who is Sachin Tendulkar")
        self.assertEqual(self.assistant.handle("volume up"), "Volume up.")

    def test_without_a_brain_unknown_requests_get_an_honest_reply(self):
        self.assertIn("didn't catch", self.assistant.handle("why is the sky blue"))
        self.assertIn("couldn't find", self.assistant.handle("open the zzqqxx thing"))

    def test_clear_harmless_commands_need_no_wake_word(self):
        for said in ("What time is it of the day?", "What time is it, Master Day?", "open chrome",
                     "open YouTube", "play Believer on YouTube", "volume up", "open my downloads"):
            self.assertTrue(self.assistant.is_clear_command(said), said)

    def test_risky_or_unclear_sentences_do_need_the_wake_word(self):
        for said in ("shut down the computer", "type hello world", "close this window", "press enter",
                     "click", "lock my laptop", "open the door", "Let them sequel you too.",
                     "I was telling my friend about the match", "why is the sky blue"):
            self.assertFalse(self.assistant.is_clear_command(said), said)

    def test_a_mangled_sentence_that_names_one_app_or_site_opens_it(self):
        self.assertEqual(self.assistant.handle("and seek for new YouTube."), "Opening it in your browser.")
        self.assertEqual(self.laptop.log, [("url", "https://www.youtube.com")])
        self.assertEqual(self.assistant.handle("come build up open the spotify"), "Opening Spotify.")

    def test_a_mangled_sentence_naming_two_things_is_not_guessed(self):
        self.assertIn("didn't catch", self.assistant.handle("something youtube something spotify"))

    def test_unknown_requests_go_to_the_brain(self):
        self.assistant.brain = SimpleNamespace(ask=lambda text: f"brain got: {text}")
        self.assertEqual(self.assistant.handle("why is the sky blue"), "brain got: why is the sky blue")


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(id, tool, **input):
    return SimpleNamespace(type="tool_use", id=id, name=tool, input=input)


class FakeClient:

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


class BrainTests(unittest.TestCase):
    def setUp(self):
        self.laptop = Laptop(dry_run=True)
        self.laptop.apps = {"Google Chrome": "Chrome"}

    def brain(self, *responses):
        self.client = FakeClient(responses)
        return Brain(self.laptop, client=self.client)

    def test_plain_question_returns_the_spoken_answer(self):
        brain = self.brain(SimpleNamespace(stop_reason="end_turn", content=[text_block("Jawaharlal Nehru.")]))
        self.assertEqual(brain.ask("who was the first prime minister of India"), "Jawaharlal Nehru.")
        req = self.client.requests[0]
        self.assertEqual(req["model"], MODEL)
        self.assertEqual(req["tools"], TOOLS)
        self.assertEqual(req["messages"][-1], {"role": "user", "content": "who was the first prime minister of India"})

    def test_tool_calls_are_run_and_their_results_sent_back(self):
        brain = self.brain(
            SimpleNamespace(stop_reason="tool_use", content=[
                tool_block("t1", "open_app", name="chrome"), tool_block("t2", "set_volume", level=30)]),
            SimpleNamespace(stop_reason="end_turn", content=[text_block("Chrome is open and volume is at 30.")]),
        )
        self.assertEqual(brain.ask("open chrome and turn it down to 30"), "Chrome is open and volume is at 30.")
        self.assertEqual(self.laptop.log, [("app", "Google Chrome"), ("volume", "=30")])
        results = self.client.requests[1]["messages"][-1]
        self.assertEqual(results["role"], "user")
        self.assertEqual([r["tool_use_id"] for r in results["content"]], ["t1", "t2"])
        self.assertEqual(results["content"][0]["content"], "Opening Google Chrome.")

    def test_shell_command_is_not_run_without_the_users_yes(self):
        brain = self.brain(
            SimpleNamespace(stop_reason="tool_use", content=[tool_block("t1", "run_powershell", command="Get-Date")]),
            SimpleNamespace(stop_reason="end_turn", content=[text_block("You said no, so I left it.")]),
        )
        brain.ask("what is the date according to powershell")
        self.assertEqual(self.laptop.log, [("confirm", "The assistant wants to run this command:\n\nGet-Date\n\nAllow it?")])
        self.assertIn("did not allow", self.client.requests[1]["messages"][-1]["content"][0]["content"])

    def test_a_failing_tool_is_reported_to_the_model_not_raised(self):
        self.laptop.media = lambda action: 1 / 0
        brain = self.brain(
            SimpleNamespace(stop_reason="tool_use", content=[tool_block("t1", "media", action="next")]),
            SimpleNamespace(stop_reason="end_turn", content=[text_block("That didn't work.")]),
        )
        self.assertEqual(brain.ask("next song but weirdly"), "That didn't work.")
        self.assertTrue(self.client.requests[1]["messages"][-1]["content"][0]["is_error"])

    def test_follow_up_questions_carry_the_earlier_turns(self):
        brain = self.brain(
            SimpleNamespace(stop_reason="end_turn", content=[text_block("Paris.")]),
            SimpleNamespace(stop_reason="end_turn", content=[text_block("About two million.")]),
        )
        brain.ask("capital of France")
        brain.ask("how many people live there")
        self.assertEqual([m["content"] for m in self.client.requests[1]["messages"]],
                         ["capital of France", "Paris.", "how many people live there"])

    def test_refusal_gives_a_short_spoken_reply(self):
        brain = self.brain(SimpleNamespace(stop_reason="refusal", content=[]))
        self.assertEqual(brain.ask("something"), "I can't help with that one.")


class SpotifyTests(unittest.TestCase):

    def client(self, responses, tmp):
        from hfc.spotify import Spotify
        sp = Spotify(client_id="cid", token_file=tmp / "token.json")
        sp._store({"access_token": "tok", "refresh_token": "ref", "expires_in": 3600})
        self.calls = []

        def fake(method, url, body=None, headers=None, form=False):
            self.calls.append((method, url.split("?")[0], body))
            return responses.pop(0)

        sp._http = fake
        return sp

    def test_play_searches_then_starts_the_track_on_the_active_device(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            sp = self.client([
                {"tracks": {"items": [
                    {"name": "Shape Of You x Mann Mera", "uri": "spotify:track:9", "album": {"uri": "spotify:album:9"},
                     "artists": [{"name": "TommyMuzzic"}]},
                    {"name": "Shape of You", "uri": "spotify:track:1", "album": {"uri": "spotify:album:1"},
                     "artists": [{"name": "Ed Sheeran"}]}]}},
                {"devices": [{"id": "phone", "is_active": False}, {"id": "laptop", "is_active": True}]},
                {},
            ], Path(d))
            self.assertEqual(sp.play("shape of you"), ("Shape of You", "Ed Sheeran"))
            self.assertEqual(self.calls[2][0], "PUT")
            self.assertEqual(self.calls[2][2], {"context_uri": "spotify:album:1",
                                                "offset": {"uri": "spotify:track:1"}, "position_ms": 0})

    def test_vague_requests_become_playlists_and_titles_stay_titles(self):
        from hfc.spotify import describe
        self.assertEqual(describe("a hindi song"), "hindi")
        self.assertEqual(describe("an English song"), "english")
        self.assertEqual(describe("a Bengali Durga Puja song"), "bengali durga puja")
        self.assertEqual(describe("a song"), "top hits")
        self.assertEqual(describe("some romantic music"), "romantic")
        self.assertIsNone(describe("shape of you"))
        self.assertIsNone(describe("believer imagine dragons"))
        self.assertIsNone(describe("a sky full of stars"))

    def test_playlist_is_chosen_by_matching_words_and_played_shuffled(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            sp = self.client([
                {"playlists": {"items": [None, {"name": "Party Mix", "uri": "spotify:playlist:p"},
                                         {"name": "Hindi Hits 2026", "uri": "spotify:playlist:h"}]}},
                {"devices": [{"id": "laptop", "is_active": True}]},
                {}, {},
            ], Path(d))
            self.assertEqual(sp.play_playlist("hindi"), "Hindi Hits 2026")
            self.assertIn("shuffle", self.calls[2][1])
            self.assertEqual(self.calls[3][2], {"context_uri": "spotify:playlist:h"})

    def test_no_open_device_starts_the_app_and_tries_again(self):
        import tempfile
        import hfc.spotify as module
        module.time.sleep = lambda s: None
        woke = []
        with tempfile.TemporaryDirectory() as d:
            sp = self.client([
                {"tracks": {"items": [{"name": "X", "uri": "spotify:track:2", "album": {"uri": "spotify:album:2"}, "artists": []}]}},
                {"devices": []}, {"devices": []}, {"devices": [{"id": "laptop", "is_active": False}]}, {},
            ], Path(d))
            sp.play("x", wake=lambda: woke.append(1))
            self.assertEqual(woke, [1])

    def test_expired_token_is_refreshed_before_use(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            sp = self.client([{"access_token": "new", "expires_in": 3600}], Path(d))
            sp._tokens["expires_at"] = 0
            self.assertEqual(sp._access_token(), "new")
            self.assertEqual(self.calls[0][2]["grant_type"], "refresh_token")
            self.assertEqual(sp._tokens["refresh_token"], "ref")

    def test_not_connected_means_not_ready(self):
        import tempfile
        from hfc.spotify import Spotify
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(Spotify(client_id="cid", token_file=Path(d) / "none.json").ready)
            self.assertFalse(Spotify(client_id="", token_file=Path(d) / "none.json").ready)


class WebSpeechTests(unittest.TestCase):

    def post(self, port, path, body):
        import json
        import urllib.request
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        return urllib.request.urlopen(req, timeout=5).read()

    def test_page_is_served_and_results_reach_the_callback(self):
        import urllib.request
        from hfc.webspeech import WebSpeech
        got, events = [], []
        web = WebSpeech(lambda text, final: got.append((text, final)), events.append, port=8799)
        web.start(open_window=False)
        try:
            page = urllib.request.urlopen("http://127.0.0.1:8799/", timeout=5).read().decode()
            self.assertIn("en-IN", page)
            self.assertFalse(web.connected)
            self.post(8799, "/heard", {"text": "open", "final": False})
            self.post(8799, "/heard", {"text": "open Chrome", "final": True})
            self.post(8799, "/event", {"error": "network"})
            self.assertEqual(got, [("open", False), ("open Chrome", True)])
            self.assertEqual(events, ["network"])
            self.assertTrue(web.connected)
        finally:
            web.close()

    def voice(self):
        from hfc.voice import Voice

        class Quiet:
            def say(self, text):
                pass

        self.ran = []
        v = Voice(lambda text: self.ran.append(text) or "ok", Quiet(), engine="web",
                  accepts=lambda text: text.lower().startswith("open"))
        v.log = lambda message: None
        v.state = "idle"
        return v

    def test_final_sentences_are_queued_and_partials_shown(self):
        v = self.voice()
        v._web_result("open", False)
        self.assertEqual(v.interim, "open")
        v._web_result("open Chrome", True)
        self.assertEqual((v.interim, v._finals.get_nowait()), ("", "open Chrome"))

    def test_our_own_spoken_reply_is_not_taken_as_a_command(self):
        v = self.voice()
        v.state = "speaking"
        v._web_result("opening Google", False)
        v.state = "idle"
        v._web_result("opening Google Chrome", True)
        self.assertTrue(v._finals.empty())
        v._web_result("open Spotify", True)
        self.assertEqual(v._finals.get_nowait(), "open Spotify")

    def test_dispatch_runs_clear_commands_and_ignores_chatter(self):
        v = self.voice()
        v._dispatch("open Chrome", "chrome", False)
        v._dispatch("I was telling my friend about the match", "chrome", False)
        self.assertEqual(self.ran, ["open Chrome"])
        self.assertEqual(v.overheard, "I was telling my friend about the match")
        v._dispatch("Computer, why is the sky blue", "chrome", False)
        self.assertEqual(self.ran[-1], "why is the sky blue")


class VoiceFrontEndTests(unittest.TestCase):
    def test_wake_word(self):
        self.assertEqual(split_wake_word("Jarvis, open Chrome."), "open Chrome")
        self.assertEqual(split_wake_word("Hey Jarvis play some music"), "play some music")
        self.assertEqual(split_wake_word("Jarvis."), "")
        self.assertIsNone(split_wake_word("I was telling my friend about Jarvis yesterday"))
        self.assertIsNone(split_wake_word("open Chrome"))

    def cut(self, flags):
        seg, found = Segmenter(), []
        for is_speech in flags:
            u = seg.feed(np.zeros(BLOCK, np.float32), is_speech)
            if u is not None:
                found.append(u)
        return found

    def test_segmenter_cuts_one_utterance_out_of_silence(self):
        per_second = RATE // BLOCK
        found = self.cut([False] * per_second + [True] * per_second + [False] * 2 * per_second)
        self.assertEqual(len(found), 1)
        self.assertGreater(len(found[0]) / RATE, 1.0)
        self.assertLess(len(found[0]) / RATE, 2.4)

    def test_segmenter_keeps_a_sentence_together_across_short_pauses(self):
        word, gap = [True] * 12, [False] * 8
        found = self.cut([False] * 10 + (word + gap) * 4 + [False] * 40)
        self.assertEqual(len(found), 1)

    def test_segmenter_ignores_a_click_or_cough(self):
        self.assertEqual(self.cut([False] * 30 + [True] * 4 + [False] * 60), [])


if __name__ == "__main__":
    unittest.main()
