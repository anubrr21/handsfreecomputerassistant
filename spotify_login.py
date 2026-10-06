from hfc.brain import load_env
from hfc.spotify import Spotify, SpotifyError

if __name__ == "__main__":
    load_env()
    spotify = Spotify()
    try:
        spotify.login()
        print("Spotify is connected. Say: play <song> on Spotify.")
    except SpotifyError as e:
        print("Could not connect:", e)
