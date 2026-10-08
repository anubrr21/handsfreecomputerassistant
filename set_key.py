import re
import tkinter as tk
from tkinter import messagebox, simpledialog

from hfc.config import ROOT

ENV = ROOT / ".env"

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    key = simpledialog.askstring("Hands Free Computer", "Paste your Anthropic API key, then press OK:", show="*", parent=root)
    if key and key.strip():
        text = ENV.read_text(encoding="utf-8") if ENV.exists() else ""
        line = "ANTHROPIC_API_KEY=" + key.strip()
        if re.search(r"^ANTHROPIC_API_KEY=.*$", text, flags=re.M):
            text = re.sub(r"^ANTHROPIC_API_KEY=.*$", lambda m: line, text, flags=re.M)
        else:
            text = text.rstrip("\n") + "\n" + line + "\n"
        ENV.write_text(text, encoding="utf-8")
        messagebox.showinfo("Hands Free Computer", "Saved. Restart the app to switch the assistant on.", parent=root)
    root.destroy()
