"""Simple window for the vision assistant: camera view, what is being said, and buttons.

Opened by vlm_loop.py --ui (run_all.py / start.sh do this automatically on the desktop).
Needs Tkinter:  sudo apt install -y python3-tk

Buttons (keyboard shortcuts in brackets):
  Navigate [N]   Describe [D] / [R]   More [M]   Stop talking [Space / Esc]   Quit [Q]
"""
import os
import sys
import time
import tkinter as tk
from tkinter import font as tkfont

import cv2

BG, PANEL, FG, MUTED = "#15171a", "#1f2226", "#f2f2f2", "#9aa0a6"
ALERT, NAV_C, DESC_C, OK_C = "#ff5252", "#4fc3f7", "#b39ddb", "#66bb6a"
VIEW_W, VIEW_H = 800, 450
STALE_S = 2.0


class App(tk.Tk):
    def __init__(self, core, view_path=None, frame_source=None):
        """core = the running vlm_loop module (speaker, set_mode, ... live in it).
        view_path = image file to show (refreshed when it changes); otherwise frame_source()."""
        super().__init__()
        self.core, self.view_path, self.frame_source = core, view_path, frame_source
        self.title("VisionAssist")
        self.configure(bg=BG)
        self.minsize(1000, 680)
        self._photo = None
        self._view_mtime = 0.0
        self._speaking_text = ""

        big = tkfont.Font(family="DejaVu Sans", size=20, weight="bold")
        mid = tkfont.Font(family="DejaVu Sans", size=14)
        btn = tkfont.Font(family="DejaVu Sans", size=16, weight="bold")
        mono = tkfont.Font(family="DejaVu Sans Mono", size=12)

        # ---- header: mode + sensing status
        head = tk.Frame(self, bg=BG)
        head.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(10, 4))
        self.mode_lbl = tk.Label(head, text="STARTING", font=big, bg=BG, fg=MUTED, padx=12, pady=4)
        self.mode_lbl.pack(side="left")
        self.status_lbl = tk.Label(head, text="", font=mid, bg=BG, fg=MUTED)
        self.status_lbl.pack(side="right")

        # ---- camera view
        self.view_box = tk.Frame(self, bg="black", width=VIEW_W, height=VIEW_H)
        self.view_box.grid(row=1, column=0, sticky="nsew", padx=(12, 6), pady=6)
        self.view_box.pack_propagate(False)              # size in pixels, not text characters
        self.view = tk.Label(self.view_box, bg="black", fg=MUTED, font=mid, text="Waiting for camera...")
        self.view.pack(fill="both", expand=True)

        # ---- buttons
        side = tk.Frame(self, bg=BG)
        side.grid(row=1, column=1, sticky="ns", padx=(6, 12), pady=6)
        buttons = [
            ("Navigate  [N]", NAV_C, lambda: core.set_mode(core.NAV)),
            ("Describe  [D]", DESC_C, lambda: core.set_mode(core.DESCRIBE)),
            ("More  [M]", DESC_C, core.request_more),
            ("Stop talking  [Space]", "#ffb74d", core.hush),
            ("Quit  [Q]", "#78909c", self.quit_app),
        ]
        for text, color, cmd in buttons:
            tk.Button(side, text=text, command=cmd, font=btn, bg=color, fg="#101010",
                      activebackground=FG, relief="flat", width=18, height=2,
                      cursor="hand2").pack(fill="x", pady=5)

        # ---- now speaking
        tk.Label(side, text="Now speaking", font=mid, bg=BG, fg=MUTED, anchor="w").pack(fill="x", pady=(16, 0))
        self.now_lbl = tk.Label(side, text="-", font=mid, bg=PANEL, fg=FG, wraplength=300,
                                justify="left", anchor="nw", padx=10, pady=10, height=4)
        self.now_lbl.pack(fill="x")

        # ---- log of everything said
        logf = tk.Frame(self, bg=BG)
        logf.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=12, pady=(4, 12))
        self.log = tk.Text(logf, height=10, bg=PANEL, fg=FG, font=mono, wrap="word",
                           relief="flat", padx=8, pady=6, state="disabled")
        scroll = tk.Scrollbar(logf, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        for tag, color in (("alert", ALERT), ("nav", NAV_C), ("narration", FG),
                           ("system", MUTED), ("cut", "#ffb74d")):
            self.log.tag_configure(tag, foreground=color)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)

        # ---- keys
        for key, fn in (("n", lambda: core.set_mode(core.NAV)), ("d", lambda: core.set_mode(core.DESCRIBE)),
                        ("r", lambda: core.set_mode(core.DESCRIBE)), ("m", core.request_more),
                        ("<space>", core.hush), ("<Escape>", core.hush), ("q", self.quit_app)):
            self.bind(key if key.startswith("<") else f"<KeyPress-{key}>", lambda e, f=fn: f())
        self.protocol("WM_DELETE_WINDOW", self.quit_app)

        self.after(100, self._tick_events)
        self.after(100, self._tick_view)
        self.after(500, self._tick_status)

    # ---------- periodic updates ----------
    def _tick_events(self):
        q = self.core.UI_EVENTS
        try:
            while q is not None and not q.empty():
                t, kind, text, prio = q.get_nowait()
                self._on_event(t, kind, text, prio)
        finally:
            if self._speaking_text and self.core.speaker.idle():
                self._speaking_text = ""
                self.now_lbl.configure(text="-", fg=FG)
            self.after(100, self._tick_events)

    def _on_event(self, t, kind, text, prio):
        stamp = time.strftime("%H:%M:%S", time.localtime(t))
        if kind == "speak":
            announce = text.endswith(" mode.")            # "Description mode." etc.
            alert = prio > self.core.INFO and not announce
            tag = ("system" if announce else "alert" if alert else
                   "narration" if self.core.current_mode() == self.core.DESCRIBE else "nav")
            self._log(f"{stamp}  {'! ' if alert else '  '}{text}", tag)
            self._speaking_text = text
            self.now_lbl.configure(text=text, fg=ALERT if alert else FG)
        elif kind == "interrupted":
            self._log(f"{stamp}     (cut off)", "cut")
        elif kind == "mode":
            desc = text == self.core.DESCRIBE
            self.mode_lbl.configure(text=text.upper(), bg=DESC_C if desc else NAV_C, fg="#101010")
            self._log(f"{stamp}  [{text} mode]", "system")
        elif kind == "status":
            self._log(f"{stamp}  [{text}]", "system")

    def _log(self, line, tag):
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n", tag)
        if int(self.log.index("end-1c").split(".")[0]) > 500:   # keep the last 500 lines
            self.log.delete("1.0", "50.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _tick_view(self):
        try:
            path = self.view_path
            if path and os.path.exists(path):
                mtime = os.path.getmtime(path)
                if mtime != self._view_mtime:
                    self._view_mtime = mtime
                    frame = cv2.imread(path)
                    if frame is not None:
                        self._show(frame)
                elif time.time() - mtime > STALE_S:
                    self.view.configure(image="", text="Camera feed stopped", fg=ALERT)
                    self._photo = None
            elif not path and self.frame_source is not None:
                frame = self.frame_source()
                if frame is not None:
                    self._show(frame)
        finally:
            self.after(80, self._tick_view)

    def _show(self, frame):
        w = max(320, self.view_box.winfo_width())
        h = max(180, self.view_box.winfo_height())
        fh, fw = frame.shape[:2]
        scale = min(w / fw, h / fh)
        frame = cv2.resize(frame, (max(1, int(fw * scale)), max(1, int(fh * scale))),
                           interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ppm = b"P6 %d %d 255\n" % (rgb.shape[1], rgb.shape[0]) + rgb.tobytes()
        self._photo = tk.PhotoImage(data=ppm, format="PPM")
        self.view.configure(image=self._photo, text="")

    def _tick_status(self):
        try:
            fp = sys.modules.get("fastpath")
            inst = getattr(fp, "_instance", None) if fp else None
            if inst is None:
                text, color = "Hazard sensing: off", MUTED
            elif inst.latest is None:
                text, color = "Hazard sensing: waiting...", MUTED
            else:
                seen, rec = inst.latest
                age = time.time() - seen
                if age > STALE_S:
                    text, color = f"Hazard sensing: LOST ({age:.0f} s)", ALERT
                else:
                    fps = rec.get("fps")
                    n = len(rec.get("objects") or [])
                    text = (f"Hazard sensing: OK  |  {fps if fps is not None else '?'} fps  |  "
                            f"{n} object{'' if n == 1 else 's'}")
                    color = OK_C
            self.status_lbl.configure(text=text, fg=color)
        finally:
            self.after(500, self._tick_status)

    def quit_app(self):
        self.core.hush()
        self.destroy()
