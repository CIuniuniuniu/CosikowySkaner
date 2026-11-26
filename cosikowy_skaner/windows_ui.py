"""Okna aplikacji i przekazywanie zdarzeń do kontrolek HUD-u."""

import tkinter as tk

import cv2
from PIL import Image, ImageTk


class ImageWindow:
    def __init__(self, title, parent=None, on_mouse=None):
        self.widget = tk.Tk() if parent is None else tk.Toplevel(parent.widget)
        self.widget.title(title)
        self.widget.protocol("WM_DELETE_WINDOW", self.close)
        self.closed = False
        self.canvas = tk.Canvas(self.widget, bg="#12181f", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.item = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.frame = None
        self.photo = None
        self.viewport = (0, 0, 1, 1)
        self.key = -1
        self.widget.bind("<KeyPress>", self._key_press)
        self.canvas.bind("<Configure>", lambda _: self._paint())
        if on_mouse is not None:
            for sequence, event, flags in (
                ("<ButtonPress-1>", cv2.EVENT_LBUTTONDOWN, cv2.EVENT_FLAG_LBUTTON),
                ("<B1-Motion>", cv2.EVENT_MOUSEMOVE, cv2.EVENT_FLAG_LBUTTON),
                ("<ButtonRelease-1>", cv2.EVENT_LBUTTONUP, 0),
            ):
                self.canvas.bind(sequence, lambda e, k=event, f=flags: self._mouse(on_mouse, e, k, f))

    def _key_press(self, event):
        if event.char:
            self.key = ord(event.char[0].lower())

    def poll(self):
        if not self.closed:
            self.widget.update()
        key, self.key = self.key, -1
        return key

    def show(self, frame):
        first_frame = self.frame is None
        self.frame = frame
        if first_frame:
            h, w = frame.shape[:2]
            self.canvas.configure(width=w, height=h)
            self.widget.update_idletasks()
        self._paint()

    def _paint(self):
        if self.closed or self.frame is None:
            return
        h, w = self.frame.shape[:2]
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        scale = min(width / w, height / h)
        display_w, display_h = max(1, round(w * scale)), max(1, round(h * scale))
        x, y = (width - display_w) // 2, (height - display_h) // 2
        image = Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
        if (display_w, display_h) != (w, h):
            image = image.resize((display_w, display_h), Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(image, master=self.widget)
        self.canvas.coords(self.item, x, y)
        self.canvas.itemconfigure(self.item, image=self.photo)
        self.viewport = (x, y, display_w, display_h)

    def _mouse(self, handler, event, kind, flags):
        if self.frame is None:
            return
        x, y, display_w, display_h = self.viewport
        h, w = self.frame.shape[:2]
        handler(kind, (event.x - x) * w // display_w,
                (event.y - y) * h // display_h, flags, None)

    def focus(self):
        self.widget.deiconify()
        self.widget.lift()

    def close(self):
        if not self.closed:
            self.closed = True
            self.widget.destroy()
