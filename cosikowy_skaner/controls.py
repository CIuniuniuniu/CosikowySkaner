"""Panel ustawień pod obrazem z kamery."""

from functools import partial

import cv2
import numpy as np

from .filters import clamp
from .overlay import (
    COL_BLUE, COL_DARK, COL_GREEN, COL_PANEL, COL_SUB, COL_TEXT, COL_TRACK,
    fill_rounded, outline_rounded, ptext, text_size,
)

PANEL_HEIGHT = 58
SETTINGS_WINDOW = "CosikowySkaner - Ustawienia"
GESTURES = (
    ("Mysz", (("mouse", "Sterowanie mysza"), ("left", "Lewy klik"),
              ("right", "Prawy klik"), ("scroll", "Przewijanie"))),
    ("Muzyka", (("play", "Play / Pauza"), ("previous", "Poprzedni utwor"),
                ("next", "Nastepny utwor"), ("volume", "Glosnosc"))),
    ("Pozostale", (("toggle", "Gest ON / OFF"), ("listen", "Skrot do Spotify"))),
)
HAND_LABELS = ("P", "L", "PL")
SLIDERS = (
    ("Czulosc", "sens", 3, 30),
    ("Przyspieszenie", "accel", 0, 10),
    ("Wygladzanie", "smooth", 0, 10),
)
SURFACE = (37, 30, 24)


class Controls:
    def __init__(self, controller):
        self.controller = controller
        self.buttons = []
        self.sliders = []
        self.dragged = None

    def draw(self, image, top, open_settings):
        width = image.shape[1]
        image[top:] = COL_DARK
        cv2.line(image, (0, top), (width - 1, top), COL_TRACK)
        self.buttons.clear()
        enabled = self.controller.enabled
        self._button(image, (14, top + 12, 112, top + 46),
                     "Wlaczone" if enabled else "Wylaczone", enabled, COL_GREEN, self._toggle)
        self._button(image, (width - 142, top + 12, width - 14, top + 46),
                     "Ustawienia", False, COL_BLUE, open_settings)
        ptext(image, "Mysz wlaczona" if self.controller.gestures["mouse"] else "Mysz wylaczona",
              128, top + 34, 0.45, COL_SUB)

    def _toggle(self):
        self.controller.set_enabled(not self.controller.enabled, "panel")

    def draw_settings(self, image):
        image[:] = COL_DARK
        self.buttons.clear()
        self.sliders.clear()
        ptext(image, "Ustawienia", 20, 41, 0.83)
        left = 20
        label_y = 80
        row_y = 91
        ptext(image, "Reka", left, label_y, 0.46, COL_SUB)
        for mode, label in enumerate(HAND_LABELS):
            x = left + mode * 35
            self._button(
                image, (x, row_y, x + 31, row_y + 33),
                label, mode == self.controller.hand_mode, COL_BLUE, partial(self.controller.set_hand_mode, mode),
            )

        left += 114
        cell = (image.shape[1] - left - 14) // len(SLIDERS)
        for index, (label, attr, low, high) in enumerate(SLIDERS):
            x = left + index * cell
            right = x + cell - 13
            value = getattr(self.controller, attr)
            raw = round(value * 10) if attr == "sens" else value
            caption = f"{value:.1f}x" if attr == "sens" else str(value)
            ptext(image, label, x, label_y, 0.44, COL_SUB)
            ptext(image, caption, right - text_size(caption, 0.45)[0], label_y, 0.45, COL_TEXT)
            cy = row_y + 16
            thumb = x + round((right - x) * (raw - low) / (high - low))
            cv2.line(image, (x, cy), (right, cy), COL_TRACK, 4, cv2.LINE_AA)
            cv2.line(image, (x, cy), (thumb, cy), COL_BLUE, 4, cv2.LINE_AA)
            cv2.circle(image, (thumb, cy), 7, COL_BLUE, -1, cv2.LINE_AA)
            self.sliders.append((attr, low, high, x, right, row_y, row_y + 33))

        y = 155
        for title, group in GESTURES:
            ptext(image, title, 20, y, 0.57, COL_SUB)
            y += 13
            for index, (key, label) in enumerate(group):
                x = 20 + (index % 2) * 336
                top = y + (index // 2) * 48
                enabled = self.controller.gestures[key]
                available = key not in ("left", "right", "scroll") or self.controller.gestures["mouse"]
                caption = f"{label}   {'ON' if enabled and available else 'OFF'}"
                action = partial(self.controller.set_gesture, key, not enabled) if available else None
                self._button(image, (x, top, x + 324, top + 37), caption,
                             enabled and available, COL_BLUE, action)
            y += ((len(group) + 1) // 2) * 48 + 16

    def _button(self, image, box, label, selected, color, action):
        x, y, right, bottom = box
        fill_rounded(image, (x, y), (right, bottom), color if selected else SURFACE, 8)
        if not selected:
            outline_rounded(image, (x, y), (right, bottom), COL_TRACK, 8)
        tw, th = text_size(label, 0.42)
        ptext(image, label, (x + right - tw) // 2, (y + bottom + th) // 2,
              0.42, COL_PANEL if selected else COL_TEXT)
        if action is not None:
            self.buttons.append((box, action))

    def on_mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONUP:
            self.dragged = None
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.dragged = None
            for (left, top, right, bottom), action in self.buttons:
                if left <= x <= right and top <= y <= bottom:
                    action()
                    return
            for slider in self.sliders:
                _, _, _, left, right, top, bottom = slider
                if left - 5 <= x <= right + 5 and top <= y <= bottom:
                    self.dragged = slider
                    self._slide(x)
                    return
        elif event == cv2.EVENT_MOUSEMOVE and self.dragged:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                self._slide(x)
            else:
                self.dragged = None

    def _slide(self, x):
        attr, low, high, left, right, _, _ = self.dragged
        value = round(low + (high - low) * clamp((x - left) / (right - left), 0, 1))
        setattr(self.controller, attr, value / 10 if attr == "sens" else value)


class SettingsWindow:
    def __init__(self, controller):
        self.controls = Controls(controller)
        self.image = np.empty((465, 704, 3), dtype=np.uint8)
        self.opened = False

    def open(self):
        if self.opened:
            return
        cv2.namedWindow(SETTINGS_WINDOW)
        cv2.setMouseCallback(SETTINGS_WINDOW, self.controls.on_mouse)
        self.opened = True
        self.controls.draw_settings(self.image)
        cv2.imshow(SETTINGS_WINDOW, self.image)

    def refresh(self):
        if not self.opened:
            return
        try:
            visible = cv2.getWindowProperty(SETTINGS_WINDOW, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            visible = 0
        if visible < 1:
            self.controls.dragged = None
            self.opened = False
            return
        self.controls.draw_settings(self.image)
        cv2.imshow(SETTINGS_WINDOW, self.image)
