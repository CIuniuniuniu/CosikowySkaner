"""Panel ustawień pod obrazem z kamery."""

import cv2

from .filters import clamp
from .overlay import (
    COL_BLUE, COL_DARK, COL_GREEN, COL_PANEL, COL_SUB, COL_TEXT, COL_TRACK,
    fill_rounded, outline_rounded, ptext, text_size,
)

PANEL_HEIGHT = 96
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

    def draw(self, image, top):
        width = image.shape[1]
        image[top:] = COL_DARK
        cv2.line(image, (0, top), (width - 1, top), COL_TRACK)
        self.buttons.clear()
        self.sliders.clear()

        left = 14
        label_y = top + 27
        row_y = top + 43
        ptext(image, "Sterowanie", left, label_y, 0.46, COL_SUB)
        enabled = self.controller.enabled
        self._button(
            image, (left, row_y, left + 87, row_y + 33),
            "Wlaczone" if enabled else "Wylaczone", enabled, COL_GREEN, "toggle",
        )

        left += 103
        ptext(image, "Reka", left, label_y, 0.46, COL_SUB)
        for mode, label in enumerate(HAND_LABELS):
            x = left + mode * 35
            self._button(
                image, (x, row_y, x + 31, row_y + 33),
                label, mode == self.controller.hand_mode, COL_BLUE, mode,
            )

        left += 114
        cell = (width - left - 14) // len(SLIDERS)
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

    def _button(self, image, box, label, selected, color, action):
        x, y, right, bottom = box
        fill_rounded(image, (x, y), (right, bottom), color if selected else SURFACE, 8)
        if not selected:
            outline_rounded(image, (x, y), (right, bottom), COL_TRACK, 8)
        tw, th = text_size(label, 0.42)
        ptext(image, label, (x + right - tw) // 2, (y + bottom + th) // 2,
              0.42, COL_PANEL if selected else COL_TEXT)
        self.buttons.append((box, action))

    def on_mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONUP:
            self.dragged = None
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.dragged = None
            for (left, top, right, bottom), action in self.buttons:
                if left <= x <= right and top <= y <= bottom:
                    if action == "toggle":
                        self.controller.set_enabled(not self.controller.enabled, "panel")
                    else:
                        self.controller.set_hand_mode(action)
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
