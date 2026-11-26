"""Ilustrowana ściąga gestów dostępna z podglądu."""

import cv2
import numpy as np

from .windows_ui import ImageWindow

from .overlay import COL_BLUE, COL_DARK, COL_PANEL, COL_SUB, COL_TRACK, fill_rounded, ptext

WINDOW = "CosikowySkaner - Gesty"

# Wyprostowane palce oznaczamy neutralnym kolorem, miejsce złączenia akcentem.
CARDS = (
    ("Ruch kursora", ("Przesuwaj dlon", "jak myszke."), "move"),
    ("Pauza kursora", ("Zacisnij piesc,", "aby zatrzymac kursor."), "fist"),
    ("Lewy klik", ("Zlacz kciuk", "ze wskazujacym."), "left"),
    ("Prawy klik", ("Zlacz kciuk", "z malym palcem."), "right"),
    ("Przewijanie", ("Pokaz V i ruszaj", "dlonia gora / dol."), "scroll"),
    ("Sterowanie ON / OFF", ("Pokaz rogi obiema", "dlonmi przez ok. 1 s."), "rock"),
    ("Play / Pauza", ("Po 4 proste palce", "i schowane kciuki.", "Obie dlonie, ok. 1 s."), "play"),
    ("Poprzedni utwor", ("Wskaz palcem w lewo.", "Odtwarzacz moze wrocic", "na poczatek utworu."), "prev"),
    ("Nastepny utwor", ("Wskaz palcem w prawo.", "Pozostale palce zegnij."), "next"),
    ("Glosnosc", ("Lewa dlon: trzy palce.", "Prawa gora / dol.", "Z mysza wybierz reke P."), "volume"),
    ("Skrot do Spotify", ("Przyloz obie dlonie", "do uszu na ok. 1 s.", "Wlacz wykrywanie twarzy."), "listen"),
)

PALM = ((-10, 9), (-13, -6), (-8, -12), (9, -12), (14, -5), (10, 10), (6, 16))
STRAIGHT = (
    ((-9, -10), (-11, -25), (-10, -33)),
    ((-3, -12), (-3, -32), (-2, -39)),
    ((3, -12), (4, -29), (5, -35)),
    ((9, -10), (12, -23), (13, -28)),
)
POSES = {
    "open": (0, 1, 2, 3), "fist": (), "point": (0,),
    "scroll": (0, 1), "rock": (0, 3), "volume": (0, 1, 2), "play": (0, 1, 2, 3),
    "left": (1, 2, 3), "right": (0, 1, 2),
}


def hand(image, x, y, pose, scale=1.0, rotation=0):
    extended = POSES[pose]

    def transform(px, py):
        if rotation:
            px, py = -rotation * py, rotation * px
        return x + round(px * scale), y + round(py * scale)

    def line(points, color=COL_SUB):
        vertices = np.array([transform(*point) for point in points], dtype=np.int32)
        cv2.polylines(image, [vertices], False, color, 2, cv2.LINE_AA)

    line(PALM)
    pinch = {"left": (0, (-12, -17)), "right": (3, (7, -2))}.get(pose)
    for index, finger in enumerate(STRAIGHT):
        if index in extended:
            line(finger)
        elif pinch and index == pinch[0]:
            line((finger[0], (finger[0][0], -21), pinch[1]), COL_BLUE)
        else:
            fx, fy = finger[0]
            line(((fx, fy), (fx, fy - 4), (fx + 2, fy - 5)))
    if pinch:
        line(((-11, 8), (-19, 0), pinch[1]), COL_BLUE)
        cv2.circle(image, transform(*pinch[1]), 3, COL_BLUE, -1, cv2.LINE_AA)
    elif pose not in ("fist", "play"):
        line(((-11, 8), (-19, 0), (-20, -9)))
    else:
        line(((-11, 5), (-4, -4), (4, -4)))


def illustration(image, x, y, pose):
    if pose in ("rock", "play"):
        hand(image, x - 16, y, pose, 0.72)
        hand(image, x + 17, y, pose, 0.72)
    elif pose in ("prev", "next"):
        hand(image, x, y - 11, "point", 0.9, -1 if pose == "prev" else 1)
    elif pose == "listen":
        cv2.ellipse(image, (x, y - 6), (12, 17), 0, 0, 360, COL_BLUE, 2, cv2.LINE_AA)
        hand(image, x - 22, y + 3, "open", 0.49)
        hand(image, x + 23, y + 3, "open", 0.49)
    elif pose == "volume":
        hand(image, x - 14, y, "volume", 0.66)
        hand(image, x + 13, y, "open", 0.55)
        arrow(image, x + 27, y - 20, x + 27, y + 7)
    else:
        hand(image, x - 4, y, "open" if pose == "move" else pose, 0.85)
        if pose == "move":
            arrow(image, x - 20, y + 20, x + 20, y + 20)
        elif pose == "scroll":
            arrow(image, x + 20, y - 28, x + 20, y + 12)


def arrow(image, x1, y1, x2, y2):
    start, end = (x1, y1), (x2, y2)
    cv2.arrowedLine(image, start, end, COL_BLUE, 2, cv2.LINE_AA, tipLength=0.18)
    cv2.arrowedLine(image, end, start, COL_BLUE, 2, cv2.LINE_AA, tipLength=0.18)


def render_tutorial():
    image = np.empty((530, 844, 3), dtype=np.uint8)
    image[:] = COL_DARK
    ptext(image, "Jak sterowac gestami", 20, 33, 0.82)
    ptext(image, "Wlacz sterowanie klawiszem T. Poszczegolne gesty wybierzesz w Ustawieniach.",
          20, 57, 0.48, COL_SUB)
    for index, (title, lines, pose) in enumerate(CARDS):
        row, col = divmod(index, 3)
        x, y = 20 + col * 272, 76 + row * 112
        fill_rounded(image, (x, y), (x + 260, y + 100), COL_PANEL, 10)
        cv2.line(image, (x + 76, y + 16), (x + 76, y + 84), COL_TRACK, 1, cv2.LINE_AA)
        illustration(image, x + 37, y + 56, pose)
        ptext(image, title, x + 86, y + 25, 0.47)
        for line_number, line in enumerate(lines):
            ptext(image, line, x + 86, y + 46 + line_number * 16, 0.42, COL_SUB)
    return image


class TutorialWindow:
    def __init__(self, parent):
        self.parent = parent
        self.window = None

    def open(self):
        if self.window is not None and not self.window.closed:
            self.window.focus()
            return
        self.window = ImageWindow(WINDOW, self.parent)
        self.window.show(render_tutorial())
