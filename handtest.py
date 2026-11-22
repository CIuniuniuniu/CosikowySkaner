# ==============================================================
# STEROWANIE MYSZKĄ GESTAMI DŁONI  (Windows)  -  wersja 4.0
#
# Instalacja:   pip install opencv-python mediapipe numpy
# Pliki obok skryptu:  hand_landmarker.task (masz już)
#                      face_landmarker.task (jeśli go brak, program
#                      sam spróbuje go pobrać przy pierwszym starcie)
#
# GESTY
#   Ruch dłoni                      - kursor (względnie, jak w myszce)
#   Pięść (wszystkie palce zgięte)  - pauza kursora ("podniesienie myszki")
#   Kciuk + wskazujący              - lewy przycisk (LPM)
#   Kciuk + mały palec              - prawy przycisk (PPM)
#   ✌ (wskazujący + środkowy)       - scroll (ruch dłoni góra/dół)
#   🤘 OBIEMA dłońmi, ok. 1 s        - ON/OFF sterowania (działa też gdy OFF)
#   Obie dłonie przy uszach, 1 s    - uruchamia Spotify ("słucham")
#   4 palce proste + kciuki schowane
#   w dłoń, OBIEMA dłońmi, ok. 1 s  - PLAY / PAUZA, dokładnie jak klawisz multimedialny
#                                     na klawiaturze (steruje tym, co aktualnie gra, np. Spotify)
#   Palec wskazujący w PRAWO (👉)    - następna piosenka (inne palce zgięte, dłoń w bok)
#   Palec wskazujący w LEWO (👈)     - poprzednia piosenka / początek obecnej (jak klawisz
#                                     "poprzedni": po 3 s utworu wraca na jego początek)
#   LEWA dłoń: TRÓJKA (wskazujący + środkowy + serdeczny proste, mały palec
#   zgięty), trzymana 1 s, przy widocznej prawej dłoni = tryb GŁOŚNOŚĆ;
#   potem PRAWA dłoń w górę = głośniej, w dół = ciszej;
#   opuszczasz lewą trójkę = koniec trybu
#
#   Skróty do aplikacji, głośność, play/pauza i zmiana piosenek działają tylko gdy sterowanie jest ON.
#   Głośność działa tylko przy suwaku "Reka" = P (prawa ręka steruje kursorem).
#
# SKRÓTY DO APLIKACJI: ścieżka do .exe w słowniku APP_PATHS albo skrót (.lnk) w folderze
#   "skroty" obok skryptu; który gest co uruchamia - w słowniku GESTURE_APPS.
#
# KLAWISZE (w oknie podglądu):
#   Q / ESC - wyjście       T - ON/OFF sterowania
#   D - liczby debug        F - skaner twarzy        M - mapka monitorów
# Z DOWOLNEGO MIEJSCA:  Ctrl + Alt + G  - ON/OFF sterowania
#
# SUWAKI W OKNIE:  Reka P/L/PL | Czulosc | Przyspiesz (przyspieszenie przy
#   szybkim ruchu) | Wygladz (mniej drżenia, ale większe opóźnienie) | Wyl/1Wl
#   Ustawienia zapisują się w pliku ustawienia_myszki.json obok skryptu.
# ==============================================================

import os
import math
import time
import json
import ctypes
import threading
import traceback
import urllib.request
from collections import deque
from types import SimpleNamespace

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

IS_WINDOWS = (os.name == "nt")

# ==============================================================
# KONFIGURACJA  (tu możesz kręcić ustawieniami)
# ==============================================================

# --- kamera ---
CAMERA_INDEX = 0
CAMERA_WIDTH = 640         # 640x480 w zupełności wystarcza do dłoni i daje 30 kl./s
CAMERA_HEIGHT = 480        #   (dla 1280x720 ustaw 1280 / 720 - ale wiele kamer da tam mniej kl./s)
CAMERA_FPS = 30            # kamera sama zejdzie niżej, jeśli nie umie tyle
CAMERA_MJPG = True         # tryb MJPG daje zwykle 30-60 kl./s zamiast 10-15
CAMERA_BACKEND = "DSHOW"   # "DSHOW", "MSMF" albo "ANY" - jeśli jest mało fps, spróbuj "MSMF"
CAMERA_EXPOSURE = None     # None = automatyczna ekspozycja (domyślnie)
                           # liczba (np. -6) = ręczna ekspozycja, mniejsza = krótszy czas
                           #   naświetlania = więcej kl./s w słabym świetle (obraz ciemniejszy)
PREVIEW_WIDTH = 960        # szerokość okna podglądu (0 = rozmiar z kamery)

# --- modele (leżą obok skryptu; brakujące zostaną pobrane) ---
HERE = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH = os.path.join(HERE, "hand_landmarker.task")
FACE_MODEL_PATH = os.path.join(HERE, "face_landmarker.task")
SETTINGS_PATH = os.path.join(HERE, "ustawienia_myszki.json")
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                  "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                  "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

HAND_DETECT_CONF = 0.5
HAND_PRESENCE_CONF = 0.4
HAND_TRACK_CONF = 0.4

# --- start i ogólne ---
START_ENABLED = False      # False = po uruchomieniu sterowanie jest wyłączone
SOUND_FEEDBACK = True      # piknięcie przy włączeniu / wyłączeniu sterowania
SWAP_HANDEDNESS = True     # tak jak w starej wersji; jeśli "P" reaguje na lewą rękę -> False
GLOBAL_HOTKEY = True       # Ctrl+Alt+G włącza/wyłącza sterowanie z dowolnego miejsca
FACE_AT_START = True       # skaner twarzy włączony od startu (klawisz F przełącza)
                           #   (gest "słucham" potrzebuje skanera twarzy!)

# --- skróty do aplikacji ---
# Aplikację można podać na dwa sposoby:
#   1) wpisać pełną ścieżkę do .exe w APP_PATHS (np. r"C:\Users\Ty\AppData\Roaming\Spotify\Spotify.exe")
#   2) wrzucić skrót (.lnk) albo .exe o nazwie "spotify" do folderu "skroty" obok skryptu
# Dla Spotify, jeśli nic nie wpiszesz, program sam sprawdzi typowe miejsca instalacji.
APPS_FOLDER = os.path.join(HERE, "skroty")
APP_PATHS = {
    "spotify": r"C:\Users\kaame\AppData\Roaming\Spotify\Spotify.exe",
}
GESTURE_APPS = {
    "listen": "spotify",   # gest "słucham" -> uruchom aplikację o tej nazwie
}
APP_COOLDOWN = 5.0         # tyle sekund przerwy zanim gest może znowu coś uruchomić

# --- domyślne ustawienia suwaków (kolejne uruchomienia pamiętają ostatnie) ---
DEFAULT_HAND_MODE = 0      # 0 = prawa, 1 = lewa, 2 = obie
DEFAULT_SENS = 10          # czułość x10  (10 = 1.0)
DEFAULT_ACCEL = 6          # przyspieszenie 0-10
DEFAULT_SMOOTH = 5         # wygładzanie 0-10

# --- kursor ---
# Jednostka ruchu to "rozmiar dłoni" (nadgarstek -> nasada środkowego palca,
# ok. 10 cm), więc czułość nie zależy od tego, jak daleko od kamery trzymasz rękę.
CURSOR_HZ = 200            # osobny wątek przesuwa kursor 200x/s (kamera daje 30-60)
OUTPUT_TAU = 0.030         # s; wygładzanie wyjścia (więcej = płynniej, ale większe opóźnienie)
G_BASE = 450.0             # px ekranu na 1 rozmiar dłoni przy wolnym ruchu (precyzja)
ACCEL_GAIN_MAX = 5.0       # o ile maks. rośnie czułość przy szybkim ruchu (suwak = 10)
V_LOW, V_HIGH = 0.30, 4.0  # rozmiary dłoni/s: od ilu rośnie przyspieszenie i gdzie jest max
V_GATE_LOW, V_GATE_HIGH = 0.06, 0.20   # wolniej niż to = kursor stoi (ignoruje drżenie)
SPEED_SMOOTH_SECONDS = 0.04
SMOOTH_CUTOFF_RANGE = (4.0, 0.7)       # filtr One Euro: suwak 0 -> 4.0 Hz, suwak 10 -> 0.7 Hz
ONE_EURO_BETA = 4.0        # im więcej, tym mniejsze opóźnienie przy szybkim ruchu
ONE_EURO_D_CUTOFF = 1.0
DPI_SPEED = True           # ta sama "odległość na ekranie" na monitorach o różnym skalowaniu
POINTER_IDS = (5, 9, 13, 17)           # punkt śledzenia = środek kostek dłoni
PALM_IDS = (0, 5, 9, 13, 17)           # środek dłoni (używany w geście "słucham")
MOVE_REARM_SECONDS = 0.10  # przerwa po pięści / scrollu zanim kursor znowu ruszy
CONTROL_GRACE = 0.25       # tyle czekamy na zgubioną dłoń, zanim zresetujemy punkt odniesienia
TRACK_TTL = 0.30           # po tylu sekundach bez wykrycia dłoń przestaje istnieć
MATCH_DISTANCE = 0.25      # (ułamek szerokości kadru) maks. przesunięcie dłoni między klatkami
DUPLICATE_HAND_DISTANCE = 0.05   # dwa wykrycia bliżej siebie = ta sama dłoń

# --- kliknięcia ---
PINCH_THRESHOLD = 0.35         # PPM (kciuk + mały palec)
RELEASE_THRESHOLD = 0.50
LEFT_PINCH_THRESHOLD = 0.24    # LPM (kciuk + wskazujący)
LEFT_RELEASE_THRESHOLD = 0.38
LEFT_APPROACH = 0.50           # kursor zwalnia, gdy palce zbliżają się do kliknięcia
RIGHT_APPROACH = 0.70
SLOW_MIN = 0.12                # minimalna czułość tuż przed kliknięciem
REWIND_SECONDS = 0.20          # przy kliknięciu kursor wraca tam, gdzie był tyle s wcześniej (tam celowałeś),
REWIND_STILL_PX = 20           #   o ile wcześniej stał w miejscu (ruszył się mniej niż tyle px)
REWIND_MAX_PX = 80             #   i o ile cofnięcie nie jest większe niż tyle px
BUTTON_LOST_GRACE = 0.30       # tyle czekamy z puszczeniem przycisku, gdy dłoń na chwilę zniknie

# --- scroll ---
SCROLL_SENSITIVITY = 12000
SCROLL_QUANTUM = 35            # najmniejsza porcja wysyłana do Windows
SCROLL_HOLD_SECONDS = 0.25     # tyle po ✌ trwa blokada PPM i kursora (kciuk bywa wtedy przy małym palcu)

# --- rozpoznawanie palców ---
EXTENDED_RATIO = 1.15          # metoda 1: stosunek odległości od nadgarstka (jak wcześniej)
FOLDED_RATIO = 1.05
EXT_ANGLE_MCP = 120.0          # metoda 2: kąty stawów w 3D (odporne na obrót dłoni)
EXT_ANGLE_PIP = 150.0
EXT_ANGLE_DIP = 140.0
THUMB_TUCKED_RATIO = 0.40      # kciuk schowany: czubek blisko środka dłoni (obraz 2D, K w podglądzie debug)
THUMB_TUCK_3D = 0.48           # kciuk schowany: to samo, ale w 3D (K3d w podglądzie debug)

# --- gest ON/OFF dwiema dłońmi ---
TOGGLE_HOLD_SECONDS = 1.0
HOLD_DRAIN_RATE = 1.5          # jak szybko "ładowanie" opada, gdy gest na chwilę zgubisz
POSE_WINDOW = 0.5              # okno głosowania (s) - pojedyncza zła klatka nic nie psuje
POSE_RATIO = 0.6               # ile klatek w oknie musi mieć daną pozę

# --- play / pauza (4 palce proste + kciuki schowane, obiema dłońmi) ---
PLAYPAUSE_HOLD_SECONDS = 1.0
PLAYPAUSE_ONLY_WHEN_ON = True  # True = gest działa tylko przy sterowaniu ON (jak skróty i głośność);
                               #   False = działa zawsze, także gdy sterowanie jest OFF
PLAYPAUSE_METHOD = "KLAWISZ"   # "KLAWISZ" = dokładnie jak fizyczny klawisz Play/Pause na klawiaturze
                               # "AUTO"    = komenda prosto do okna Spotify; gdy Spotify jest zamknięty,
                               #             program go uruchamia
PLAYPAUSE_AUTOPLAY_AFTER_LAUNCH = True   # po uruchomieniu Spotify od razu włącz muzykę
PLAYPAUSE_AUTOPLAY_DELAY = 3.0           # s zwłoki od pojawienia się okna Spotify do "play"

# --- następna / poprzednia piosenka (palec wskazujący w bok: 👉 / 👈) ---
# Używa tej samej metody co play/pauza (PLAYPAUSE_METHOD) i tego samego przełącznika
# PLAYPAUSE_ONLY_WHEN_ON. Działa dowolną dłonią.
SKIP_HOLD_SECONDS = 0.35       # tyle trzeba trzymać wskazanie, żeby zmienić piosenkę
POINT_HORIZONTAL_RATIO = 1.5   # palec musi być bardziej poziomy niż pionowy (|dx| > ratio * |dy|)
POINT_MIN_LENGTH = 0.45        # minimalna długość palca w obrazie (w rozmiarach dłoni) - odsiewa palec "do kamery"

# --- gest "słucham" (obie dłonie przy uszach) ---
LISTEN_HOLD_SECONDS = 1.0      # tyle trzeba trzymać ręce przy uszach
LISTEN_NEAR = 0.70             # jak blisko ucha musi być środek dłoni (w szerokościach twarzy)
LISTEN_EAR_OUTSET = 0.05       # o ile "ucho" jest przesunięte na zewnątrz od krawędzi twarzy
LISTEN_FACE_MEMORY = 1.5       # s; ile pamiętamy położenie uszu, gdy dłonie zasłonią twarz

# --- głośność dwiema rękami ---
VOLUME_ENABLED = True
VOLUME_HOLD_SECONDS = 1.0      # tyle trzeba trzymać lewą trójkę, żeby włączyć tryb
                               #   (jeśli włącza się przypadkiem - zwiększ)
VOLUME_EXIT_GRACE = 0.25       # tyle s tolerancji na mignięcie lewej trójki w trybie
VOLUME_KEYS_PER_FRAME = 60.0   # ile "kliknięć" głośności na przesunięcie dłoni o całą wysokość
                               #   kadru (1 kliknięcie = 2% głośności Windows)

# --- skaner twarzy ---
FACE_MAX_FPS = 20
FACE_PROC_WIDTH = 640

# --- okno ---
WINDOW_NAME = "Sterowanie myszka"
FONT = cv2.FONT_HERSHEY_SIMPLEX

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# stawy palców: (nasada, staw 2, staw 3, czubek)
FINGERS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

# ==============================================================
# NARZĘDZIA MATEMATYCZNE I FILTRY
# ==============================================================


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(x):
    x = clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def distance(a, b):
    # odległość 2D w znormalizowanych współrzędnych obrazu (jak w starej wersji -
    # progi kliknięć są dobrane właśnie do tej miary)
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def vangle(a, b, c):
    # kąt (w stopniach) w punkcie b między odcinkami b->a i b->c
    v1 = a - b
    v2 = c - b
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    cosv = float(np.dot(v1, v2)) / (n1 * n2)
    return math.degrees(math.acos(clamp(cosv, -1.0, 1.0)))


def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro2D:
    # Filtr One Euro: przy wolnym ruchu mocno wygładza (znika drżenie),
    # przy szybkim prawie nie opóźnia. Prędkość liczymy w "rozmiarach dłoni"
    # na sekundę, więc ustawienia nie zależą od odległości od kamery.

    def __init__(self, min_cutoff=1.5, beta=ONE_EURO_BETA, d_cutoff=ONE_EURO_D_CUTOFF):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self.t = None
        self.raw = None
        self.x = 0.0
        self.y = 0.0
        self.dx = 0.0
        self.dy = 0.0

    def __call__(self, x, y, t, scale=1.0):
        if self.t is None:
            self.t = t
            self.raw = (x, y)
            self.x, self.y = x, y
            return x, y

        dt = max(t - self.t, 1e-3)
        self.t = t

        rdx = (x - self.raw[0]) / dt
        rdy = (y - self.raw[1]) / dt
        self.raw = (x, y)

        ad = _alpha(self.d_cutoff, dt)
        self.dx += ad * (rdx - self.dx)
        self.dy += ad * (rdy - self.dy)

        speed = math.hypot(self.dx, self.dy) / max(scale, 1e-6)
        cutoff = self.min_cutoff + self.beta * speed
        a = _alpha(cutoff, dt)

        self.x += a * (x - self.x)
        self.y += a * (y - self.y)
        return self.x, self.y


class OneEuro1D:
    def __init__(self, min_cutoff=1.2, beta=12.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.t = None
        self.raw = 0.0
        self.x = 0.0
        self.dx = 0.0

    def __call__(self, x, t):
        if self.t is None:
            self.t = t
            self.raw = x
            self.x = x
            return x
        dt = max(t - self.t, 1e-3)
        self.t = t
        rdx = (x - self.raw) / dt
        self.raw = x
        self.dx += _alpha(self.d_cutoff, dt) * (rdx - self.dx)
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        self.x += _alpha(cutoff, dt) * (x - self.x)
        return self.x


class Debounce:
    # Stan zmienia się dopiero po kilku kolejnych klatkach - pojedyncza
    # błędnie rozpoznana klatka nie robi już kliknięć ani pauz.

    def __init__(self, on_frames=2, off_frames=2):
        self.on_frames = on_frames
        self.off_frames = off_frames
        self.state = False
        self.count = 0

    def update(self, raw):
        raw = bool(raw)
        if raw == self.state:
            self.count = 0
        else:
            self.count += 1
            need = self.on_frames if raw else self.off_frames
            if self.count >= need:
                self.state = raw
                self.count = 0
        return self.state


class Voter:
    # Pamięta ostatnie ~0.5 s i mówi "poza jest aktywna", gdy większość
    # klatek ją miała. Gesty trzymane dwiema dłońmi nie gubią się już przez
    # jedną czy dwie źle rozpoznane klatki.

    def __init__(self, window=POSE_WINDOW, ratio=POSE_RATIO, min_samples=3):
        self.window = window
        self.ratio = ratio
        self.min_samples = min_samples
        self.items = deque()

    def add(self, t, value):
        self.items.append((t, bool(value)))
        self._prune(t)

    def _prune(self, now):
        while self.items and now - self.items[0][0] > self.window:
            self.items.popleft()

    def active(self, now):
        self._prune(now)
        n = len(self.items)
        if n < self.min_samples:
            return False
        return sum(1 for _, v in self.items if v) / n >= self.ratio


class HoldCharge:
    # Gest do przytrzymania: pasek "ładuje się", gdy gest jest pokazany,
    # a gdy go na moment zgubisz - tylko powoli opada (zamiast zerować się
    # od razu). update() zwraca True dokładnie raz, gdy pasek się naładuje.

    def __init__(self, hold_seconds, drain_rate=HOLD_DRAIN_RATE):
        self.hold = hold_seconds
        self.drain = drain_rate
        self.charge = 0.0
        self.latched = False
        self.t_prev = None

    def update(self, active, now):
        dt = 0.0 if self.t_prev is None else clamp(now - self.t_prev, 0.0, 0.2)
        self.t_prev = now
        fired = False
        if active:
            self.charge = min(self.hold, self.charge + dt)
            if self.charge >= self.hold and not self.latched:
                self.latched = True
                fired = True
        else:
            self.charge = max(0.0, self.charge - dt * self.drain)
            if self.charge <= 0.0:
                self.latched = False
        return fired

    def progress(self):
        if self.latched:
            return 0.0
        return clamp(self.charge / self.hold, 0.0, 1.0)


class PinchButton:
    # Przycisk myszy sterowany szczypnięciem. Histereza (inny próg wciśnięcia
    # i puszczenia) + potwierdzenie przez 2 klatki + chwila tolerancji na
    # zniknięcie dłoni (przeciąganie nie urywa się przy mrugnięciu kamery).

    def __init__(self, press_thr, release_thr):
        self.press_thr = press_thr
        self.release_thr = release_thr
        self.held = False
        self.cnt = 0
        self.lost_since = None
        # Kliknięcie wymaga PRZEJŚCIA z otwartej dłoni do szczypnięcia. Po każdej
        # blokadzie (scroll, gest dwiema dłońmi, wyłączenie) przycisk jest
        # "rozbrojony" i uzbraja się dopiero, gdy palce są wyraźnie rozwarte -
        # dzięki temu sama poza dłoni (np. kciuk przy małym palcu w ✌) nie klika.
        self.armed = False
        self.arm_cnt = 0

    def disarm(self):
        self.armed = False
        self.arm_cnt = 0

    def update(self, values, now):
        # values: odległości palców (znormalizowane) dla dłoni, które mogą klikać
        # zwraca "down", "up" albo None
        if not self.held:
            self.lost_since = None

            if not self.armed:
                if values and min(values) > self.release_thr:
                    self.arm_cnt += 1
                    if self.arm_cnt >= 2:
                        self.armed = True
                else:
                    self.arm_cnt = 0
                self.cnt = 0
                return None

            if values:
                vmin = min(values)
                if vmin < self.press_thr * 0.6:        # bardzo wyraźne szczypnięcie - od razu
                    self.held = True
                    self.cnt = 0
                    return "down"
                if vmin < self.press_thr:
                    self.cnt += 1
                    if self.cnt >= 2:
                        self.held = True
                        self.cnt = 0
                        return "down"
                else:
                    self.cnt = 0
            else:
                self.cnt = 0
            return None

        if not values:
            if self.lost_since is None:
                self.lost_since = now
            elif now - self.lost_since > BUTTON_LOST_GRACE:
                self.force_release()
                return "up"
            return None

        self.lost_since = None
        if min(values) > self.release_thr:
            self.cnt += 1
            if self.cnt >= 2:
                self.force_release()
                return "up"
        else:
            self.cnt = 0
        return None

    def force_release(self):
        self.held = False
        self.cnt = 0
        self.lost_since = None

# ==============================================================
# WARSTWA SYSTEMOWA (Windows API, monitory, kursor, kamera)
# ==============================================================

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
VK_CONTROL = 0x11
VK_MENU = 0x12      # Alt
VK_G = 0x47
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_PLAY_PAUSE = 0xB3
WM_APPCOMMAND = 0x0319
APPCOMMAND_MEDIA_NEXTTRACK = 11
APPCOMMAND_MEDIA_PREVIOUSTRACK = 12
APPCOMMAND_MEDIA_PLAY_PAUSE = 14
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

if IS_WINDOWS:
    from ctypes import wintypes

    ULONG_PTR = ctypes.c_size_t

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ULONG_PTR),
        ]

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ULONG_PTR),
        ]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [
            ("uMsg", wintypes.DWORD),
            ("wParamL", wintypes.WORD),
            ("wParamH", wintypes.WORD),
        ]

    class _INPUT_UNION(ctypes.Union):
        _fields_ = [
            ("mi", MOUSEINPUT),
            ("ki", KEYBDINPUT),
            ("hi", HARDWAREINPUT),
        ]

    class INPUT(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [
            ("type", wintypes.DWORD),
            ("u", _INPUT_UNION),
        ]


def set_dpi_awareness():
    # Spójne współrzędne (piksele fizyczne) także na monitorach z różnym
    # skalowaniem - dzięki temu kursor nie blokuje się na granicach monitorów.
    # Musi być wywołane przed utworzeniem okna i pierwszym użyciem kursora.
    try:
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


class WinAPI:
    def __init__(self):
        u = ctypes.windll.user32
        self.user32 = u

        u.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
        u.SetCursorPos.restype = wintypes.BOOL
        u.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
        u.GetCursorPos.restype = wintypes.BOOL
        u.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
        u.SendInput.restype = wintypes.UINT
        u.GetAsyncKeyState.argtypes = [ctypes.c_int]
        u.GetAsyncKeyState.restype = ctypes.c_short
        u.keybd_event.argtypes = [wintypes.BYTE, wintypes.BYTE, wintypes.DWORD, ULONG_PTR]
        u.keybd_event.restype = None
        u.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
        u.MapVirtualKeyW.restype = wintypes.UINT

        # okna i procesy (szukanie okna Spotify)
        self._enum_proc_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        u.EnumWindows.argtypes = [self._enum_proc_type, wintypes.LPARAM]
        u.EnumWindows.restype = wintypes.BOOL
        u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        u.GetWindowThreadProcessId.restype = wintypes.DWORD
        u.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        u.GetClassNameW.restype = ctypes.c_int
        u.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        u.GetWindowTextLengthW.restype = ctypes.c_int
        u.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        u.PostMessageW.restype = wintypes.BOOL

        k = ctypes.windll.kernel32
        self.kernel32 = k
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.OpenProcess.restype = wintypes.HANDLE
        k.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)
        ]
        k.QueryFullProcessImageNameW.restype = wintypes.BOOL
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        k.CloseHandle.restype = wintypes.BOOL

    # ---- czas systemowy (dokładniejsze "sleep") ----
    def timer_begin(self):
        try:
            ctypes.windll.winmm.timeBeginPeriod(1)
        except Exception:
            pass

    def timer_end(self):
        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        except Exception:
            pass

    # ---- kursor i przyciski ----
    def get_cursor_pos(self):
        point = wintypes.POINT()
        self.user32.GetCursorPos(ctypes.byref(point))
        return point.x, point.y

    def set_cursor_pos(self, x, y):
        self.user32.SetCursorPos(int(x), int(y))

    def _mouse(self, flags, data=0):
        inp = INPUT()
        inp.type = INPUT_MOUSE
        inp.mi = MOUSEINPUT(0, 0, data & 0xFFFFFFFF, flags, 0, 0)
        self.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    def left_down(self):
        self._mouse(MOUSEEVENTF_LEFTDOWN)

    def left_up(self):
        self._mouse(MOUSEEVENTF_LEFTUP)

    def right_down(self):
        self._mouse(MOUSEEVENTF_RIGHTDOWN)

    def right_up(self):
        self._mouse(MOUSEEVENTF_RIGHTUP)

    def wheel(self, delta):
        self._mouse(MOUSEEVENTF_WHEEL, int(delta))

    def key_down(self, vk):
        return bool(self.user32.GetAsyncKeyState(vk) & 0x8000)

    # ---- głośność ----
    def volume_keys(self, n):
        # n > 0 = głośniej, n < 0 = ciszej. Wysyłamy klawisze multimedialne,
        # więc Windows pokazuje swoje okienko głośności. 1 naciśnięcie = 2%.
        count = min(abs(int(n)), 25)

        if count == 0:
            return

        vk = VK_VOLUME_UP if n > 0 else VK_VOLUME_DOWN
        arr = (INPUT * (count * 2))()

        for i in range(count):
            for j, flags in enumerate((
                KEYEVENTF_EXTENDEDKEY,
                KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP,
            )):
                item = arr[i * 2 + j]
                item.type = INPUT_KEYBOARD
                item.ki = KEYBDINPUT(vk, 0, flags, 0, 0)

        self.user32.SendInput(len(arr), arr, ctypes.sizeof(INPUT))

    def media_key(self, vk):
        # Dokładnie to, co robi fizyczny klawisz multimedialny na klawiaturze
        # (tak samo działa np. pyautogui.press("playpause")): wciśnięcie
        # i puszczenie klawisza, z kodem skanowania.
        scan = self.user32.MapVirtualKeyW(vk, 0) & 0xFF

        self.user32.keybd_event(vk, scan, KEYEVENTF_EXTENDEDKEY, 0)
        time.sleep(0.02)
        self.user32.keybd_event(vk, scan, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)

    def media_play_pause(self):
        self.media_key(VK_MEDIA_PLAY_PAUSE)

    def media_track(self, direction):
        # direction > 0 = następna piosenka, < 0 = poprzednia / początek obecnej
        self.media_key(VK_MEDIA_NEXT_TRACK if direction > 0 else VK_MEDIA_PREV_TRACK)

    # ---- Spotify: komenda prosto do jego okna ----
    def _process_name(self, pid):
        h = self.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)

        if not h:
            return ""

        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(1024)

            if self.kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                return os.path.basename(buf.value).lower()
        finally:
            self.kernel32.CloseHandle(h)

        return ""

    def find_spotify_window(self):
        # Główne okno Spotify (klasa Chrome_WidgetWin_0, niepusty tytuł). Okno
        # może być zminimalizowane albo schowane w zasobniku - nadal je znajdzie.
        found = []

        def callback(hwnd, lparam):
            try:
                pid = wintypes.DWORD()
                self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

                if self._process_name(pid.value) != "spotify.exe":
                    return True

                cls = ctypes.create_unicode_buffer(256)
                self.user32.GetClassNameW(hwnd, cls, 256)

                if (
                    cls.value.startswith("Chrome_WidgetWin")
                    and self.user32.GetWindowTextLengthW(hwnd) > 0
                ):
                    found.append((0 if cls.value == "Chrome_WidgetWin_0" else 1, hwnd))
            except Exception:
                pass

            return True

        self.user32.EnumWindows(self._enum_proc_type(callback), 0)

        if not found:
            return None

        found.sort()
        return found[0][1]

    def spotify_command(self, cmd):
        # True, jeśli udało się wysłać komendę (APPCOMMAND_*) do okna Spotify
        hwnd = self.find_spotify_window()

        if not hwnd:
            return False

        self.user32.PostMessageW(hwnd, WM_APPCOMMAND, 0, cmd << 16)
        return True

    def spotify_play_pause(self):
        return self.spotify_command(APPCOMMAND_MEDIA_PLAY_PAUSE)

    def spotify_track(self, direction):
        return self.spotify_command(
            APPCOMMAND_MEDIA_NEXTTRACK if direction > 0 else APPCOMMAND_MEDIA_PREVIOUSTRACK
        )

    # ---- monitory ----
    def _dpi_scale(self, hmon):
        try:
            dpi_x = ctypes.c_uint()
            dpi_y = ctypes.c_uint()
            hr = ctypes.windll.shcore.GetDpiForMonitor(
                ctypes.c_void_p(hmon), 0, ctypes.byref(dpi_x), ctypes.byref(dpi_y)
            )
            if hr == 0 and dpi_x.value > 0:
                return dpi_x.value / 96.0
        except Exception:
            pass
        return 1.0

    def enum_monitors(self):
        # lista (lewo, góra, prawo, dół, skala_dpi) w pikselach fizycznych
        found = []

        proc_type = ctypes.WINFUNCTYPE(
            wintypes.BOOL,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.RECT),
            ctypes.c_ssize_t,
        )

        def callback(hmon, hdc, rect_ptr, data):
            r = rect_ptr.contents
            found.append((r.left, r.top, r.right, r.bottom, self._dpi_scale(hmon)))
            return True

        proc = proc_type(callback)
        self.user32.EnumDisplayMonitors(None, None, proc, 0)
        return found

    # ---- dźwięk ----
    def beep(self, freq, ms):
        def worker():
            try:
                import winsound
                winsound.Beep(int(freq), int(ms))
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()


# ---- skróty do aplikacji ----

def resolve_app(name):
    # Szuka aplikacji po nazwie: 1) APP_PATHS, 2) folder "skroty", 3) (tylko Spotify)
    # typowe miejsca instalacji. Zwraca ścieżkę albo None.
    key = str(name).lower()

    path = APP_PATHS.get(key, "")

    if path and os.path.exists(path):
        return path

    if os.path.isdir(APPS_FOLDER):
        for fname in sorted(os.listdir(APPS_FOLDER)):
            stem, ext = os.path.splitext(fname)

            if stem.lower() == key and ext.lower() in (".lnk", ".exe", ".url", ".bat"):
                return os.path.join(APPS_FOLDER, fname)

    if key == "spotify":
        candidates = [
            os.path.join(os.environ.get("APPDATA", ""), "Spotify", "Spotify.exe"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WindowsApps", "Spotify.exe"),
        ]

        for c in candidates:
            if c and os.path.exists(c):
                return c

    return None


def launch_app(name):
    path = resolve_app(name)

    if path is None:
        print("Nie znalazlem aplikacji '%s'. Wpisz sciezke w APP_PATHS albo wrzuc skrot do folderu: %s" % (
            name, APPS_FOLDER
        ))
        return False

    try:
        os.startfile(path)
        print("Uruchomiono:", path)
        return True
    except OSError as e:
        print("Nie udalo sie uruchomic %s: %s" % (path, e))
        return False


class Desktop:
    # Geometria pulpitu z wielu monitorów. Kursor jest "przyklejany" do
    # najbliższego punktu należącego do któregoś z monitorów, więc:
    #  - nigdy nie wypada poza ekrany,
    #  - ześlizguje się wzdłuż krawędzi (jak prawdziwa myszka),
    #  - przechodzi między monitorami także wtedy, gdy mają różne rozmiary
    #    i są przesunięte względem siebie (np. środkowy, lewy i górny).

    def __init__(self, monitors):
        self.set(monitors)

    def set(self, monitors):
        mons = [tuple(m) for m in monitors] or [(0, 0, 1920, 1080, 1.0)]
        self.mons = mons
        self.left = min(m[0] for m in mons)
        self.top = min(m[1] for m in mons)
        self.right = max(m[2] for m in mons)
        self.bottom = max(m[3] for m in mons)

    def nearest(self, x, y):
        # (x, y, skala_dpi) najbliższego punktu należącego do któregoś monitora
        best = None
        best_d = None
        for (l, t, r, b, s) in self.mons:
            cx = clamp(x, l, r - 1)
            cy = clamp(y, t, b - 1)
            d = (cx - x) ** 2 + (cy - y) ** 2
            if d == 0.0:
                return x, y, s
            if best_d is None or d < best_d:
                best_d = d
                best = (cx, cy, s)
        return best

    def clamp(self, x, y):
        nx, ny, _ = self.nearest(x, y)
        return nx, ny

    def scale_at(self, x, y):
        return clamp(self.nearest(x, y)[2], 0.75, 4.0)


class CursorDriver(threading.Thread):
    # Osobny wątek, który przesuwa kursor 200x na sekundę. Kamera daje nowe
    # dane tylko 30-60x/s, więc sam ruch dłoni dawałby "schodkowy", szarpany
    # kursor. Tutaj każda porcja ruchu z kamery jest rozkładana na kolejne
    # milisekundy (wygładzanie wykładnicze), a pozycja jest liczona w
    # ułamkach pikseli.

    def __init__(self, api, desktop):
        super().__init__(daemon=True)
        self.api = api
        self.desktop = desktop
        self.lock = threading.Lock()
        self.pos_lock = threading.Lock()
        self.pend_x = 0.0
        self.pend_y = 0.0
        self.fx = None
        self.fy = None
        self.sx = None
        self.sy = None
        self.running = True

    def push(self, dx, dy):
        with self.lock:
            self.pend_x += dx
            self.pend_y += dy

    def cancel(self):
        with self.lock:
            self.pend_x = 0.0
            self.pend_y = 0.0

    def position(self):
        if self.fx is None:
            return None
        return self.fx, self.fy

    def step(self, dt):
        with self.lock:
            k = 1.0 - math.exp(-dt / OUTPUT_TAU)
            mx = self.pend_x * k
            my = self.pend_y * k
            self.pend_x -= mx
            self.pend_y -= my
            if abs(self.pend_x) < 0.005 and abs(self.pend_y) < 0.005:
                self.pend_x = 0.0
                self.pend_y = 0.0

        if mx == 0.0 and my == 0.0:
            return

        with self.pos_lock:
            cx, cy = self.api.get_cursor_pos()

            # Ktoś inny (np. prawdziwa myszka) ruszył kursor - zaczynamy od jego pozycji.
            if self.sx is None or abs(cx - self.sx) > 1 or abs(cy - self.sy) > 1:
                self.fx = float(cx)
                self.fy = float(cy)

            nx, ny = self.desktop.clamp(self.fx + mx, self.fy + my)
            self.fx = nx
            self.fy = ny

            ix = int(round(nx))
            iy = int(round(ny))

            if ix != cx or iy != cy:
                self.api.set_cursor_pos(ix, iy)

            self.sx = ix
            self.sy = iy

    def warp(self, x, y):
        # Natychmiastowe ustawienie kursora w podanym miejscu (bez ruchu "w drodze").
        self.cancel()

        with self.pos_lock:
            nx, ny = self.desktop.clamp(x, y)
            self.fx = nx
            self.fy = ny
            ix = int(round(nx))
            iy = int(round(ny))
            self.api.set_cursor_pos(ix, iy)
            self.sx = ix
            self.sy = iy

    def run(self):
        period = 1.0 / CURSOR_HZ
        last = time.perf_counter()
        nxt = last + period
        errors = 0

        while self.running:
            now = time.perf_counter()
            dt = min(now - last, 0.1)
            last = now

            try:
                self.step(dt)
            except Exception:
                errors += 1
                if errors <= 3:
                    traceback.print_exc()
                time.sleep(0.05)

            nxt += period
            delay = nxt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.perf_counter()

    def stop(self):
        self.running = False
        self.join(1.0)


def camera_backend_id():
    if not IS_WINDOWS:
        return cv2.CAP_ANY

    name = str(CAMERA_BACKEND).upper()

    if name == "MSMF":
        return cv2.CAP_MSMF
    if name == "ANY":
        return cv2.CAP_ANY

    return cv2.CAP_DSHOW


class CameraThread(threading.Thread):
    # Kamera czyta klatki w osobnym wątku i zawsze trzyma tylko najnowszą -
    # dzięki temu opóźnienie nie rośnie, gdy przetwarzanie chwilę potrwa.
    #
    # Inicjalizacja jest ostrożna: po ustawieniu parametrów sprawdzamy, co
    # kamera naprawdę przyjęła. Jeśli prosiliśmy o MJPG, a sterownik został
    # przy surowym YUY2 (to zwykle właśnie daje ~10 kl./s), próbujemy
    # ustawić parametry w innej kolejności.

    def __init__(self, index, width, height, fps):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(index, camera_backend_id())
        self.width = width
        self.height = height
        self.target_fps = fps

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.opened = self.cap.isOpened()
        self.mode_text = "?"

        if self.opened:
            self._configure()

        self.cond = threading.Condition()
        self.frame = None
        self.seq = 0
        self.stamp = 0.0
        self.running = True
        self.failed = False
        self.fps = 0.0

    def _fourcc_text(self):
        try:
            code = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            text = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
            text = text.strip("\x00 ")
            return text if text.isprintable() and text else "?"
        except Exception:
            return "?"

    def _apply(self, mjpg_first):
        cap = self.cap
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        if CAMERA_MJPG and mjpg_first:
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if CAMERA_MJPG and not mjpg_first:
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        if CAMERA_EXPOSURE is not None:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)      # tryb ręczny (DirectShow)
            cap.set(cv2.CAP_PROP_EXPOSURE, float(CAMERA_EXPOSURE))

    def _configure(self):
        self._apply(mjpg_first=True)
        fcc = self._fourcc_text()

        if CAMERA_MJPG and fcc != "MJPG":
            # MJPG nie został przyjęty - druga próba w innej kolejności
            self._apply(mjpg_first=False)
            fcc = self._fourcc_text()

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = float(self.cap.get(cv2.CAP_PROP_FPS))

        self.mode_text = "%s %dx%d" % (fcc, w, h)

        print("Kamera: %dx%d  fps(deklarowane)=%.1f  format=%s  backend=%s" % (
            w, h, f, fcc, str(CAMERA_BACKEND).upper()
        ))

        if CAMERA_MJPG and fcc != "MJPG":
            print("UWAGA: kamera nie przyjela MJPG (format: %s). To zwykle oznacza ~10 kl./s." % fcc)
            print("       Sprobuj CAMERA_BACKEND = \"MSMF\" albo mniejszej rozdzielczosci (640x480).")

    def run(self):
        fails = 0
        last = None

        while self.running:
            ok, frame = self.cap.read()

            if not ok or frame is None:
                fails += 1
                if fails > 100:
                    self.failed = True
                    break
                time.sleep(0.02)
                continue

            fails = 0
            t = time.perf_counter()

            if last is not None and t > last:
                self.fps += 0.1 * (1.0 / (t - last) - self.fps)
            last = t

            with self.cond:
                self.frame = frame
                self.seq += 1
                self.stamp = t
                self.cond.notify_all()

        with self.cond:
            self.cond.notify_all()

    def wait_frame(self, last_seq, timeout=0.5):
        with self.cond:
            self.cond.wait_for(
                lambda: self.seq != last_seq or self.failed or not self.running,
                timeout
            )
            return self.frame, self.seq, self.stamp

    def stop(self):
        self.running = False
        self.join(1.0)
        self.cap.release()


def ensure_model(path, url):
    # Zwraca True, jeśli plik modelu istnieje (po ewentualnym pobraniu).
    if os.path.exists(path) and os.path.getsize(path) > 100000:
        return True

    name = os.path.basename(path)
    tmp = path + ".part"

    try:
        print("Pobieram model:", name, "...")
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
        print("Model pobrany:", name)
        return True
    except Exception as e:
        print("Nie udalo sie pobrac modelu %s: %s" % (name, e))
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return False


def load_settings():
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_settings(data):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

# ==============================================================
# ANALIZA DŁONI I ŚLEDZENIE
# ==============================================================


def world_array(world):
    # punkty 3D w metrach (MediaPipe "world landmarks") jako tablica 21x3
    try:
        arr = np.array([[p.x, p.y, p.z] for p in world], dtype=np.float32)
        return arr if arr.shape == (21, 3) else None
    except Exception:
        return None


def finger_ratio(hand, tip_id, pip_id):
    # > EXTENDED_RATIO = palec prosty, < FOLDED_RATIO = palec zgięty (obraz 2D)
    wrist = hand[0]
    d_pip = distance(hand[pip_id], wrist)

    if d_pip <= 0:
        return 1.0

    return distance(hand[tip_id], wrist) / d_pip


def analyze_hand(hand, world=None, left_held=False, right_held=False):
    # Każdy gest jest rozpoznawany dwiema metodami:
    #  1) stosunki odległości w obrazie 2D (jak w starej wersji),
    #  2) kąty stawów w 3D - działają też, gdy dłoń jest obrócona lub pochylona.
    # Gesty dwiema dłońmi akceptują wynik którejkolwiek z nich (łatwiej je
    # pokazać), a przytrzymanie i tak chroni przed przypadkowym wyzwoleniem.
    wrist = hand[0]
    palm_size = distance(wrist, hand[5])

    info = {
        "left_norm": 99.0,
        "right_norm": 99.0,
        "index": 1.0,
        "middle": 1.0,
        "ring": 1.0,
        "pinky": 1.0,
        "thumb_tucked": 99.0,
        "thumb_tuck3d": 99.0,
        "ext_a": {name: False for name in FINGERS},
        "peace": False,
        "fist": False,
        "rock": False,
        "three": False,
        "four_tucked": False,
        "point": False,
        "point_dir": 0,
    }

    if palm_size <= 0:
        return info

    # Odległości kciuka od wskazującego / małego palca (względem rozmiaru dłoni)
    info["left_norm"] = distance(hand[4], hand[8]) / palm_size
    info["right_norm"] = distance(hand[4], hand[20]) / palm_size

    # --- metoda 1: stosunki odległości (2D) ---
    ratios = {}
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        ratios[name] = finger_ratio(hand, tip, pip)
        info[name] = ratios[name]

    ext_r = {name: r > EXTENDED_RATIO for name, r in ratios.items()}
    fold_r = {name: r < FOLDED_RATIO for name, r in ratios.items()}

    # Odległość czubka kciuka od środka dłoni (mała = kciuk schowany)
    mid_x = (hand[0].x + hand[9].x) / 2
    mid_y = (hand[0].y + hand[9].y) / 2

    info["thumb_tucked"] = (
        ((hand[4].x - mid_x) ** 2 + (hand[4].y - mid_y) ** 2) ** 0.5
        / palm_size
    )

    # --- metoda 2: kąty stawów w 3D ---
    ext_a = {name: False for name in FINGERS}
    tuck3d = 99.0

    if world is not None:
        centre = world[[0, 5, 9, 13, 17]].mean(axis=0)
        palm_len = float(np.linalg.norm(world[9] - world[0]))

        if palm_len > 1e-6:
            tuck3d = float(np.linalg.norm(world[4] - centre)) / palm_len

        for name, (mcp, pip, dip, tip) in FINGERS.items():
            a_mcp = vangle(world[0], world[mcp], world[pip])
            a_pip = vangle(world[mcp], world[pip], world[dip])
            a_dip = vangle(world[pip], world[dip], world[tip])
            ext_a[name] = (
                a_mcp > EXT_ANGLE_MCP
                and a_pip > EXT_ANGLE_PIP
                and a_dip > EXT_ANGLE_DIP
            )

    info["ext_a"] = ext_a
    info["thumb_tuck3d"] = tuck3d

    # SCROLL (✌) = wskazujący i środkowy wyprostowane, serdeczny zgięty
    index_up = hand[8].y < hand[6].y
    middle_up = hand[12].y < hand[10].y
    ring_folded = hand[16].y > hand[14].y

    info["peace"] = index_up and middle_up and ring_folded

    # 🤘 = wskazujący i mały proste, środkowy i serdeczny zgięte (kciuk dowolnie)
    rock_r = (
        ext_r["index"] and ext_r["pinky"]
        and fold_r["middle"] and fold_r["ring"]
    )
    rock_a = (
        ext_a["index"] and ext_a["pinky"]
        and not ext_a["middle"] and not ext_a["ring"]
    )
    info["rock"] = bool(rock_r or rock_a)

    # 4 palce proste + kciuk schowany w środek dłoni (play / pauza).
    # Metoda 3D wymaga prostego małego palca, żeby trójka nie była brana za to.
    four_r = all(ext_r.values()) and info["thumb_tucked"] < THUMB_TUCKED_RATIO
    four_a = (
        sum(ext_a.values()) >= 3
        and ext_a["index"] and ext_a["pinky"]
        and tuck3d < THUMB_TUCK_3D
    )
    info["four_tucked"] = bool(four_r or four_a)

    # WSKAZANIE (👉 / 👈) = wskazujący prosty, środkowy, serdeczny i mały zgięte
    # (kciuk dowolnie). Kierunek (lewo/prawo) ustala HandTracker, bo potrzebuje
    # rozmiaru kadru.
    index_ext = ext_r["index"] or ext_a["index"]
    others_down = all(
        (not ext_r[n]) and (not ext_a[n]) for n in ("middle", "ring", "pinky")
    )
    info["point"] = bool(index_ext and others_down)

    # TRÓJKA = wskazujący, środkowy i serdeczny proste, mały palec zgięty
    # (kciuk dowolnie). Mała, celowa poza - zwykła otwarta dłoń jej nie włączy.
    three_up_r = ext_r["index"] and ext_r["middle"] and ext_r["ring"]
    three_up_a = ext_a["index"] and ext_a["middle"] and ext_a["ring"]
    pinky_down = (not ext_r["pinky"]) and (not ext_a["pinky"])
    info["three"] = bool((three_up_r or three_up_a) and pinky_down)

    # PIĘŚĆ = wszystkie 4 palce zgięte. Szczypanie ma pierwszeństwo,
    # żeby kliknięcie z zagiętymi palcami nie było brane za pięść.
    left_limit = LEFT_RELEASE_THRESHOLD if left_held else LEFT_PINCH_THRESHOLD
    right_limit = RELEASE_THRESHOLD if right_held else PINCH_THRESHOLD

    in_pinch = (
        info["left_norm"] < left_limit
        or info["right_norm"] < right_limit
    )

    info["fist"] = all(fold_r.values()) and not in_pinch

    return info


class HandTrack:
    # Jedna "śledzona" dłoń z własną historią (głosowanie pozy, etykieta
    # lewa/prawa uśredniona z kilku klatek). Dzięki temu migotanie etykiet
    # MediaPipe nie przełącza sterowania z ręki na rękę.

    def __init__(self, tid, now):
        self.id = tid
        self.last_seen = now
        self.present = False
        self.hand = None
        self.world = None
        self.votes = deque(maxlen=12)
        self.label = "Right"
        self.conf = 0.0
        self.info = None
        self.rock_v = Voter()
        self.three_v = Voter()
        self.four_v = Voter()
        self.point_l_v = Voter(window=0.3)      # wskazanie w lewo
        self.point_r_v = Voter(window=0.3)      # wskazanie w prawo
        self.fist_d = Debounce(2, 2)
        self.peace_d = Debounce(2, 2)
        self.trail = deque()        # (czas, x, y, rozmiar) z ostatnich ~0.4 s
        self.activity = 0.0         # jak szybko dłoń się przesuwa (rozmiary dłoni / s)


class HandTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def update(self, dets, now, W, H, left_held=False, right_held=False):
        # dets: lista obiektów z polami label, conf, hand, world
        for tr in self.tracks:
            tr.present = False

        # dopasowanie wykryć do istniejących dłoni po pozycji nadgarstka
        pairs = []

        for ti, tr in enumerate(self.tracks):
            for di, d in enumerate(dets):
                dist = math.hypot(
                    (tr.hand[0].x - d.hand[0].x) * W,
                    (tr.hand[0].y - d.hand[0].y) * H
                ) / W
                pairs.append((dist, ti, di))

        pairs.sort()

        used_t = set()
        used_d = set()
        assigned = {}

        for dist, ti, di in pairs:
            if dist > MATCH_DISTANCE:
                break
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            assigned[di] = self.tracks[ti]

        for di, d in enumerate(dets):
            tr = assigned.get(di)

            if tr is None:
                tr = HandTrack(self.next_id, now)
                self.next_id += 1
                self.tracks.append(tr)

            self._apply(tr, d, now, W, H, left_held, right_held)

        self.tracks = [t for t in self.tracks if now - t.last_seen <= TRACK_TTL]

        while len(self.tracks) > 2:
            oldest = min(self.tracks, key=lambda t: t.last_seen)
            self.tracks.remove(oldest)

        return self.tracks

    def _apply(self, tr, d, now, W, H, left_held, right_held):
        tr.hand = d.hand
        tr.world = d.world
        tr.last_seen = now
        tr.present = True
        tr.conf = d.conf

        tr.votes.append((d.label, d.conf))
        score = {"Right": 0.0, "Left": 0.0}

        for lab, c in tr.votes:
            score[lab] += c

        tr.label = "Right" if score["Right"] >= score["Left"] else "Left"

        info = analyze_hand(d.hand, d.world, left_held, right_held)
        tr.info = info

        tr.rock_v.add(now, info["rock"])
        tr.three_v.add(now, info["three"])
        tr.four_v.add(now, info["four_tucked"])
        tr.fist_d.update(info["fist"])
        tr.peace_d.update(info["peace"])

        # aktywność = przesunięcie dłoni w ostatnich ~0.3 s (szum się uśrednia)
        cx = sum(d.hand[i].x for i in POINTER_IDS) / len(POINTER_IDS) * W
        cy = sum(d.hand[i].y for i in POINTER_IDS) / len(POINTER_IDS) * H
        size = math.hypot((d.hand[0].x - d.hand[9].x) * W, (d.hand[0].y - d.hand[9].y) * H)

        # kierunek wskazującego palca (w obrazie podglądu, czyli tak jak widzisz
        # siebie w lustrze: lewo = Twoja lewa strona)
        direction = 0

        if info["point"] and size > 1.0:
            pdx = (d.hand[8].x - d.hand[5].x) * W
            pdy = (d.hand[8].y - d.hand[5].y) * H

            if (
                abs(pdx) > POINT_MIN_LENGTH * size
                and abs(pdx) > POINT_HORIZONTAL_RATIO * abs(pdy)
            ):
                direction = 1 if pdx > 0 else -1

        info["point_dir"] = direction
        tr.point_l_v.add(now, direction < 0)
        tr.point_r_v.add(now, direction > 0)

        tr.trail.append((now, cx, cy, size))

        while tr.trail and now - tr.trail[0][0] > 0.4:
            tr.trail.popleft()

        t0, x0, y0, _ = tr.trail[0]

        if now - t0 >= 0.25 and size > 1.0:
            tr.activity = math.hypot(cx - x0, cy - y0) / size / (now - t0)


# ==============================================================
# SKANER TWARZY (rysuje punkty twarzy i daje położenie uszu do gestu "słucham")
# ==============================================================

# Numery punktów siatki twarzy MediaPipe (478 punktów, razem ze źrenicami)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
             365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
             132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386,
            387, 388, 466]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159,
             160, 161, 246]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270,
              269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310,
              311, 312, 13, 82, 81, 80, 191]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# punkty siatki na obu bokach twarzy, na wysokości uszu
FACE_SIDE_A = 234
FACE_SIDE_B = 454

FACE_LOOPS = [
    (FACE_OVAL, (255, 255, 0)),
    (LEFT_EYE, (0, 255, 255)),
    (RIGHT_EYE, (0, 255, 255)),
    (LIPS_OUTER, (80, 80, 255)),
    (LIPS_INNER, (80, 80, 255)),
    (LEFT_IRIS, (255, 128, 0)),
    (RIGHT_IRIS, (255, 128, 0)),
]

FACE_LINES = [
    ([276, 283, 282, 295, 285], (0, 255, 255)),
    ([300, 293, 334, 296, 336], (0, 255, 255)),
    ([46, 53, 52, 65, 55], (0, 255, 255)),
    ([70, 63, 105, 66, 107], (0, 255, 255)),
]


class FaceScanner(threading.Thread):
    # Działa we własnym wątku, więc nie spowalnia śledzenia dłoni.
    # Wynik: self.points = tablica (N, 2) ze znormalizowanymi współrzędnymi
    # punktów twarzy (albo None, gdy twarzy nie widać).

    def __init__(self):
        super().__init__(daemon=True)
        self.enabled = FACE_AT_START
        self.status = "start"       # start / pobieranie / ok / blad
        self.points = None
        self.stamp = 0.0
        self.busy = False
        self.last_submit = 0.0
        self.running = True
        self._cond = threading.Condition()
        self._frame = None
        self._frame_t = 0.0

    def wants_frame(self, now):
        return (
            self.enabled
            and self.status == "ok"
            and not self.busy
            and now - self.last_submit >= 1.0 / FACE_MAX_FPS
        )

    def submit(self, small_bgr, t):
        with self._cond:
            self._frame = small_bgr
            self._frame_t = t
            self.busy = True
            self.last_submit = t
            self._cond.notify()

    def run(self):
        landmarker = None

        try:
            self.status = "pobieranie"

            if not ensure_model(FACE_MODEL_PATH, FACE_MODEL_URL):
                self.status = "blad"
                return

            options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )

            landmarker = vision.FaceLandmarker.create_from_options(options)
            self.status = "ok"

            t0 = time.perf_counter()
            last_ts = -1

            while self.running:
                with self._cond:
                    self._cond.wait_for(
                        lambda: self._frame is not None or not self.running,
                        0.5
                    )
                    frame = self._frame
                    frame_t = self._frame_t
                    self._frame = None

                if frame is None:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts = max(int((frame_t - t0) * 1000), last_ts + 1)
                last_ts = ts

                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                    ts
                )

                if result.face_landmarks:
                    self.points = np.array(
                        [[p.x, p.y] for p in result.face_landmarks[0]],
                        dtype=np.float32
                    )
                    self.stamp = time.perf_counter()
                else:
                    self.points = None

                self.busy = False

        except Exception:
            traceback.print_exc()
            self.status = "blad"

        finally:
            self.busy = False
            if landmarker is not None:
                try:
                    landmarker.close()
                except Exception:
                    pass

    def stop(self):
        self.running = False
        with self._cond:
            self._cond.notify_all()
        self.join(1.0)


def draw_face(img, pts_norm):
    h, w = img.shape[:2]
    pts = (pts_norm * np.array([w, h], dtype=np.float32)).astype(np.int32)

    # wszystkie punkty siatki (małe kwadraciki 2x2)
    xs = np.clip(pts[:, 0], 0, w - 2)
    ys = np.clip(pts[:, 1], 0, h - 2)

    for dx in (0, 1):
        for dy in (0, 1):
            img[ys + dy, xs + dx] = (60, 230, 60)

    # kontury: owal twarzy, oczy, brwi, usta, tęczówki
    n = len(pts)

    for idx, color in FACE_LOOPS:
        if max(idx) < n:
            cv2.polylines(img, [pts[idx]], True, color, 1, cv2.LINE_AA)

    for idx, color in FACE_LINES:
        if max(idx) < n:
            cv2.polylines(img, [pts[idx]], False, color, 1, cv2.LINE_AA)

    # ramka z narożnikami wokół twarzy
    x0, y0 = int(pts[:, 0].min()), int(pts[:, 1].min())
    x1, y1 = int(pts[:, 0].max()), int(pts[:, 1].max())
    size = max(8, int(0.2 * min(x1 - x0, y1 - y0)))
    color = (255, 255, 0)

    for px, py, sx, sy in (
        (x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)
    ):
        cv2.line(img, (px, py), (px + sx * size, py), color, 2, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + sy * size), color, 2, cv2.LINE_AA)

    cv2.putText(img, "TWARZ", (x0, max(14, y0 - 8)), FONT, 0.5, color, 1, cv2.LINE_AA)


# ==============================================================
# GEST "SŁUCHAM" (obie dłonie przy uszach)
# ==============================================================


class ListenGesture:
    # Z siatki twarzy bierzemy dwa punkty po bokach głowy (na wysokości uszu)
    # i liczymy "strefy uszu". Gest jest aktywny, gdy jedna dłoń jest przy
    # lewym uchu, a druga przy prawym (środek dłoni w strefie). Położenie uszu
    # pamiętamy ok. 1.5 s - dłonie przy uszach mogą na chwilę zasłonić twarz.
    # Nie używamy etykiet Left/Right z MediaPipe, tylko pozycji dłoni w kadrze.

    def __init__(self):
        self.ears = None            # (lewe, prawe) w pikselach kamery
        self.radius = 0.0           # promień strefy ucha w pikselach
        self.seen_t = -99.0
        self.near_flags = (False, False)
        self.voter = Voter(window=0.4, ratio=0.6, min_samples=3)

    def update_face(self, face, now, W, H):
        pts = face.points

        if (
            not face.enabled
            or pts is None
            or now - face.stamp > 0.5
            or len(pts) <= max(FACE_SIDE_A, FACE_SIDE_B)
        ):
            return

        a = np.array([pts[FACE_SIDE_A][0] * W, pts[FACE_SIDE_A][1] * H], dtype=np.float32)
        b = np.array([pts[FACE_SIDE_B][0] * W, pts[FACE_SIDE_B][1] * H], dtype=np.float32)

        fw = float(np.linalg.norm(a - b))     # szerokość twarzy w pikselach

        if fw < 20.0:
            return

        mid = (a + b) / 2.0
        da = (a - mid) / (fw / 2.0)           # wektory jednostkowe "na zewnątrz"
        db = (b - mid) / (fw / 2.0)

        ea = a + da * fw * LISTEN_EAR_OUTSET
        eb = b + db * fw * LISTEN_EAR_OUTSET

        left, right = (ea, eb) if ea[0] <= eb[0] else (eb, ea)

        self.ears = (left, right)
        self.radius = fw * LISTEN_NEAR
        self.seen_t = now

    def check(self, present, now, W, H):
        # zwraca (poza_słucham, czy_jakakolwiek_dłoń_jest_przy_uchu)
        fresh = self.ears is not None and now - self.seen_t <= LISTEN_FACE_MEMORY
        hands = [t for t in present if t.hand is not None]

        if not fresh or not hands:
            self.near_flags = (False, False)
            self.voter.add(now, False)
            return self.voter.active(now), False

        r = self.radius
        left_ear, right_ear = self.ears
        centres = []

        for t in hands:
            h = t.hand
            cx = sum(h[i].x for i in PALM_IDS) / len(PALM_IDS) * W
            cy = sum(h[i].y for i in PALM_IDS) / len(PALM_IDS) * H
            centres.append((cx, cy))

        near_l = any(math.hypot(c[0] - left_ear[0], c[1] - left_ear[1]) < r for c in centres)
        near_r = any(math.hypot(c[0] - right_ear[0], c[1] - right_ear[1]) < r for c in centres)
        self.near_flags = (near_l, near_r)

        pose = False

        if len(centres) >= 2:
            centres.sort(key=lambda c: c[0])
            c1, c2 = centres[0], centres[-1]

            pose = (
                math.hypot(c1[0] - left_ear[0], c1[1] - left_ear[1]) < r
                and math.hypot(c2[0] - right_ear[0], c2[1] - right_ear[1]) < r
            )

        self.voter.add(now, pose)

        return self.voter.active(now), (near_l or near_r)


# ==============================================================
# STEROWANIE  (cała logika: gesty -> mysz)
# ==============================================================


class Controller:
    def __init__(self, api, desktop, driver):
        self.api = api
        self.desktop = desktop
        self.driver = driver

        # ustawienia (zmieniane suwakami)
        self.enabled = START_ENABLED
        self.enabled_changed = False
        self.hand_mode = DEFAULT_HAND_MODE
        self.sens = DEFAULT_SENS / 10.0
        self.accel = DEFAULT_ACCEL
        self.smooth = DEFAULT_SMOOTH

        # przyciski myszy
        self.left_btn = PinchButton(LEFT_PINCH_THRESHOLD, LEFT_RELEASE_THRESHOLD)
        self.right_btn = PinchButton(PINCH_THRESHOLD, RELEASE_THRESHOLD)

        # gesty dwiema dłońmi
        self.toggle_hold = HoldCharge(TOGGLE_HOLD_SECONDS)

        # gest "słucham" -> skrót do aplikacji
        self.listen = ListenGesture()
        self.listen_hold = HoldCharge(LISTEN_HOLD_SECONDS)
        self.app_cooldown_until = 0.0

        # głośność
        self.vol_hold = HoldCharge(VOLUME_HOLD_SECONDS)
        self.vol_mode = False
        self.vol_pose_t = -99.0
        self.vol_filter = OneEuro1D()
        self.vol_prev = None
        self.vol_acc = 0.0

        # play / pauza (4 palce + kciuki schowane, obie dłonie)
        self.play_hold = HoldCharge(PLAYPAUSE_HOLD_SECONDS)
        self.play_busy = False      # True, gdy Spotify jest właśnie uruchamiany

        # następna / poprzednia piosenka (palec wskazujący w bok)
        self.skip_next_hold = HoldCharge(SKIP_HOLD_SECONDS)
        self.skip_prev_hold = HoldCharge(SKIP_HOLD_SECONDS)

        # wskaźnik (ruch kursora)
        self.pointer = OneEuro2D()
        self.ref_valid = False
        self.prev = (0.0, 0.0)
        self.prev_t = 0.0
        self.speed = 0.0
        self.scale = None
        self.rearm_until = 0.0
        self.lost_since = None
        self.control_id = None
        self.scroll_hold_until = 0.0
        self.hist = deque()             # (czas, x, y) kursora z ostatniej sekundy

        # scroll
        self.scroll_filter = OneEuro1D()
        self.scroll_prev = None
        self.scroll_acc = 0.0

        # dane dla nakładki na obraz
        self.hud = SimpleNamespace(
            rock_n=0, toggle_p=0.0,
            listen_p=0.0, listen_pose=False,
            vol_p=0.0, vol_mode=False,
            four_n=0, play_p=0.0, play_pose=False,
            skip_dir=0, skip_p=0.0,
            lock=False, scroll=False, paused=False,
            control_id=None, pointer=None,
        )

    # ---------- włączanie / wyłączanie ----------

    def set_enabled(self, value, source="", now=None):
        value = bool(value)

        if value == self.enabled:
            return

        self.enabled = value
        self.enabled_changed = True

        if not value:
            self.release_buttons()
            self.reset_scroll()

        self.invalidate_pointer(time.perf_counter() if now is None else now)

        print("Sterowanie: %s%s" % (
            "ON" if value else "OFF",
            " (%s)" % source if source else ""
        ))

        if SOUND_FEEDBACK:
            self.api.beep(1000 if value else 500, 120)

    # ---------- pomocnicze ----------

    def invalidate_pointer(self, now):
        # "Myszka uniesiona": po powrocie dłoń dostaje nowy punkt odniesienia,
        # więc kursor nie skacze.
        self.rearm_until = now + MOVE_REARM_SECONDS
        self.ref_valid = False
        self.driver.cancel()

    def pick_control(self, sel):
        # Która dłoń steruje kursorem (trzymamy się tej samej). W trybie "obie"
        # (PL) sterowanie przechodzi na drugą dłoń, gdy ta dotychczasowa stoi,
        # a druga wyraźnie się rusza.
        current = None

        for t in sel:
            if t.id == self.control_id:
                current = t
                break

        if current is not None:
            if self.hand_mode == 2 and len(sel) > 1 and current.activity < 0.2:
                other = max(
                    (t for t in sel if t is not current),
                    key=lambda t: t.activity
                )

                if other.activity > 0.45:
                    self.control_id = other.id
                    self.ref_valid = False
                    return other

            return current

        if not sel:
            return None

        best = max(sel, key=lambda t: t.conf)

        if self.control_id != best.id:
            self.control_id = best.id
            self.ref_valid = False

        return best

    def release_buttons(self):
        if self.left_btn.held:
            self.api.left_up()
            self.left_btn.force_release()

        if self.right_btn.held:
            self.api.right_up()
            self.right_btn.force_release()

        self.left_btn.disarm()
        self.right_btn.disarm()

    def reset_scroll(self):
        self.scroll_prev = None
        self.scroll_acc = 0.0

    def _hist_at(self, t):
        # pozycja kursora z chwili t (ostatni zapis nie późniejszy niż t)
        found = None

        for rec in self.hist:
            if rec[0] <= t:
                found = rec
            else:
                break

        return None if found is None else (found[1], found[2])

    def _rewind(self, now):
        # Szczypnięcie przesuwa całą dłoń. Jeśli tuż przed szczypnięciem
        # kursor stał w miejscu (celowałeś), cofamy go tam, gdzie był, gdy
        # zaczynałeś zaciskać palce - kliknięcie trafia w to, w co celowałeś.
        if not self.hist:
            return

        p_ref = self._hist_at(now - REWIND_SECONDS)
        p_before = self._hist_at(now - REWIND_SECONDS - 0.25)

        if p_ref is None or p_before is None:
            return

        if math.hypot(p_ref[0] - p_before[0], p_ref[1] - p_before[1]) > REWIND_STILL_PX:
            return

        current = self.hist[-1]

        if math.hypot(current[1] - p_ref[0], current[2] - p_ref[1]) > REWIND_MAX_PX:
            return

        self.driver.warp(p_ref[0], p_ref[1])

    def _button_down(self, now):
        # Przy wciśnięciu cofamy kursor tam, gdzie celowałeś, i kasujemy
        # ruch "w drodze". Potem kursor porusza się normalnie (bez blokady).
        self._rewind(now)
        self.driver.cancel()

    def pinch_slow(self, info):
        # Kursor zwalnia, gdy palce zbliżają się do kliknięcia - dzięki temu
        # kliknięcie nie "ucieka" z celu.
        sl = smoothstep(
            (info["left_norm"] - LEFT_PINCH_THRESHOLD)
            / (LEFT_APPROACH - LEFT_PINCH_THRESHOLD)
        )
        sr = smoothstep(
            (info["right_norm"] - PINCH_THRESHOLD)
            / (RIGHT_APPROACH - PINCH_THRESHOLD)
        )
        return lerp(SLOW_MIN, 1.0, min(sl, sr))

    # ---------- skróty do aplikacji ----------

    def run_gesture_app(self, gesture, now):
        name = GESTURE_APPS.get(gesture)

        if not name or now < self.app_cooldown_until:
            return

        self.app_cooldown_until = now + APP_COOLDOWN

        if launch_app(name) and SOUND_FEEDBACK:
            self.api.beep(1200, 100)

    # ---------- play / pauza ----------

    def do_play_pause(self):
        try:
            if PLAYPAUSE_METHOD.upper() != "AUTO":
                self.api.media_play_pause()
                print("Play/pauza: wyslano klawisz multimedialny (jak na klawiaturze)")
            else:
                self._play_pause_spotify_window()
        except Exception:
            print("Play/pauza: BLAD (patrz nizej)")
            traceback.print_exc()

    def _play_pause_spotify_window(self):
        if self.play_busy:
            print("Play/pauza: Spotify wlasnie sie uruchamia, poczekaj chwile")
            return

        if self.api.spotify_play_pause():
            print("Play/pauza: wyslano do okna Spotify")
            return

        # Spotify nie jest uruchomiony - uruchamiamy go (w tle, żeby nie zawiesić podglądu)
        print("Play/pauza: Spotify nie dziala - uruchamiam")
        self.play_busy = True
        threading.Thread(target=self._launch_spotify_and_play, daemon=True).start()

    def _launch_spotify_and_play(self):
        try:
            if not launch_app("spotify"):
                return

            deadline = time.time() + 25.0

            while time.time() < deadline:
                time.sleep(0.5)

                if self.api.find_spotify_window():
                    break
            else:
                print("Play/pauza: nie doczekalem sie okna Spotify")
                return

            if PLAYPAUSE_AUTOPLAY_AFTER_LAUNCH:
                time.sleep(PLAYPAUSE_AUTOPLAY_DELAY)

                if self.api.spotify_play_pause():
                    print("Play/pauza: Spotify uruchomiony, wlaczam muzyke")
        except Exception:
            traceback.print_exc()
        finally:
            self.play_busy = False

    # ---------- następna / poprzednia piosenka ----------

    def do_skip(self, direction):
        name = "nastepna" if direction > 0 else "poprzednia"

        try:
            if PLAYPAUSE_METHOD.upper() != "AUTO":
                self.api.media_track(direction)
                print("Piosenka (%s): wyslano klawisz multimedialny" % name)
            elif self.api.spotify_track(direction):
                print("Piosenka (%s): wyslano do okna Spotify" % name)
            else:
                print("Piosenka (%s): Spotify nie dziala" % name)
                return
        except Exception:
            print("Piosenka (%s): BLAD (patrz nizej)" % name)
            traceback.print_exc()
            return

        if SOUND_FEEDBACK:
            self.api.beep(1500 if direction > 0 else 1100, 50)

    # ---------- głośność ----------

    def update_volume(self, present, blocked_start, blocked_stay, now):
        # LEWA dłoń (ta z mniejszym x w kadrze) pokazuje trójkę -> po
        # VOLUME_HOLD_SECONDS włącza się tryb głośności. W trybie PRAWA dłoń
        # (ta z większym x) w górę = głośniej, w dół = ciszej. Lewa trójka
        # opuszczona = koniec trybu. Pozycje, a nie etykiety MediaPipe,
        # decydują która dłoń jest lewa, więc migotanie etykiet nie przeszkadza.
        allowed = VOLUME_ENABLED and self.enabled and self.hand_mode == 0

        left_t = None
        right_t = None

        if len(present) >= 2:
            ordered = sorted(present, key=lambda t: t.hand[0].x)
            left_t, right_t = ordered[0], ordered[-1]

        ok = allowed and not blocked_start and left_t is not None and left_t.info is not None
        raw = bool(ok and left_t.info["three"])
        vote = bool(ok and left_t.three_v.active(now))

        if raw:
            self.vol_pose_t = now

        if self.vol_hold.update(vote, now):
            self.vol_mode = True
            self.vol_prev = None
            self.vol_acc = 0.0
            self.invalidate_pointer(now)

            if SOUND_FEEDBACK:
                self.api.beep(1400, 60)

        if not self.vol_mode:
            return

        if (not allowed) or blocked_stay or now - self.vol_pose_t > VOLUME_EXIT_GRACE:
            self.vol_mode = False
            self.vol_prev = None
            self.vol_acc = 0.0

            if SOUND_FEEDBACK:
                self.api.beep(900, 60)

            return

        # sterowanie tylko, gdy w tej klatce lewa trójka naprawdę jest pokazana
        if right_t is None or now - self.vol_pose_t > 0.12:
            self.vol_prev = None
            return

        hand = right_t.hand
        raw_y = sum(hand[i].y for i in POINTER_IDS) / len(POINTER_IDS)

        if self.vol_prev is None:
            self.vol_filter = OneEuro1D()
            self.vol_prev = self.vol_filter(raw_y, now)
            self.vol_acc = 0.0
            return

        y = self.vol_filter(raw_y, now)
        self.vol_acc += (self.vol_prev - y) * VOLUME_KEYS_PER_FRAME
        self.vol_prev = y

        n = int(self.vol_acc)

        if n != 0:
            self.api.volume_keys(n)
            self.vol_acc -= n

    # ---------- przyciski i scroll ----------

    def handle_buttons(self, sel, scroll_active, now):
        left_values = [t.info["left_norm"] for t in sel]
        right_values = [t.info["right_norm"] for t in sel]

        # Gdy scroll jest aktywny, całkowicie blokujemy PPM.
        # Jeśli PPM był już wciśnięty, natychmiast go zwalniamy.
        if scroll_active:
            right_values = []

            if self.right_btn.held:
                self.api.right_up()
                self.right_btn.force_release()

            self.right_btn.disarm()

        event = self.left_btn.update(left_values, now)

        if event == "down":
            self._button_down(now)
            self.api.left_down()
        elif event == "up":
            self.api.left_up()

        if not scroll_active:
            event = self.right_btn.update(right_values, now)

            if event == "down":
                self._button_down(now)
                self.api.right_down()
            elif event == "up":
                self.api.right_up()

    def handle_scroll(self, sel, now):
        # SCROLL: wykonaj ✌ i przesuwaj dłoń góra/dół
        peace_hands = [t for t in sel if t.peace_d.state]

        if not peace_hands:
            self.reset_scroll()
            return

        hand = peace_hands[0].hand
        raw_y = (hand[0].y + hand[9].y) / 2

        if self.scroll_prev is None:
            self.scroll_filter = OneEuro1D()
            self.scroll_prev = self.scroll_filter(raw_y, now)
            self.scroll_acc = 0.0
            return

        y = self.scroll_filter(raw_y, now)
        movement = self.scroll_prev - y
        self.scroll_prev = y

        self.scroll_acc += movement * SCROLL_SENSITIVITY

        if abs(self.scroll_acc) >= SCROLL_QUANTUM:
            wheel_delta = int(self.scroll_acc)
            self.api.wheel(wheel_delta)
            self.scroll_acc -= wheel_delta

    # ---------- ruch kursora ----------

    def move(self, tr, now, tcap, W, H):
        hand = tr.hand

        # Punkt śledzenia: środek kostek dłoni - prawie nie rusza się, gdy
        # zginasz palce (np. przy kliknięciu).
        px = sum(hand[i].x for i in POINTER_IDS) / len(POINTER_IDS) * W
        py = sum(hand[i].y for i in POINTER_IDS) / len(POINTER_IDS) * H

        # Rozmiar dłoni w pikselach - jednostka, w której liczymy ruch.
        size = math.hypot(
            (hand[0].x - hand[9].x) * W,
            (hand[0].y - hand[9].y) * H
        )

        if size < 8.0:
            return

        self.scale = size if self.scale is None else self.scale + 0.2 * (size - self.scale)

        low, high = SMOOTH_CUTOFF_RANGE
        self.pointer.min_cutoff = low * (high / low) ** (self.smooth / 10.0)

        if not self.ref_valid:
            self.pointer.reset()
            fx, fy = self.pointer(px, py, tcap, self.scale)
            self.prev = (fx, fy)
            self.prev_t = tcap
            self.speed = 0.0
            self.ref_valid = True
            return

        fx, fy = self.pointer(px, py, tcap, self.scale)

        dt = max(tcap - self.prev_t, 1e-3)
        du = (fx - self.prev[0]) / self.scale       # ruch w rozmiarach dłoni
        dv = (fy - self.prev[1]) / self.scale
        self.prev = (fx, fy)
        self.prev_t = tcap

        inst = math.hypot(du, dv) / dt
        a = 1.0 - math.exp(-dt / SPEED_SMOOTH_SECONDS)
        self.speed += a * (inst - self.speed)

        # Miękka martwa strefa na PRĘDKOŚCI: drżenie dłoni (bardzo wolne,
        # losowe ruchy) jest wygaszane, a prawdziwy ruch przechodzi płynnie.
        gate = smoothstep(
            (self.speed - V_GATE_LOW) / (V_GATE_HIGH - V_GATE_LOW)
        )

        # Przyspieszenie: im szybszy ruch dłoni, tym dalszy skok kursora
        accel = 1.0 + ACCEL_GAIN_MAX * (self.accel / 10.0) * smoothstep(
            (self.speed - V_LOW) / (V_HIGH - V_LOW)
        )

        if self.left_btn.held or self.right_btn.held:
            slow = 1.0
        else:
            slow = self.pinch_slow(tr.info)

        # Ta sama "odległość na ekranie" na monitorach o różnym skalowaniu
        dpi = 1.0

        if DPI_SPEED:
            pos = self.driver.position()

            if pos is None:
                pos = self.api.get_cursor_pos()

            dpi = self.desktop.scale_at(pos[0], pos[1])

        gain = G_BASE * max(self.sens, 0.2) * accel * slow * dpi * gate

        self.driver.push(du * gain, dv * gain)

    # ---------- jedna klatka ----------

    def step(self, tracks, now, tcap, W, H):
        present = [t for t in tracks if t.present]
        hud = self.hud

        # 1) ON/OFF dwiema dłońmi (🤘) - działa niezależnie od ON/OFF i od wyboru ręki
        rock_n = sum(1 for t in tracks if t.rock_v.active(now))
        toggle_active = rock_n >= 2

        if self.toggle_hold.update(toggle_active, now):
            self.set_enabled(not self.enabled, "gest", now)

        # 2) "Słucham" (obie dłonie przy uszach) -> skrót do aplikacji.
        # Działa tylko przy sterowaniu ON.
        listen_raw, near_ear = self.listen.check(present, now, W, H)
        listen_pose = bool(listen_raw and self.enabled and not toggle_active)

        if self.listen_hold.update(listen_pose, now):
            self.run_gesture_app("listen", now)

        # 3) Głośność. Start jest blokowany, gdy jakakolwiek dłoń jest przy uchu
        # (żeby "słucham" nie włączało głośności); w trakcie trybu blokują go tylko
        # gesty dwiema dłońmi (prawa dłoń może spokojnie wędrować w górę).
        self.update_volume(
            present,
            toggle_active or listen_pose or near_ear,
            toggle_active or listen_pose,
            now
        )

        # 4) Play / pauza: 4 palce proste + kciuki schowane, obiema dłońmi.
        # Nie działa, gdy dłonie są przy uszach ani w trybie głośności.
        four_n = sum(1 for t in present if t.four_v.active(now))
        play_pose = bool(
            four_n >= 2
            and (self.enabled or not PLAYPAUSE_ONLY_WHEN_ON)
            and not toggle_active
            and not listen_pose
            and not near_ear
            and not self.vol_mode
        )

        if self.play_hold.update(play_pose, now):
            self.do_play_pause()

            if SOUND_FEEDBACK:
                self.api.beep(1300, 60)

        # 4b) Następna / poprzednia piosenka: palec wskazujący w bok (dowolna dłoń).
        # Nie działa podczas innych gestów dwiema dłońmi ani w trybie głośności.
        prev_vote = any(t.point_l_v.active(now) for t in present)
        next_vote = any(t.point_r_v.active(now) for t in present)

        skip_ok = (
            (self.enabled or not PLAYPAUSE_ONLY_WHEN_ON)
            and not toggle_active
            and not listen_pose
            and not play_pose
            and not self.vol_mode
        )

        prev_pose = bool(skip_ok and prev_vote and not next_vote)
        next_pose = bool(skip_ok and next_vote and not prev_vote)

        if self.skip_prev_hold.update(prev_pose, now):
            self.do_skip(-1)

        if self.skip_next_hold.update(next_pose, now):
            self.do_skip(1)

        skip_dir = -1 if prev_pose else (1 if next_pose else 0)

        # Podczas tych gestów kursor i kliknięcia są zablokowane
        lock = (
            toggle_active or listen_pose or self.vol_mode or play_pose
            or skip_dir != 0
        )

        # 5) Wybór dłoni
        want = (
            None if self.hand_mode == 2
            else "Right" if self.hand_mode == 0
            else "Left"
        )

        sel = [t for t in present if want is None or t.label == want]
        ctl = self.pick_control(sel)

        # Scroll (✌): blokada PPM i kursora działa od razu (surowa poza, bez
        # opóźnienia debounce) i trwa chwilę po zakończeniu gestu.
        if any(t.peace_d.state or t.info["peace"] for t in sel):
            self.scroll_hold_until = now + SCROLL_HOLD_SECONDS

        scroll_active = now < self.scroll_hold_until
        rock_pause = ctl is not None and ctl.rock_v.active(now)
        paused = ctl is not None and (ctl.fist_d.state or rock_pause)

        # 6) Kliknięcia i scroll
        # (bez dłoni też wywołujemy obsługę przycisków - dzięki temu trzymany
        # przycisk nie urywa się przy chwilowym zniknięciu dłoni z kadru)
        clicks_allowed = (
            self.enabled
            and not lock
            and not rock_pause
        )

        if clicks_allowed:
            self.handle_buttons(sel, scroll_active, now)
            self.handle_scroll(sel, now)
        else:
            self.release_buttons()
            self.reset_scroll()

        # 7) Ruch kursora
        if ctl is not None:
            self.lost_since = None

        if (not self.enabled) or lock or scroll_active or paused:
            # pięść / scroll / gest / wyłączone = "myszka uniesiona"
            self.invalidate_pointer(now)

        elif ctl is None:
            # Dłoń na chwilę zniknęła - czekamy chwilę, zanim zresetujemy punkt odniesienia
            if self.lost_since is None:
                self.lost_since = now
            elif now - self.lost_since > CONTROL_GRACE:
                self.invalidate_pointer(now)

        else:
            if now < self.rearm_until:
                self.ref_valid = False
            else:
                self.move(ctl, now, tcap, W, H)

        # historia pozycji kursora (do cofania przy kliknięciu)
        pos = self.driver.position()

        if pos is not None:
            self.hist.append((now, pos[0], pos[1]))

        while self.hist and now - self.hist[0][0] > 1.0:
            self.hist.popleft()

        # 8) Dane do nakładki
        hud.rock_n = rock_n
        hud.toggle_p = self.toggle_hold.progress()
        hud.listen_p = self.listen_hold.progress()
        hud.listen_pose = listen_pose
        hud.vol_p = self.vol_hold.progress()
        hud.vol_mode = self.vol_mode
        hud.four_n = four_n
        hud.play_p = self.play_hold.progress()
        hud.play_pose = play_pose
        hud.skip_dir = skip_dir
        hud.skip_p = max(self.skip_prev_hold.progress(), self.skip_next_hold.progress())
        hud.lock = lock
        hud.scroll = scroll_active
        hud.paused = paused
        hud.control_id = ctl.id if ctl is not None else None

        if ctl is not None:
            hud.pointer = (
                sum(ctl.hand[i].x for i in POINTER_IDS) / len(POINTER_IDS),
                sum(ctl.hand[i].y for i in POINTER_IDS) / len(POINTER_IDS),
            )
        else:
            hud.pointer = None


# ==============================================================
# NAKŁADKA NA OBRAZ (rysowanie podglądu)
# ==============================================================

# paleta (BGR)
COL_TEXT = (240, 240, 240)
COL_SUB = (178, 178, 178)
COL_DIM = (112, 110, 108)
COL_PANEL = (30, 26, 24)
COL_TRACK = (68, 64, 62)
COL_KEY = (86, 82, 80)
COL_DARK = (28, 28, 28)
COL_GREEN = (120, 222, 110)
COL_RED = (96, 96, 240)
COL_AMBER = (58, 190, 250)
COL_BLUE = (250, 178, 84)
COL_PURPLE = (228, 112, 204)

FONT_TITLE = cv2.FONT_HERSHEY_DUPLEX


def put_text(img, text, org, scale=0.55, color=(255, 255, 255), thick=1):
    # tekst z czarną obwódką - czytelny na obrazie z kamery
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def ptext(img, text, x, y, scale=0.5, color=COL_TEXT, thick=1, font=FONT):
    # czysty tekst (bez obwódki) - na ciemnych panelach
    cv2.putText(img, text, (int(x), int(y)), font, scale, color, thick, cv2.LINE_AA)


def text_size(text, scale=0.5, thick=1, font=FONT):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    return tw, th


def fill_rounded(img, p1, p2, color, r=8):
    x1, y1 = p1
    x2, y2 = p2
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))

    if r == 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        return

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)

    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def outline_rounded(img, p1, p2, color, r=8, thick=1):
    x1, y1 = p1
    x2, y2 = p2
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))

    if r == 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        return

    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thick, cv2.LINE_AA)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thick, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thick, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thick, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 0, 180, 270, color, thick, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 0, 270, 360, color, thick, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thick, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 0, 90, 180, color, thick, cv2.LINE_AA)


def blend_rounded(img, p1, p2, color, alpha=0.6, r=10):
    # półprzezroczysty zaokrąglony prostokąt (liczony tylko na wycinku obrazu)
    ih, iw = img.shape[:2]
    x1 = max(0, p1[0])
    y1 = max(0, p1[1])
    x2 = min(iw, p2[0])
    y2 = min(ih, p2[1])

    if x2 - x1 < 2 or y2 - y1 < 2:
        return

    roi = img[y1:y2, x1:x2]
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    fill_rounded(mask, (p1[0] - x1, p1[1] - y1), (p2[0] - x1 - 1, p2[1] - y1 - 1), 255, r)

    m = mask > 0
    roi[m] = (
        roi[m].astype(np.float32) * (1.0 - alpha)
        + np.array(color, dtype=np.float32) * alpha
    ).astype(np.uint8)


def chip_width(text, scale=0.45, pad=9):
    return text_size(text, scale)[0] + 2 * pad


def draw_chip(img, x, y, text, fg, bg, scale=0.45, pad=9, h=22, border=None):
    # "pastylka" z tekstem; pilnuje, żeby nie wyjść poza obraz
    ih, iw = img.shape[:2]
    tw, th = text_size(text, scale)
    w = tw + 2 * pad

    x = int(clamp(x, 2, max(2, iw - w - 2)))
    y = int(clamp(y, 2, max(2, ih - h - 2)))

    fill_rounded(img, (x, y), (x + w, y + h), bg, h // 2)

    if border is not None:
        outline_rounded(img, (x, y), (x + w, y + h), border, h // 2)

    ptext(img, text, x + pad, y + (h + th) // 2, scale, fg)
    return w


def draw_progress(img, x, y, w, value, color, h=6):
    fill_rounded(img, (x, y), (x + w, y + h), COL_TRACK, h // 2)

    fw = int(w * clamp(value, 0.0, 1.0))

    if fw >= h:
        fill_rounded(img, (x, y), (x + fw, y + h), color, h // 2)
    elif fw > 0:
        cv2.circle(img, (x + h // 2, y + h // 2), h // 2, color, -1, cv2.LINE_AA)


def draw_row(img, x, y, w, label, hint, value, progress, active, enabled=True, value_color=None):
    # jeden wiersz gestu: nazwa + podpowiedź po lewej, stan po prawej, pasek postępu pod spodem
    if active:
        label_color = COL_TEXT
    else:
        label_color = COL_SUB if enabled else COL_DIM

    ptext(img, label, x, y, 0.5, label_color, 1)

    lw, _ = text_size(label, 0.5)
    ptext(img, hint, x + lw + 8, y, 0.38, COL_DIM)

    if value_color is None:
        value_color = COL_GREEN if active else (COL_SUB if enabled else COL_DIM)

    vw, _ = text_size(value, 0.45)
    ptext(img, value, x + w - vw, y, 0.45, value_color)

    draw_progress(img, x, y + 8, w, progress, COL_GREEN if enabled else COL_DIM)


def draw_hands(img, tracks, hud, debug):
    h, w = img.shape[:2]

    for tr in tracks:
        if not tr.present:
            continue

        color = COL_GREEN if tr.label == "Right" else COL_BLUE
        pts = [(int(p.x * w), int(p.y * h)) for p in tr.hand]

        for a, b in CONNECTIONS:
            cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)

        for i, p in enumerate(pts):
            if i in (4, 8, 12, 16, 20):             # czubki palców
                cv2.circle(img, p, 6, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, p, 6, color, 2, cv2.LINE_AA)
            else:
                cv2.circle(img, p, 3, (255, 255, 255), -1, cv2.LINE_AA)

        wrist_x, wrist_y = pts[0]
        name = "PRAWA" if tr.label == "Right" else "LEWA"

        if tr.id == hud.control_id:
            name += "  STERUJE"

        draw_chip(img, wrist_x + 12, wrist_y - 36, name, COL_DARK, color, 0.45, 9, 22)

        if debug and tr.info:
            info = tr.info
            ext = info["ext_a"]
            tags = []

            if info["rock"]:
                tags.append("ROCK")
            if info["three"]:
                tags.append("TROJKA")
            if info["four_tucked"]:
                tags.append("4+KCIUK")
            if info["point_dir"] < 0:
                tags.append("WSKAZ<<")
            if info["point_dir"] > 0:
                tags.append("WSKAZ>>")
            if info["fist"]:
                tags.append("PIESC")
            if info["peace"]:
                tags.append("SCROLL")

            debug_y = min(wrist_y + 25, h - 70)

            put_text(
                img,
                "I%.2f M%.2f R%.2f P%.2f K%.2f" % (
                    info["index"], info["middle"], info["ring"],
                    info["pinky"], info["thumb_tucked"]
                ),
                (wrist_x + 10, debug_y), 0.45
            )

            put_text(
                img,
                "3D: I%s M%s R%s P%s  K%.2f" % (
                    "+" if ext["index"] else "-",
                    "+" if ext["middle"] else "-",
                    "+" if ext["ring"] else "-",
                    "+" if ext["pinky"] else "-",
                    info["thumb_tuck3d"],
                ),
                (wrist_x + 10, debug_y + 18), 0.45
            )

            put_text(
                img, " ".join(tags),
                (wrist_x + 10, debug_y + 38), 0.5, (0, 255, 255)
            )

    # znacznik punktu, którym sterujesz kursorem
    if hud.pointer is not None:
        cx = int(hud.pointer[0] * w)
        cy = int(hud.pointer[1] * h)

        color = COL_AMBER if (hud.paused or hud.lock or hud.scroll) else COL_GREEN

        cv2.circle(img, (cx, cy), 15, COL_DARK, 4, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 15, color, 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 3, color, -1, cv2.LINE_AA)


def draw_listen(img, listen, now, cam_w, cam_h):
    # Strefy uszu (tryb debug): zielone, gdy któraś dłoń jest w środku.
    if listen.ears is None or now - listen.seen_t > LISTEN_FACE_MEMORY:
        return

    h, w = img.shape[:2]
    sx = w / float(cam_w)
    sy = h / float(cam_h)
    r = max(4, int(listen.radius * sx))

    for ear, near in zip(listen.ears, listen.near_flags):
        c = (int(ear[0] * sx), int(ear[1] * sy))
        color = COL_GREEN if near else (160, 160, 160)
        cv2.circle(img, c, r, color, 1, cv2.LINE_AA)
        cv2.circle(img, c, 3, color, -1, cv2.LINE_AA)


def draw_minimap(img, desktop, cursor):
    # Mała mapka monitorów: widać, jak program widzi Twoje ułożenie ekranów
    # i gdzie jest teraz kursor (monitor z kursorem jest jaśniejszy).
    h, w = img.shape[:2]
    bw, bh = 244, 142

    x1 = w - bw - 12
    y1 = h - 34 - bh - 10
    x2 = x1 + bw
    y2 = y1 + bh

    blend_rounded(img, (x1, y1), (x2, y2), COL_PANEL, 0.68, 12)
    outline_rounded(img, (x1, y1), (x2, y2), COL_TRACK, 12)
    ptext(img, "PULPIT", x1 + 14, y1 + 21, 0.42, COL_SUB)

    area_x = x1 + 14
    area_y = y1 + 32
    area_w = bw - 28
    area_h = bh - 32 - 14

    dw = max(1, desktop.right - desktop.left)
    dh = max(1, desktop.bottom - desktop.top)
    s = min(area_w / dw, area_h / dh)

    ox = area_x + (area_w - dw * s) / 2.0
    oy = area_y + (area_h - dh * s) / 2.0

    for i, (l, t, r, b, _scale) in enumerate(desktop.mons):
        p1 = (int(ox + (l - desktop.left) * s), int(oy + (t - desktop.top) * s))
        p2 = (int(ox + (r - desktop.left) * s) - 1, int(oy + (b - desktop.top) * s) - 1)

        has_cursor = cursor is not None and l <= cursor[0] < r and t <= cursor[1] < b

        fill_rounded(img, p1, p2, (92, 84, 80) if has_cursor else (56, 52, 50), 3)
        outline_rounded(img, p1, p2, COL_TEXT if has_cursor else COL_SUB, 3)
        ptext(img, str(i + 1), p1[0] + 7, p1[1] + 17, 0.45, COL_TEXT if has_cursor else COL_SUB)

    if cursor is not None:
        cx = int(ox + (cursor[0] - desktop.left) * s)
        cy = int(oy + (cursor[1] - desktop.top) * s)
        cv2.circle(img, (cx, cy), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 3, COL_RED, -1, cv2.LINE_AA)


def draw_help_bar(img):
    # pasek z klawiszami na dole podglądu
    h, w = img.shape[:2]
    blend_rounded(img, (0, h - 34), (w, h + 2), COL_PANEL, 0.72, 0)

    items = (
        ("Q", "wyjscie"),
        ("T", "on/off"),
        ("D", "debug"),
        ("F", "twarz"),
        ("M", "mapa"),
        ("Ctrl+Alt+G", "on/off z dowolnego miejsca"),
    )

    x = 14
    y = h - 27

    for key, label in items:
        kw = chip_width(key, 0.4, 7)
        lw, _ = text_size(label, 0.4)

        if x + kw + 6 + lw > w - 10:
            break

        draw_chip(img, x, y, key, COL_TEXT, COL_KEY, 0.4, 7, 18)
        ptext(img, label, x + kw + 6, y + 13, 0.4, COL_SUB)
        x += kw + 6 + lw + 20


def draw_hud(img, ctrl, cam_fps, cam_mode, cv_fps, n_hands, face, now):
    h, w = img.shape[:2]
    hud = ctrl.hud

    accent = COL_GREEN if ctrl.enabled else COL_RED

    # --- panel ---
    px, py, pw, ph = 12, 12, 376, 340
    blend_rounded(img, (px, py), (px + pw, py + ph), COL_PANEL, 0.70, 14)
    outline_rounded(img, (px, py), (px + pw, py + ph), accent, 14)

    ix = px + 16
    iw = pw - 32

    # --- nagłówek: kropka, tytuł, ON/OFF ---
    cv2.circle(img, (ix + 6, py + 32), 7, accent, -1, cv2.LINE_AA)
    ptext(img, "STEROWANIE", ix + 24, py + 40, 0.72, COL_TEXT, 1, FONT_TITLE)

    badge = "ON" if ctrl.enabled else "OFF"
    bw = chip_width(badge, 0.5, 14)
    draw_chip(img, ix + iw - bw, py + 18, badge, COL_DARK, accent, 0.5, 14, 26)

    # --- aktualny tryb + stan przycisków myszy ---
    if not ctrl.enabled:
        mode, mode_color = "WYLACZONE (rock x2)", COL_RED
    elif hud.vol_mode:
        mode, mode_color = "GLOSNOSC", COL_AMBER
    elif hud.listen_pose:
        mode, mode_color = "SLUCHAM", COL_AMBER
    elif hud.play_pose:
        mode, mode_color = "PLAY / PAUZA", COL_AMBER
    elif hud.skip_dir != 0:
        mode, mode_color = "PIOSENKA", COL_AMBER
    elif hud.lock:
        mode, mode_color = "GEST 2 DLONI", COL_AMBER
    elif hud.paused:
        mode, mode_color = "PAUZA KURSORA", COL_AMBER
    elif hud.scroll:
        mode, mode_color = "SCROLL", COL_AMBER
    else:
        mode, mode_color = "GOTOWE", COL_GREEN

    draw_chip(img, ix, py + 56, mode, mode_color, COL_TRACK, 0.45, 10, 22, border=mode_color)

    xr = ix + iw

    for label, held in (("PPM", ctrl.right_btn.held), ("LPM", ctrl.left_btn.held)):
        cw = chip_width(label, 0.42, 9)
        xr -= cw
        draw_chip(
            img, xr, py + 56, label,
            COL_DARK if held else COL_DIM,
            COL_PURPLE if held else COL_TRACK,
            0.42, 9, 22
        )
        xr -= 6

    # --- statystyki ---
    ptext(img, "CAM %.0f fps  %s" % (cam_fps, cam_mode), ix, py + 100, 0.4, COL_SUB)

    right_stats = "CV %.0f fps   dlonie %d" % (cv_fps, n_hands)
    rw, _ = text_size(right_stats, 0.4)
    ptext(img, right_stats, ix + iw - rw, py + 100, 0.4, COL_SUB)

    cv2.line(img, (ix, py + 112), (ix + iw, py + 112), COL_TRACK, 1, cv2.LINE_AA)

    # --- wiersze gestów ---
    # (nazwa, podpowiedź, stan, postęp, aktywny, dostępny, kolor stanu)
    rows = []

    rows.append((
        "ON / OFF", "rock x2 obiema dlonmi",
        "%d/2" % hud.rock_n, hud.toggle_p, hud.rock_n >= 2, True, None
    ))

    if not face.enabled:
        rows.append(("SLUCHAM", "rece przy uszach", "twarz OFF (F)", hud.listen_p, False, False, None))
    else:
        rows.append((
            "SLUCHAM", "rece przy uszach",
            "TAK" if hud.listen_pose else "-", hud.listen_p, hud.listen_pose, True, None
        ))

    if not VOLUME_ENABLED:
        rows.append(("GLOSNOSC", "lewa trojka", "wylaczona", hud.vol_p, False, False, None))
    elif ctrl.hand_mode != 0:
        rows.append(("GLOSNOSC", "lewa trojka", "tylko Reka = P", hud.vol_p, False, False, None))
    else:
        rows.append((
            "GLOSNOSC", "lewa trojka",
            "AKTYWNA" if hud.vol_mode else "-", hud.vol_p, hud.vol_mode, True, None
        ))

    if (not ctrl.enabled) and PLAYPAUSE_ONLY_WHEN_ON:
        rows.append((
            "PLAY / PAUZA", "4 palce + kciuki",
            "wlacz sterowanie", hud.play_p, False, True, COL_AMBER
        ))
    else:
        rows.append((
            "PLAY / PAUZA", "4 palce + kciuki",
            "%d/2" % hud.four_n, hud.play_p, hud.four_n >= 2, True, None
        ))

    if hud.skip_dir < 0:
        skip_value = "<< poprzednia"
    elif hud.skip_dir > 0:
        skip_value = "nastepna >>"
    else:
        skip_value = "-"

    rows.append((
        "PIOSENKA", "palec w bok",
        skip_value, hud.skip_p, hud.skip_dir != 0, True, None
    ))

    y = py + 138

    for label, hint, value, progress, active, available, value_color in rows:
        draw_row(img, ix, y, iw, label, hint, value, progress, active, available, value_color)
        y += 36

    # --- stopka panelu: skaner twarzy ---
    cv2.line(img, (ix, py + ph - 38), (ix + iw, py + ph - 38), COL_TRACK, 1, cv2.LINE_AA)

    if not face.enabled:
        face_text, face_color = "Twarz: wylaczona (F)", COL_DIM
    elif face.status == "pobieranie":
        face_text, face_color = "Twarz: pobieranie modelu...", COL_AMBER
    elif face.status == "blad":
        face_text, face_color = "Twarz: blad modelu (patrz konsola)", COL_RED
    elif face.points is not None and now - face.stamp < 0.5:
        face_text, face_color = "Twarz: wykryta", COL_GREEN
    else:
        face_text, face_color = "Twarz: szukam...", COL_AMBER

    cv2.circle(img, (ix + 5, py + ph - 18), 5, face_color, -1, cv2.LINE_AA)
    ptext(img, face_text, ix + 18, py + ph - 13, 0.45, COL_SUB)

    draw_help_bar(img)


# ==============================================================
# WYKRYWANIE DŁONI NA KLATCE
# ==============================================================


def build_detections(results):
    # Unikalne dłonie. Duplikaty (to samo wykrycie dwa razy) usuwamy po
    # pozycji nadgarstka, a NIE po etykiecie - MediaPipe często daje obu
    # dłoniom tę samą etykietę Left/Right.
    dets = []

    if not results.hand_landmarks:
        return dets

    for i, hand in enumerate(results.hand_landmarks):
        category = results.handedness[i][0]
        label = category.category_name

        if SWAP_HANDEDNESS:
            label = "Right" if label == "Left" else "Left"

        if label not in ("Left", "Right"):
            label = "Right"

        world = None

        if i < len(results.hand_world_landmarks):
            world = world_array(results.hand_world_landmarks[i])

        det = SimpleNamespace(
            label=label, conf=category.score, hand=hand, world=world
        )

        duplicate = None

        for j, other in enumerate(dets):
            if distance(hand[0], other.hand[0]) < DUPLICATE_HAND_DISTANCE:
                duplicate = j
                break

        if duplicate is None:
            dets.append(det)
        elif det.conf > dets[duplicate].conf:
            dets[duplicate] = det

    return dets


# ==============================================================
# PROGRAM GŁÓWNY
# ==============================================================


def main():
    if not IS_WINDOWS:
        print("Ten program dziala tylko w systemie Windows.")
        return

    set_dpi_awareness()

    api = WinAPI()
    api.timer_begin()

    if not ensure_model(HAND_MODEL_PATH, HAND_MODEL_URL):
        api.timer_end()
        raise FileNotFoundError("Nie znaleziono pliku hand_landmarker.task!")

    # ---------- monitory ----------
    desktop = Desktop(api.enum_monitors())

    print("Wykryte monitory:", len(desktop.mons))

    for i, m in enumerate(desktop.mons):
        print("  %d: %dx%d  pozycja (%d, %d)  skalowanie %d%%" % (
            i + 1, m[2] - m[0], m[3] - m[1], m[0], m[1], round(m[4] * 100)
        ))

    print("Obszar pulpitu:", desktop.right - desktop.left, "x", desktop.bottom - desktop.top)

    # ---------- skróty do aplikacji ----------
    for gesture, app in GESTURE_APPS.items():
        path = resolve_app(app)

        if path:
            print("Skrot '%s' (gest %s): %s" % (app, gesture, path))
        else:
            print("UWAGA: nie znalazlem aplikacji '%s' (gest %s). Wpisz sciezke w APP_PATHS "
                  "albo wrzuc skrot do folderu: %s" % (app, gesture, APPS_FOLDER))

    # ---------- MediaPipe ----------
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=HAND_DETECT_CONF,
        min_hand_presence_confidence=HAND_PRESENCE_CONF,
        min_tracking_confidence=HAND_TRACK_CONF,
    )

    landmarker = vision.HandLandmarker.create_from_options(options)

    # ---------- kamera ----------
    cam = CameraThread(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)

    if not cam.opened:
        landmarker.close()
        api.timer_end()
        raise RuntimeError("Nie udalo sie uruchomic kamery!")

    start_t = time.perf_counter()
    cam.start()

    driver = CursorDriver(api, desktop)
    driver.start()

    ctrl = Controller(api, desktop, driver)
    tracker = HandTracker()

    face = FaceScanner()
    face.start()

    # ---------- okno i suwaki ----------
    saved = load_settings()

    def saved_int(key, default, lo, hi):
        try:
            return int(clamp(int(saved.get(key, default)), lo, hi))
        except Exception:
            return default

    show_debug = bool(saved.get("debug", True))
    show_map = bool(saved.get("map", True))
    face.enabled = bool(saved.get("face", FACE_AT_START))

    cv2.namedWindow(WINDOW_NAME)

    cv2.createTrackbar(
        "Reka: P/L/PL", WINDOW_NAME,
        saved_int("hand", DEFAULT_HAND_MODE, 0, 2), 2, lambda x: None
    )
    cv2.createTrackbar(
        "Czulosc", WINDOW_NAME,
        saved_int("sens", DEFAULT_SENS, 3, 30), 30, lambda x: None
    )
    cv2.createTrackbar(
        "Przyspiesz", WINDOW_NAME,
        saved_int("accel", DEFAULT_ACCEL, 0, 10), 10, lambda x: None
    )
    cv2.createTrackbar(
        "Wygladz", WINDOW_NAME,
        saved_int("smooth", DEFAULT_SMOOTH, 0, 10), 10, lambda x: None
    )
    cv2.createTrackbar(
        "Wyl/1Wl", WINDOW_NAME,
        1 if ctrl.enabled else 0, 1, lambda x: None
    )

    print("Program uruchomiony.")
    print("Q lub ESC - zamknij, T lub Ctrl+Alt+G - ON/OFF, D - debug, F - twarz, M - mapka monitorow.")

    last_seq = 0
    last_ts = -1
    cv_fps = 0.0
    last_loop = start_t
    last_monitor_check = start_t
    hotkey_prev = False
    last_tb = 1 if ctrl.enabled else 0
    closed_count = 0
    errors = 0

    try:
        while True:
            frame, seq, tcap = cam.wait_frame(last_seq, 0.2)

            if cam.failed:
                print("Nie udalo sie pobrac obrazu z kamery.")
                break

            if frame is None or seq == last_seq:
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    break

                continue

            last_seq = seq

            try:
                now = time.perf_counter()

                loop_dt = now - last_loop
                last_loop = now

                if loop_dt > 0:
                    cv_fps += 0.1 * (1.0 / loop_dt - cv_fps)

                # ---------- obraz ----------
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                timestamp_ms = max(int((tcap - start_t) * 1000), last_ts + 1)
                last_ts = timestamp_ms

                results = landmarker.detect_for_video(mp_image, timestamp_ms)

                # ---------- ustawienia z suwaków ----------
                ctrl.hand_mode = cv2.getTrackbarPos("Reka: P/L/PL", WINDOW_NAME)
                ctrl.sens = max(3, cv2.getTrackbarPos("Czulosc", WINDOW_NAME)) / 10.0
                ctrl.accel = cv2.getTrackbarPos("Przyspiesz", WINDOW_NAME)
                ctrl.smooth = cv2.getTrackbarPos("Wygladz", WINDOW_NAME)

                tb_enabled = cv2.getTrackbarPos("Wyl/1Wl", WINDOW_NAME)

                if tb_enabled != last_tb:
                    last_tb = tb_enabled
                    ctrl.set_enabled(bool(tb_enabled), "suwak")

                # ---------- dłonie i sterowanie ----------
                dets = build_detections(results)

                tracks = tracker.update(
                    dets, now, width, height,
                    ctrl.left_btn.held, ctrl.right_btn.held
                )

                # położenie uszu z ostatniej twarzy (do gestu "słucham")
                ctrl.listen.update_face(face, now, width, height)

                ctrl.step(tracks, now, tcap, width, height)

                # ---------- globalny skrót Ctrl+Alt+G ----------
                if GLOBAL_HOTKEY:
                    hotkey = (
                        api.key_down(VK_CONTROL)
                        and api.key_down(VK_MENU)
                        and api.key_down(VK_G)
                    )

                    if hotkey and not hotkey_prev:
                        ctrl.set_enabled(not ctrl.enabled, "Ctrl+Alt+G")

                    hotkey_prev = hotkey

                # ---------- układ monitorów (wykrywa podłączenie/odłączenie) ----------
                if now - last_monitor_check > 2.0:
                    last_monitor_check = now
                    monitors_now = api.enum_monitors()

                    if monitors_now and monitors_now != desktop.mons:
                        desktop.set(monitors_now)
                        print("Zmienil sie uklad monitorow, wykryto:", len(monitors_now))

                # ---------- skaner twarzy ----------
                if face.wants_frame(now):
                    face_h = max(1, int(height * FACE_PROC_WIDTH / width))
                    face.submit(
                        cv2.resize(
                            frame, (FACE_PROC_WIDTH, face_h),
                            interpolation=cv2.INTER_AREA
                        ),
                        tcap
                    )

                # ---------- podgląd ----------
                if PREVIEW_WIDTH and PREVIEW_WIDTH != width:
                    preview_h = int(height * PREVIEW_WIDTH / width)
                    preview = cv2.resize(
                        frame, (PREVIEW_WIDTH, preview_h),
                        interpolation=cv2.INTER_AREA
                    )
                else:
                    preview = frame

                face_pts = face.points

                if face.enabled and face_pts is not None and now - face.stamp < 0.5:
                    draw_face(preview, face_pts)

                if show_debug:
                    draw_listen(preview, ctrl.listen, now, width, height)

                draw_hands(preview, tracks, ctrl.hud, show_debug)

                if show_map:
                    cursor = driver.position()

                    if cursor is None:
                        cursor = api.get_cursor_pos()

                    draw_minimap(preview, desktop, cursor)

                draw_hud(
                    preview, ctrl, cam.fps, cam.mode_text, cv_fps,
                    sum(1 for t in tracks if t.present), face, now
                )

                cv2.imshow(WINDOW_NAME, preview)

                key = cv2.waitKey(1) & 0xFF

                if 65 <= key <= 90:
                    key += 32

                if key == ord("q") or key == 27:
                    break
                elif key == ord("t"):
                    ctrl.set_enabled(not ctrl.enabled, "klawisz T")
                elif key == ord("d"):
                    show_debug = not show_debug
                elif key == ord("f"):
                    face.enabled = not face.enabled
                elif key == ord("m"):
                    show_map = not show_map

                # suwak ON/OFF zawsze zgodny ze stanem (gest, klawisz, skrót)
                if ctrl.enabled_changed:
                    ctrl.enabled_changed = False
                    last_tb = 1 if ctrl.enabled else 0
                    cv2.setTrackbarPos("Wyl/1Wl", WINDOW_NAME, last_tb)

                # zamknięcie okna krzyżykiem
                try:
                    visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
                except cv2.error:
                    visible = 0

                closed_count = closed_count + 1 if visible < 1 else 0

                if closed_count >= 5:
                    break

                errors = 0

            except Exception:
                errors += 1

                if errors <= 5:
                    traceback.print_exc()

                if errors > 100:
                    print("Zbyt wiele bledow - koniec programu.")
                    break

    except KeyboardInterrupt:
        pass

    finally:
        # Zabezpieczenie przed pozostawieniem wciśniętych przycisków
        try:
            ctrl.release_buttons()
        except Exception:
            pass

        save_settings({
            "hand": int(ctrl.hand_mode),
            "sens": int(round(ctrl.sens * 10)),
            "accel": int(ctrl.accel),
            "smooth": int(ctrl.smooth),
            "face": bool(face.enabled),
            "debug": bool(show_debug),
            "map": bool(show_map),
        })

        driver.stop()
        face.stop()
        cam.stop()
        landmarker.close()
        cv2.destroyAllWindows()
        api.timer_end()

        print("Program zakonczony.")


if __name__ == "__main__":
    main()
