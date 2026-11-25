import cv2
import os
import sys


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
if getattr(sys, "frozen", False):
    HERE = os.path.dirname(sys.executable)
else:
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
APP_PATHS = {}
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
