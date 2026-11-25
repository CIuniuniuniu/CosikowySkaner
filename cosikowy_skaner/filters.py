from collections import deque
import math
import numpy as np

from .config import (
    BUTTON_LOST_GRACE,
    HOLD_DRAIN_RATE,
    ONE_EURO_BETA,
    ONE_EURO_D_CUTOFF,
    POSE_RATIO,
    POSE_WINDOW,
)


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
