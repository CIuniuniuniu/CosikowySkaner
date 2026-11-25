import math
import threading
import time
import traceback

from .config import CURSOR_HZ, OUTPUT_TAU
from .filters import clamp


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
