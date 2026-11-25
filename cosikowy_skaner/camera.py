import cv2
import threading
import time

from .config import CAMERA_BACKEND, CAMERA_EXPOSURE, CAMERA_MJPG, IS_WINDOWS


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
        backend = camera_backend_id()
        candidates = [backend]
        if IS_WINDOWS:
            candidates.extend(option for option in (cv2.CAP_DSHOW, cv2.CAP_MSMF) if option != backend)
        for option in candidates:
            self.cap = cv2.VideoCapture(index, option)
            if self.cap.isOpened():
                break
            self.cap.release()
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
