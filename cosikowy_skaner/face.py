import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
import threading
import time
import traceback
from mediapipe.tasks.python import vision

from .config import FACE_AT_START, FACE_MAX_FPS, FACE_MODEL_PATH, FACE_MODEL_URL
from .storage import ensure_model


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
