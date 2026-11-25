import cv2
import mediapipe as mp
from mediapipe.tasks import python
import time
from contextlib import ExitStack
import traceback
from mediapipe.tasks.python import vision

from .camera import CameraThread
from .config import (
    APPS_FOLDER,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    DEFAULT_ACCEL,
    DEFAULT_HAND_MODE,
    DEFAULT_SENS,
    DEFAULT_SMOOTH,
    FACE_AT_START,
    FACE_MODEL_PATH,
    FACE_PROC_WIDTH,
    GESTURE_APPS,
    GLOBAL_HOTKEY,
    HAND_DETECT_CONF,
    HAND_MODEL_PATH,
    HAND_MODEL_URL,
    HAND_PRESENCE_CONF,
    HAND_TRACK_CONF,
    PREVIEW_WIDTH,
    WINDOW_NAME,
)
from .controller import Controller
from .cursor import CursorDriver, Desktop
from .face import FaceScanner
from .filters import clamp
from .hands import HandTracker, build_detections
from .overlay import draw_face, draw_hands, draw_hud, draw_listen, draw_minimap
from .shortcuts import resolve_app
from .storage import ensure_model, load_settings, save_settings
from .windows import VK_CONTROL, VK_G, VK_MENU, WinAPI, set_dpi_awareness


def create_hand_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=HAND_DETECT_CONF,
        min_hand_presence_confidence=HAND_PRESENCE_CONF,
        min_tracking_confidence=HAND_TRACK_CONF,
    )

    return vision.HandLandmarker.create_from_options(options)

def check_models():
    with create_hand_landmarker():
        pass
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
    )
    with vision.FaceLandmarker.create_from_options(options):
        pass


def run():
    set_dpi_awareness()
    api = WinAPI()
    with ExitStack() as resources:
        api.timer_begin()
        resources.callback(api.timer_end)
        run_camera_loop(api, resources)
    print("Program zakonczony.")


def run_camera_loop(api, resources):
    if not ensure_model(HAND_MODEL_PATH, HAND_MODEL_URL):
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
    landmarker = create_hand_landmarker()
    resources.callback(landmarker.close)

    # ---------- kamera ----------
    cam = CameraThread(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)

    if not cam.opened:
        raise RuntimeError("Nie udalo sie uruchomic kamery!")

    start_t = time.perf_counter()
    cam.start()
    resources.callback(cam.stop)

    driver = CursorDriver(api, desktop)
    driver.start()
    resources.callback(driver.stop)

    ctrl = Controller(api, desktop, driver)
    tracker = HandTracker()

    face = FaceScanner()
    face.start()
    resources.callback(face.stop)

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
    resources.callback(cv2.destroyAllWindows)

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
