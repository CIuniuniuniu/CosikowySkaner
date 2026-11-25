from types import SimpleNamespace
from collections import deque
import math
import numpy as np
import threading
import time
import traceback

from .config import (
    ACCEL_GAIN_MAX,
    APP_COOLDOWN,
    CONTROL_GRACE,
    DEFAULT_ACCEL,
    DEFAULT_HAND_MODE,
    DEFAULT_SENS,
    DEFAULT_SMOOTH,
    DPI_SPEED,
    GESTURE_APPS,
    G_BASE,
    LEFT_APPROACH,
    LEFT_PINCH_THRESHOLD,
    LEFT_RELEASE_THRESHOLD,
    LISTEN_EAR_OUTSET,
    LISTEN_FACE_MEMORY,
    LISTEN_HOLD_SECONDS,
    LISTEN_NEAR,
    MOVE_REARM_SECONDS,
    PALM_IDS,
    PINCH_THRESHOLD,
    PLAYPAUSE_AUTOPLAY_AFTER_LAUNCH,
    PLAYPAUSE_AUTOPLAY_DELAY,
    PLAYPAUSE_HOLD_SECONDS,
    PLAYPAUSE_METHOD,
    PLAYPAUSE_ONLY_WHEN_ON,
    POINTER_IDS,
    RELEASE_THRESHOLD,
    REWIND_MAX_PX,
    REWIND_SECONDS,
    REWIND_STILL_PX,
    RIGHT_APPROACH,
    SCROLL_HOLD_SECONDS,
    SCROLL_QUANTUM,
    SCROLL_SENSITIVITY,
    SKIP_HOLD_SECONDS,
    SLOW_MIN,
    SMOOTH_CUTOFF_RANGE,
    SOUND_FEEDBACK,
    SPEED_SMOOTH_SECONDS,
    START_ENABLED,
    TOGGLE_HOLD_SECONDS,
    VOLUME_ENABLED,
    VOLUME_EXIT_GRACE,
    VOLUME_HOLD_SECONDS,
    VOLUME_KEYS_PER_FRAME,
    V_GATE_HIGH,
    V_GATE_LOW,
    V_HIGH,
    V_LOW,
)
from .face import FACE_SIDE_A, FACE_SIDE_B
from .filters import (
    HoldCharge,
    OneEuro1D,
    OneEuro2D,
    PinchButton,
    Voter,
    lerp,
    smoothstep,
)
from .shortcuts import launch_app


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

    def set_hand_mode(self, mode):
        if mode != self.hand_mode:
            self.hand_mode = mode
            self.control_id = None
            self.suspend_input(time.perf_counter())

    def suspend_input(self, now):
        self.release_buttons()
        self.reset_scroll()
        self.scroll_hold_until = 0.0
        self.invalidate_pointer(now)

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
