from types import SimpleNamespace
from collections import deque
import math
import numpy as np

from .config import (
    DUPLICATE_HAND_DISTANCE,
    EXTENDED_RATIO,
    EXT_ANGLE_DIP,
    EXT_ANGLE_MCP,
    EXT_ANGLE_PIP,
    FINGERS,
    FOLDED_RATIO,
    LEFT_PINCH_THRESHOLD,
    LEFT_RELEASE_THRESHOLD,
    MATCH_DISTANCE,
    PINCH_THRESHOLD,
    POINTER_IDS,
    POINT_HORIZONTAL_RATIO,
    POINT_MIN_LENGTH,
    RELEASE_THRESHOLD,
    SWAP_HANDEDNESS,
    THUMB_TUCKED_RATIO,
    THUMB_TUCK_3D,
    TRACK_TTL,
)
from .filters import Debounce, Voter, distance, vangle


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
