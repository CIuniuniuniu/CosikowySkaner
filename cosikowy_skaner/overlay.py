import cv2
import numpy as np

from .config import (
    CONNECTIONS,
    FONT,
    LISTEN_FACE_MEMORY,
    PLAYPAUSE_ONLY_WHEN_ON,
    VOLUME_ENABLED,
)
from .face import FACE_LINES, FACE_LOOPS
from .filters import clamp


# Paleta BGR używana również przez panel ustawień.
COL_TEXT = (248, 241, 232)
COL_SUB = (207, 188, 164)
COL_DIM = (159, 143, 126)
COL_PANEL = (40, 31, 23)
COL_TRACK = (76, 60, 48)
COL_KEY = (95, 75, 58)
COL_DARK = (31, 24, 18)
COL_GREEN = (169, 224, 112)
COL_RED = (137, 133, 247)
COL_AMBER = (96, 198, 250)
COL_BLUE = (229, 189, 114)
COL_PURPLE = (228, 132, 204)

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

    fill_rounded(img, (x, y), (x + w, y + h), bg, 7)

    if border is not None:
        outline_rounded(img, (x, y), (x + w, y + h), border, 7)

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

    blend_rounded(img, (x1, y1), (x2, y2), COL_PANEL, 0.90, 12)
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
    hud = ctrl.hud

    accent = COL_GREEN if ctrl.enabled else COL_SUB

    # --- panel ---
    px, py, pw, ph = 12, 12, 376, 340
    blend_rounded(img, (px, py), (px + pw, py + ph), COL_PANEL, 0.91, 14)
    outline_rounded(img, (px, py), (px + pw, py + ph), COL_TRACK, 14)

    ix = px + 16
    iw = pw - 32

    # --- nagłówek: kropka, tytuł, ON/OFF ---
    cv2.circle(img, (ix + 5, py + 33), 4, accent, -1, cv2.LINE_AA)
    ptext(img, "CosikowySkaner", ix + 19, py + 40, 0.72, COL_TEXT, 1, FONT_TITLE)

    badge = "ON" if ctrl.enabled else "OFF"
    bw = chip_width(badge, 0.5, 14)
    draw_chip(img, ix + iw - bw, py + 18, badge, COL_DARK, accent, 0.5, 14, 26)

    # --- aktualny tryb + stan przycisków myszy ---
    if not ctrl.enabled:
        mode, mode_color = "PAUZA STEROWANIA", COL_SUB
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
    elif not ctrl.gestures["mouse"]:
        mode, mode_color = "MYSZ WYLACZONA", COL_BLUE
    elif hud.paused:
        mode, mode_color = "PAUZA KURSORA", COL_AMBER
    elif hud.scroll:
        mode, mode_color = "SCROLL", COL_AMBER
    else:
        mode, mode_color = "GOTOWE", COL_GREEN

    draw_chip(img, ix, py + 56, mode, mode_color, COL_TRACK, 0.45, 10, 22, border=None)

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
    elif ctrl.hand_mode != 0 and ctrl.gestures["mouse"]:
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

    available_gestures = (
        ctrl.gestures["toggle"], ctrl.gestures["listen"], ctrl.gestures["volume"],
        ctrl.gestures["play"], ctrl.gestures["previous"] or ctrl.gestures["next"],
    )
    rows = [
        row if enabled else (row[0], row[1], "wylaczony", 0, False, False, None)
        for row, enabled in zip(rows, available_gestures)
    ]

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
