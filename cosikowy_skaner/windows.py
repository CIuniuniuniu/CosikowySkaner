import ctypes
import os
import threading
import time

from .config import IS_WINDOWS


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
