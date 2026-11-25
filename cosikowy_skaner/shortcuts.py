import os

from .config import APPS_FOLDER, APP_PATHS


def resolve_app(name):
    # Szuka aplikacji po nazwie: 1) APP_PATHS, 2) folder "skroty", 3) (tylko Spotify)
    # typowe miejsca instalacji. Zwraca ścieżkę albo None.
    key = str(name).lower()

    path = APP_PATHS.get(key, "")

    if path and os.path.exists(path):
        return path

    if os.path.isdir(APPS_FOLDER):
        for fname in sorted(os.listdir(APPS_FOLDER)):
            stem, ext = os.path.splitext(fname)

            if stem.lower() == key and ext.lower() in (".lnk", ".exe", ".url", ".bat"):
                return os.path.join(APPS_FOLDER, fname)

    if key == "spotify":
        candidates = [
            os.path.join(os.environ.get("APPDATA", ""), "Spotify", "Spotify.exe"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WindowsApps", "Spotify.exe"),
        ]

        for c in candidates:
            if c and os.path.exists(c):
                return c

    return None


def launch_app(name):
    path = resolve_app(name)

    if path is None:
        print("Nie znalazlem aplikacji '%s'. Wpisz sciezke w APP_PATHS albo wrzuc skrot do folderu: %s" % (
            name, APPS_FOLDER
        ))
        return False

    try:
        os.startfile(path)
        print("Uruchomiono:", path)
        return True
    except OSError as e:
        print("Nie udalo sie uruchomic %s: %s" % (path, e))
        return False
