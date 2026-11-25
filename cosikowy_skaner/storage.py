import json
import os
import urllib.request

from .config import SETTINGS_PATH


def ensure_model(path, url):
    # Zwraca True, jeśli plik modelu istnieje (po ewentualnym pobraniu).
    if os.path.exists(path) and os.path.getsize(path) > 100000:
        return True

    name = os.path.basename(path)
    tmp = path + ".part"

    try:
        print("Pobieram model:", name, "...")
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
        print("Model pobrany:", name)
        return True
    except Exception as e:
        print("Nie udalo sie pobrac modelu %s: %s" % (name, e))
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return False


def load_settings():
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_settings(data):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass
