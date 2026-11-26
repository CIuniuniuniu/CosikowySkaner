"""Uruchamia CosikowySkaner na Windowsie."""

import argparse
import ctypes
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="CosikowySkaner")
    parser.add_argument("--check", action="store_true", help="Sprawdź zależności i modele bez kamery.")
    args = parser.parse_args()

    if not args.check and os.name != "nt":
        parser.exit(1, "CosikowySkaner wymaga systemu Windows.\n")

    try:
        from cosikowy_skaner.app import check_models, run
    except (ImportError, OSError) as error:
        raise RuntimeError(f"Nie można załadować zależności: {error}") from error

    if args.check:
        check_models()
        print("Zaleznosci i oba modele wczytane poprawnie.")
    else:
        run()


def run_packaged():
    log_folder = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "CosikowySkaner"
    log_path = log_folder / "CosikowySkaner.log"
    try:
        log_folder.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            with redirect_stdout(log), redirect_stderr(log):
                try:
                    main()
                except Exception:
                    traceback.print_exc()
                    raise
    except Exception as error:
        if "--check" not in sys.argv:
            ctypes.windll.user32.MessageBoxW(
                None,
                f"{error}\n\nDziennik: {log_path}",
                "CosikowySkaner",
                0x10,
            )
        raise SystemExit(1) from None


if __name__ == "__main__":
    if getattr(sys, "frozen", False):
        run_packaged()
    else:
        main()
