"""Uruchamia CosikowySkaner na Windowsie."""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="CosikowySkaner")
    parser.add_argument("--check", action="store_true", help="Sprawdź zależności i modele bez kamery.")
    args = parser.parse_args()

    if not args.check and os.name != "nt":
        parser.exit(1, "CosikowySkaner wymaga systemu Windows.\n")

    try:
        from cosikowy_skaner.app import check_models, run
    except (ImportError, OSError) as error:
        parser.exit(1, f"Nie można załadować zależności: {error}\nUruchom start.bat, jeśli używasz kodu źródłowego.\n")

    if args.check:
        check_models()
        print("Zależności i oba modele wczytane poprawnie.")
    else:
        run()


if __name__ == "__main__":
    main()
