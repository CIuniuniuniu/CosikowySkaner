# CosikowySkaner

Sterowanie myszą gestami dłoni. Aplikacja wymaga Windowsa i kamery.

## Uruchomienie wersji EXE

W zakładce **Actions** otwórz udany przebieg **Windows executable** i pobierz artefakt **CosikowySkaner-Windows-x64**. Rozpakuj całą paczkę i kliknij dwukrotnie **CosikowySkaner.exe**. Pliki modeli oraz folder `_internal` powinny zostać obok pliku EXE.

## Uruchomienie z kodu

Zainstaluj Python 3.12 x64 z launcherem `py`, a następnie uruchom `start.bat`. Przy pierwszym starcie skrypt tworzy środowisko i instaluje zależności.

- **T** lub **Ctrl+Alt+G** — włącz lub wyłącz sterowanie.
- **Q** lub **Escape** — zamknij program.
- Indeks kamery i skróty do aplikacji można ustawić w `cosikowy_skaner/config.py`.

## Ustawienia gestów

Przycisk **?** otwiera ilustrowany przewodnik po gestach.

Przycisk **Ustawienia** otwiera osobne okno z wyborem ręki i suwakami czułości, przyspieszenia i wygładzania. Każdy gest można przełączyć osobno.

Aby sterować tylko muzyką, zostaw główne sterowanie włączone, wyłącz **Sterowanie myszą**, **Gest ON / OFF** i **Skrót do Spotify**, a następnie wybierz gesty muzyczne. Wyłączenie myszy blokuje ruch kursora, oba kliknięcia i przewijanie. Po ponownym włączeniu myszy wracają poprzednie ustawienia kliknięć i przewijania. Program zapisuje ustawienia przy zamknięciu.

Moduły w `cosikowy_skaner/` obejmują kamerę, śledzenie dłoni i twarzy, obsługę gestów, kursor i API Windows oraz nakładkę podglądu. Plik `handtest.py` uruchamia aplikację.
