@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 goto error
)
if not exist ".venv\cosik-requirements.txt" goto install
fc /b requirements.txt ".venv\cosik-requirements.txt" >nul 2>&1
if errorlevel 1 goto install
goto run

:install
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error
copy /y requirements.txt ".venv\cosik-requirements.txt" >nul
if errorlevel 1 goto error

:run
".venv\Scripts\python.exe" handtest.py
if not errorlevel 1 exit /b 0

:error
echo Uruchomienie nie powiodlo sie. Sprawdz blad powyzej i instalacje Pythona 3.12 x64.
pause
exit /b 1
