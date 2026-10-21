@echo off
REM ---------------------------------------------------------------
REM  S2 SR Bench - start the app from Windows (runs inside WSL Ubuntu)
REM  Double-click this file, or run it from cmd / PowerShell.
REM ---------------------------------------------------------------
setlocal

REM Python environment inside Ubuntu (see README):
set "VENV=~/venvs/s2sr-gpu"
REM WSL distribution; leave empty to use your default one (check with: wsl -l -v)
set "DISTRO="
set "PORT=8501"

set "WSLARGS="
if not "%DISTRO%"=="" set "WSLARGS=-d %DISTRO%"

wsl %WSLARGS% -e bash -c "test -f %VENV%/bin/activate" || (
  echo.
  echo Could not find the Python environment %VENV% in WSL.
  echo Create it first - see "Install" in README.md.
  pause
  exit /b 1
)

echo Starting S2 SR Bench on http://localhost:%PORT%  ^(close this window or press Ctrl+C to stop^)
REM Open the browser once the server has had a few seconds to start
start "" /b cmd /c "timeout /t 8 /nobreak >nul & start http://localhost:%PORT%"

wsl %WSLARGS% --cd "%~dp0." -e bash -lc "source %VENV%/bin/activate && streamlit run app.py --server.headless true --server.port %PORT%"

echo.
echo The app has stopped.
pause
