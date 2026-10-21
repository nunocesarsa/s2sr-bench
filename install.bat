@echo off
REM ---------------------------------------------------------------
REM  S2 SR Bench - one-time setup (runs inside WSL Ubuntu)
REM  Creates ~/venvs/s2sr-gpu with Python 3.12 + PyTorch (CUDA 12.6)
REM  and installs everything in requirements.txt. Takes 5-15 minutes.
REM ---------------------------------------------------------------
setlocal
REM WSL distribution; leave empty to use your default one (check with: wsl -l -v)
set "DISTRO="
set "WSLARGS="
if not "%DISTRO%"=="" set "WSLARGS=-d %DISTRO%"

wsl %WSLARGS% --cd "%~dp0." -e bash install.sh
if errorlevel 1 (
  echo.
  echo Setup failed - see the messages above.
) else (
  echo.
  echo Setup finished. Double-click run_app.bat to start the app.
)
pause
