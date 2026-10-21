@echo off
REM ---------------------------------------------------------------
REM  S2 SR Bench - download sample scene(s) into the "samples" folder
REM  Each sample = 10-band Sentinel-2 L2A subset + high-res image of the same area.
REM  After downloading, pick "Sample scenes" in the app (works offline).
REM
REM    download_sample.bat                 default: amsterdam
REM    download_sample.bat all             every preset
REM    download_sample.bat berlin          one preset (amsterdam, rotterdam_port,
REM                                        flevoland_farmland, berlin)
REM    download_sample.bat --name paris --lat 48.8566 --lon 2.3522
REM ---------------------------------------------------------------
setlocal
set "VENV=~/venvs/s2sr-gpu"
set "DISTRO="
set "WSLARGS="
if not "%DISTRO%"=="" set "WSLARGS=-d %DISTRO%"

set "ARGS=--preset amsterdam"
if not "%~1"=="" (
  echo %~1| findstr /b /c:"--" >nul && (set "ARGS=%*") || (set "ARGS=--preset %*")
)

wsl %WSLARGS% --cd "%~dp0." -e bash -lc "source %VENV%/bin/activate && python -m s2sr.cli sample %ARGS%"
echo.
pause
