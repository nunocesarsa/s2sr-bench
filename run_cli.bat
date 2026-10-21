@echo off
REM ---------------------------------------------------------------
REM  S2 SR Bench - command-line runs from Windows (inside WSL Ubuntu)
REM  Examples:
REM    run_cli.bat models
REM    run_cli.bat run --stac 52.37 4.90 --start 2025-06-01 --end 2025-08-31 --models bicubic_x4,sen2sr_lite_rgbn_x4 --out results/amsterdam
REM  Paths to local files must be Linux paths, e.g. /mnt/d/data/scene.tif
REM ---------------------------------------------------------------
setlocal
set "VENV=~/venvs/s2sr-gpu"
set "DISTRO="
set "WSLARGS="
if not "%DISTRO%"=="" set "WSLARGS=-d %DISTRO%"

if "%~1"=="" (
  wsl %WSLARGS% --cd "%~dp0." -e bash -lc "source %VENV%/bin/activate && python -m s2sr.cli --help"
) else (
  wsl %WSLARGS% --cd "%~dp0." -e bash -lc "source %VENV%/bin/activate && python -m s2sr.cli %*"
)
