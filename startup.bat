@echo off
setlocal enabledelayedexpansion

echo Starting Terrain STL Generator...

:: Create venv if needed (don't delete it every time)
if not exist .venv (
  echo Creating virtual environment...
  py -m venv .venv
)

echo Activating environment...
call .venv\Scripts\activate.bat

echo Installing requirements...
python -m pip install --upgrade pip >nul
pip install -r WEBAPP\requirements.txt

:: Load API key from API_KEY.txt if env var not already set
if "!OPEN_TOPO_API_KEY!"=="" (
  if exist API_KEY.txt (
    set /p OPEN_TOPO_API_KEY=<API_KEY.txt
  )
)

if "!OPEN_TOPO_API_KEY!"=="" (
  echo WARNING: Missing OpenTopography API key.
  echo   Set OPEN_TOPO_API_KEY, or create API_KEY.txt at the repo root.
  echo   STL generation will fail until this is configured.
)

echo Starting Uvicorn server...
python -m uvicorn WEBAPP.main:app --reload --host 127.0.0.1 --port 8000

echo Server stopped.
pause
endlocal
