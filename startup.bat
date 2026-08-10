@echo off
setlocal enabledelayedexpansion

echo Starting Terrain STL Generator...

:: 1) Python backend environment
if not exist .venv (
  echo Creating virtual environment...
  py -m venv .venv
)
call .venv\Scripts\activate.bat

echo Installing backend requirements...
python -m pip install --upgrade pip >nul
pip install -r backend\requirements.txt

:: 2) Build the React frontend so FastAPI can serve it at /
where npm >nul 2>nul
if %ERRORLEVEL%==0 (
  echo Building frontend...
  pushd frontend
  call npm install
  call npm run build
  popd
) else (
  echo WARNING: npm not found; skipping frontend build.
  echo   Install Node.js, then run:  cd frontend ^&^& npm install ^&^& npm run build
)

:: 3) API key check
if not exist .env (
  echo WARNING: No .env found. Copy .env.example to .env and set OPEN_TOPO_API_KEY.
  echo   STL generation will fail until this is configured.
)

:: 4) Run the server (serves the built frontend at http://127.0.0.1:8020)
echo Starting server on http://127.0.0.1:8020 ...
python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8020

echo Server stopped.
pause
endlocal
