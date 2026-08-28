@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo Starting Terrain STL Generator...

:: 1) Python virtual environment
if not exist .venv (
  echo Creating virtual environment...
  py -m venv .venv
)

:: 2) Backend deps — install only if requirements.txt changed since last install
set "PIP_STAMP=.venv\requirements.installed.txt"
set NEED_PIP=1
if exist "%PIP_STAMP%" (
  fc /b backend\requirements.txt "%PIP_STAMP%" >nul 2>&1 && set NEED_PIP=0
)
if !NEED_PIP!==1 (
  echo Installing backend requirements...
  .venv\Scripts\python.exe -m pip install -r backend\requirements.txt || goto :fail
  copy /y backend\requirements.txt "%PIP_STAMP%" >nul
) else (
  echo Backend requirements up to date.
)

:: 3) Frontend deps — install only if package-lock.json changed since last install
where npm >nul 2>nul
if not %ERRORLEVEL%==0 (
  echo ERROR: npm not found. Install Node.js, then rerun this script.
  goto :fail
)
set "NPM_STAMP=frontend\node_modules\.package-lock.stamp"
set NEED_NPM=1
if exist "%NPM_STAMP%" (
  fc /b frontend\package-lock.json "%NPM_STAMP%" >nul 2>&1 && set NEED_NPM=0
)
if !NEED_NPM!==1 (
  echo Installing frontend dependencies...
  pushd frontend
  call npm install || (popd & goto :fail)
  popd
  copy /y frontend\package-lock.json "%NPM_STAMP%" >nul
) else (
  echo Frontend dependencies up to date.
)

:: 4) API key check
if not exist .env (
  echo WARNING: No .env found. Copy .env.example to .env and set OPEN_TOPO_API_KEY.
  echo   STL generation will fail until this is configured.
)

:: 5) Launch backend and frontend in their own windows
echo Starting backend on http://127.0.0.1:8020 ...
start "TOPO2STL backend" cmd /k ".venv\Scripts\python.exe -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8020 --reload --reload-dir backend"

echo Starting frontend on http://localhost:5173 ...
start "TOPO2STL frontend" cmd /k "cd frontend && npm run dev"

echo Both windows launched. App: http://localhost:5173
endlocal
exit /b 0

:fail
echo.
echo Startup failed — see the error above.
pause
endlocal
exit /b 1
