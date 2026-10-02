@echo off
REM One-click launcher for the T58 web app on Windows. Double-click this file.
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
if not exist .venv (
  echo First run: creating a virtual environment, this takes a minute...
  %PY% -m venv .venv || goto :fail
)
call .venv\Scripts\activate.bat
if not exist .venv\.t58_deps_installed (
  echo Installing dependencies...
  python -m pip install --upgrade pip
  python -m pip install -r config\requirements.txt || goto :fail
  echo done> .venv\.t58_deps_installed
)
python run_web.py
goto :eof
:fail
echo.
echo Setup failed. Make sure Python 3.10 or newer is installed from https://www.python.org/downloads/
pause
