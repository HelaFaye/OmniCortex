@echo off
rem Start OmniCortex Studio (if it is not running) and open it in the browser.
set HERE=%~dp0
if exist "%HERE%venv\Scripts\python.exe" (set PY=%HERE%venv\Scripts\python.exe) else (set PY=python)
"%PY%" "%HERE%scripts\launch.py" %*
