@echo off
rem Double-click to install OmniCortex on Windows. Finds Python 3.10 or newer (and
rem offers to install it), then opens the installer in your browser.
setlocal
cd /d "%~dp0"
set "PYEXE="
set "PYARG="
where py >nul 2>nul
if errorlevel 1 goto trypython
py -3 -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
if errorlevel 1 goto trypython
set "PYEXE=py"
set "PYARG=-3"
goto run

:trypython
rem "python" may be the Microsoft Store's placeholder (in WindowsApps): skip that one.
for /f "delims=" %%p in ('where python 2^>nul') do call :check "%%p"
if defined PYEXE goto run

echo.
echo OmniCortex needs Python 3.10 or newer, and none was found.
where winget >nul 2>nul
if errorlevel 1 goto manual
choice /m "Install Python 3.12 now with winget"
if errorlevel 2 goto manual
winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
echo.
echo Python is installed. Double-click "Install OmniCortex" again to continue.
pause
exit /b 0

:manual
echo Opening python.org. Install Python, tick "Add python.exe to PATH", then run this again.
start "" "https://www.python.org/downloads/windows/"
pause
exit /b 1

:check
echo %~1 | find /i "WindowsApps" >nul
if not errorlevel 1 goto :eof
if defined PYEXE goto :eof
"%~1" -c "import sys;sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>nul
if not errorlevel 1 set "PYEXE=%~1"
goto :eof

:run
echo Starting the OmniCortex installer in your browser (keep this window open)...
"%PYEXE%" %PYARG% scripts\installer.py %*
if errorlevel 1 pause
