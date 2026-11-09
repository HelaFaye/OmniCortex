@echo off
rem OmniCortex Remote: double-click to start. This computer becomes a HOST that a controller
rem (OmniCortex Studio, or an MCP client) can use after you pair it.
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 -m omnicortex_remote %*
) else (
  python -m omnicortex_remote %*
)
if errorlevel 1 pause
