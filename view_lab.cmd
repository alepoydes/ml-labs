@echo off
uv run --directory "%~dp0." --locked --extra gui python -m coursekit.gui %*
exit /b %errorlevel%
