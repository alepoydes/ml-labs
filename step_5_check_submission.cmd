@echo off
uv run --directory "%~dp0." --locked python -m coursekit.cli check %*
exit /b %errorlevel%
