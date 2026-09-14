@echo off
uv run --directory "%~dp0." --locked python -m coursekit.cli new %*
exit /b %errorlevel%
