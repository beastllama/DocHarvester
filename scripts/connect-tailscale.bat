@echo off
rem Double-click to share DocHarvester on Tailscale for Hermes. See connect-tailscale.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0connect-tailscale.ps1"
pause
