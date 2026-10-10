# Shares DocHarvester on your Tailscale network, so Hermes on your other computers can connect.
# Run once on the computer that runs DocHarvester. Safe to run again.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$tailscale = (Get-Command tailscale -ErrorAction SilentlyContinue).Source
if (-not $tailscale -and (Test-Path "C:\Program Files\Tailscale\tailscale.exe")) {
    $tailscale = "C:\Program Files\Tailscale\tailscale.exe"
}
if (-not $tailscale) { Write-Host "Tailscale is not installed. Install it from tailscale.com, sign in, then run this again."; exit 1 }
if (-not (Test-Path ".env")) { Write-Host "No .env file here. Set up DocHarvester first (copy env.example to .env)."; exit 1 }

$name = ((& $tailscale status --json | ConvertFrom-Json).Self.DNSName).TrimEnd(".")
if (-not $name) { Write-Host "Tailscale is not signed in. Open Tailscale, sign in, then run this again."; exit 1 }

# Tailnet only (not the internet). If Tailscale prints a link to enable HTTPS, open it, click Enable, run this again.
& $tailscale serve --bg 8000
if ($LASTEXITCODE -ne 0) { exit 1 }

$url = "https://$name"
$lines = @(Get-Content ".env" | Where-Object { $_ -notmatch "^PUBLIC_URL=" }) + "PUBLIC_URL=$url"
Set-Content ".env" $lines
docker compose up -d backend

Write-Host ""
Write-Host "Done. DocHarvester is on your Tailscale network at $url"
Write-Host "Next: open DocHarvester, click your avatar, API tokens, Create token, Copy config, paste into Hermes."
