#!/usr/bin/env bash
# Shares DocHarvester on your Tailscale network, so Hermes on your other computers can connect.
# Run once on the computer that runs DocHarvester. Safe to run again.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v tailscale >/dev/null || { echo "Tailscale is not installed. Install it from tailscale.com, sign in, then run this again."; exit 1; }
[ -f .env ] || { echo "No .env file here. Set up DocHarvester first (copy env.example to .env)."; exit 1; }

name=$(tailscale status --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')
[ -n "$name" ] || { echo "Tailscale is not signed in. Sign in, then run this again."; exit 1; }

# Tailnet only (not the internet). If Tailscale prints a link to enable HTTPS, open it, click Enable, run this again.
tailscale serve --bg 8000

url="https://$name"
grep -v '^PUBLIC_URL=' .env > .env.tmp || true
echo "PUBLIC_URL=$url" >> .env.tmp
mv .env.tmp .env
docker compose up -d backend

echo
echo "Done. DocHarvester is on your Tailscale network at $url"
echo "Next: open DocHarvester, click your avatar, API tokens, Create token, Copy config, paste into Hermes."
