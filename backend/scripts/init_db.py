"""
Compatibility entry point. Creates the first admin only if missing.

Prefer `python backend/scripts/create_admin.py`. Set ADMIN_PASSWORD in .env first.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.scripts.create_admin import create_admin_user

if __name__ == "__main__":
    asyncio.run(create_admin_user())
