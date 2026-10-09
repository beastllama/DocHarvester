#!/usr/bin/env python3
"""
Create the first admin user for DocHarvester.

Set ADMIN_PASSWORD in .env (at least 12 characters) before running.
ADMIN_EMAIL is optional and defaults to admin@docharvester.com.

Safe to re-run: if the admin already exists, the password is NOT changed.
"""
import asyncio
import os
import sys
from pathlib import Path

# Repo root holds the `backend` package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sqlalchemy import select
from backend.database import AsyncSessionLocal
from backend.models import User
from backend.api.auth import get_password_hash

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@docharvester.com")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")


async def create_admin_user():
    """Create the admin user if it does not exist. Never changes an existing password."""
    if len(ADMIN_PASSWORD) < 12:
        print("ADMIN_PASSWORD is missing or shorter than 12 characters. Set it in .env and run again.")
        sys.exit(1)

    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(User).where(User.email == ADMIN_EMAIL))
            if result.scalar_one_or_none():
                print(f"Admin '{ADMIN_EMAIL}' already exists. Password was not changed.")
                return

            session.add(User(
                email=ADMIN_EMAIL,
                hashed_password=get_password_hash(ADMIN_PASSWORD),
                full_name="Admin User",
                is_active=True,
                is_admin=True,
            ))
            await session.commit()
            print(f"Admin '{ADMIN_EMAIL}' created.")
    except Exception as e:
        print(f"Error creating admin user: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(create_admin_user())
