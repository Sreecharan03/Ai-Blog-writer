# test_vector.py
from __future__ import annotations

import os
import sys
import psycopg2
from pathlib import Path
from dotenv import load_dotenv
def _req(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v
def build_dsn() -> str:
    host = _req("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    dbname = _req("DB_NAME")
    user = _req("DB_USER")
    password = _req("DB_PASSWORD")
    sslmode = os.getenv("DB_SSLMODE", "require")
    return (
        f"host={host} port={port} dbname={dbname} "
        f"user={user} password={password} sslmode={sslmode}"
    )
def main() -> int:
    # Load .env from repo root
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        print("⚠️  .env not found at", env_path)

    dsn = build_dsn()
    print("Connecting with DSN (redacted password):")
    print(dsn.replace(os.getenv("DB_PASSWORD", ""), "<REDACTED>"))

    try:
        conn = psycopg2.connect(dsn, connect_timeout=8)
    except Exception as e:
        print("❌ DB connect failed:", repr(e))
        return 1

    try:
        with conn.cursor() as cur:
            # Basic info
            cur.execute("select version(), current_database(), current_user;")
            version, database, user = cur.fetchone()
            print("\n✅ Connected")
            print("DB:", database)
            print("User:", user)
            print("Version:", version)

            # Extensions check
            cur.execute(
                """
                select extname, extversion
                from pg_extension
                where extname in ('vector','pgcrypto')
                order by extname;
                """
            )
            rows = cur.fetchall()
            exts = {name: ver for (name, ver) in rows}
            print("\nExtensions found:", exts)

            print("vector_enabled:", "vector" in exts)
            print("pgcrypto_enabled:", "pgcrypto" in exts)

            # Vector sanity test (only if enabled)
            if "vector" in exts:
                # This will fail if vector is not installed
                cur.execute("select '[1,2,3]'::vector;")
                v = cur.fetchone()[0]
                print("\n✅ vector type works. Example:", v)
            else:
                print("\n⚠️ vector extension not enabled yet.")
                print("Run this in Supabase SQL Editor:")
                print("  create extension if not exists vector;")

    except Exception as e:
        print("❌ Query failed:", repr(e))
        return 2
    finally:
        conn.close()

    print("\n✅ Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
