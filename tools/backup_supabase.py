"""
tools/backup_supabase.py

Manual backup of the three real-data tables (athletes, teams, sessions)
to a single timestamped JSON file — a free safety net that works
regardless of Supabase plan tier. Automatic daily backups / point-in-time
recovery are a paid-tier Supabase feature; this script doesn't depend on
that, and is a genuinely separate copy either way — a backup that only
ever lives inside the same project it's backing up doesn't protect
against that project having a problem.

Run manually, or schedule it (Windows Task Scheduler, cron, etc.) to run
daily:

    python tools/backup_supabase.py [--out-dir backups]

Never deletes anything — every run adds one new timestamped file. Prune
old ones yourself if disk space becomes a concern.

Requires SUPABASE_URL / SUPABASE_KEY (see profile_store.py's setup notes).
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import profile_store as store

TABLES = ["athletes", "teams", "sessions"]


def _fetch_all(client, table: str, page_size: int = 1000) -> list:
    """Pages through with .range() rather than assuming a table will
    always stay under Supabase's single-response row cap."""
    rows = []
    start = 0
    while True:
        result = client.table(table).select("*").range(start, start + page_size - 1).execute()
        page = result.data or []
        rows.extend(page)
        if len(page) < page_size:
            break
        start += page_size
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="backups", help="Directory to write the backup file into")
    args = parser.parse_args()

    client = store.get_client()
    backup = {"taken_at": datetime.now(timezone.utc).isoformat()}
    for table in TABLES:
        rows = _fetch_all(client, table)
        backup[table] = rows
        print(f"  {table}: {len(rows)} row(s)")

    os.makedirs(args.out_dir, exist_ok=True)
    filename = os.path.join(
        args.out_dir,
        f"backup_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json",
    )
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(backup, f, indent=2, default=str)

    print(f"Wrote {filename}")


if __name__ == "__main__":
    main()
