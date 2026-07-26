"""
tools/export_training_data.py

Turns the (auto_detected, coach_confirmed) label pairs that streamlit_app.py
has been silently logging into every saved session — real ground truth,
collected for free as a side effect of normal coaching use — into a flat,
versioned CSV. This is the raw material for a future release-point/event
correction model; before this script, it only ever existed buried inside
each session's `metrics` JSONB column, unusable for training anything.

Admin-only. Reads across EVERY coach's sessions (not scoped to one coach —
profile_store's per-coach functions don't apply here), using the same
Supabase secret-key client as the rest of the app. Run this from your own
machine, never expose it inside the Streamlit app itself:

    python tools/export_training_data.py [--out training_data/export.csv]

Requires SUPABASE_URL / SUPABASE_KEY (see profile_store.py's setup notes).
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import profile_store as store

FIELDNAMES = [
    "session_id", "athlete_id", "session_date", "camera_mode", "fps",
    "camera_angle_confirmed",
    "br_auto_detected", "br_auto_confidence", "br_coach_confirmed", "br_corrected",
    "ffc_auto_detected", "ffc_coach_confirmed", "ffc_corrected",
    "bfc_auto_detected", "bfc_coach_confirmed", "bfc_corrected",
    "wrist_point_corrected", "wrist_x", "wrist_y",
]


def _fetch_all_sessions(client, page_size: int = 1000) -> list:
    """Supabase caps a single response at page_size rows — page through
    with .range() until a page comes back short, rather than assuming
    every project will always stay under one page forever."""
    rows = []
    start = 0
    while True:
        result = (
            client.table("sessions")
            .select("id, athlete_id, session_date, camera_mode, fps, metrics")
            .order("session_date")
            .range(start, start + page_size - 1)
            .execute()
        )
        page = result.data or []
        rows.extend(page)
        if len(page) < page_size:
            break
        start += page_size
    return rows


def _event_pair(metrics: dict, key: str, has_confidence: bool) -> dict:
    conf = metrics.get(key) or {}
    auto = conf.get("auto_detected")
    confirmed = conf.get("coach_confirmed")
    corrected = (auto != confirmed) if auto is not None and confirmed is not None else None
    out = {"auto_detected": auto, "coach_confirmed": confirmed, "corrected": corrected}
    if has_confidence:
        out["auto_confidence"] = conf.get("auto_confidence")
    return out


def flatten_session(row: dict) -> dict:
    metrics = row.get("metrics") or {}

    br = _event_pair(metrics, "_release_frame_confirmed", has_confidence=True)
    ffc = _event_pair(metrics, "_ffc_frame_confirmed", has_confidence=False)
    bfc = _event_pair(metrics, "_bfc_frame_confirmed", has_confidence=False)

    wrist_point = metrics.get("_wrist_point_corrected")
    wrist_x, wrist_y = (wrist_point[0], wrist_point[1]) if wrist_point else (None, None)

    return {
        "session_id": row.get("id"),
        "athlete_id": row.get("athlete_id"),
        "session_date": row.get("session_date"),
        "camera_mode": row.get("camera_mode"),
        "fps": row.get("fps"),
        "camera_angle_confirmed": metrics.get("_camera_angle_confirmed"),
        "br_auto_detected": br["auto_detected"],
        "br_auto_confidence": br["auto_confidence"],
        "br_coach_confirmed": br["coach_confirmed"],
        "br_corrected": br["corrected"],
        "ffc_auto_detected": ffc["auto_detected"],
        "ffc_coach_confirmed": ffc["coach_confirmed"],
        "ffc_corrected": ffc["corrected"],
        "bfc_auto_detected": bfc["auto_detected"],
        "bfc_coach_confirmed": bfc["coach_confirmed"],
        "bfc_corrected": bfc["corrected"],
        "wrist_point_corrected": wrist_point is not None,
        "wrist_x": wrist_x,
        "wrist_y": wrist_y,
    }


def _print_summary(flat_rows: list):
    total = len(flat_rows)
    print(f"\n{total} session(s) exported.\n")
    for label, key in [("Ball release (BR)", "br_corrected"),
                        ("Front foot contact (FFC)", "ffc_corrected"),
                        ("Back foot contact (BFC)", "bfc_corrected")]:
        labeled = [r[key] for r in flat_rows if r[key] is not None]
        if not labeled:
            print(f"  {label}: no labeled sessions yet")
            continue
        corrected = sum(1 for v in labeled if v)
        print(f"  {label}: {len(labeled)} labeled, {corrected} corrected "
              f"({corrected / len(labeled):.0%} auto-tracking miss rate)")
    wrist_labeled = [r["wrist_point_corrected"] for r in flat_rows]
    wrist_corrected = sum(1 for v in wrist_labeled if v)
    print(f"  Wrist/release point: {len(wrist_labeled)} sessions, {wrist_corrected} corrected "
          f"({(wrist_corrected / total if total else 0):.0%} auto-tracking miss rate)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_out = f"training_data/export_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
    parser.add_argument("--out", default=default_out, help="Output CSV path")
    args = parser.parse_args()

    client = store.get_client()
    rows = _fetch_all_sessions(client)
    flat_rows = [flatten_session(r) for r in rows]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(flat_rows)

    print(f"Wrote {args.out}")
    _print_summary(flat_rows)


if __name__ == "__main__":
    main()
