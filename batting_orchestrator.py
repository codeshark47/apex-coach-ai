"""
batting_orchestrator.py

Batting Analysis module — deliberately its OWN file, isolated from
orchestrator.py/dual_camera_orchestrator.py, per the explicit design
decision to add batting as a separate module rather than touch the
working, already-verified bowling pipeline. Mirrors orchestrator.py's
two-stage structure exactly (extract_and_detect_* then run_*_analysis)
so the Streamlit UI can show a confirmable event guess before paying for
the more expensive video-rendering stage — same reasoning as
orchestrator.extract_and_detect_events's docstring.

Single-camera only, by design: a coach stands behind the stumps filming
the batter with one phone, same physical setup already used for several
ball-tracking clips this project has. No 3D/dual-camera complexity is
needed for batting technique analysis (head position, foot alignment,
weight transfer, downswing plane, top-elbow) — all of it is measurable
from one reasonably side-on-ish view.
"""

import os

import pandas as pd

import monitoring
from main import extract_video_landmarks
from batting_events import detect_batting_hand, detect_batting_events
from batting_kinematics import (
    calculate_head_movement,
    calculate_front_foot_alignment,
    calculate_weight_transfer,
    calculate_downswing_plane,
    calculate_top_elbow_angle,
)


def extract_and_detect_batting_events(video_path: str,
                                       output_dir: str = "output",
                                       batting_hand_override: str = None,
                                       seed_point: tuple = None,
                                       seed_frame_index: int = 0,
                                       extra_seeds: list = None) -> dict:
    """
    STAGE 1+2: landmark extraction + leading-side detection + phase-event
    detection — everything needed to show the coach a confirmable
    stance/backlift/contact/follow-through guess before the full
    metrics+video stage runs. See run_batting_analysis, which calls this
    internally and accepts its output via `precomputed` to avoid
    re-running it.

    Returns {"status": "success", "df": DataFrame, "csv_path": str,
    "fps": float, "batting_hand": str, "events": dict} or
    {"status": "failed", "stage": str, "message": str}.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "batting_landmarks.csv")

    extraction = extract_video_landmarks(video_path, csv_path,
                                          seed_point=seed_point,
                                          seed_frame_index=seed_frame_index,
                                          extra_seeds=extra_seeds)

    if extraction["status"] == "error":
        return {
            "status": "failed",
            "stage": "perception",
            "message": extraction["error_message"],
        }

    df = pd.read_csv(csv_path)
    fps = extraction["fps"]

    events = detect_batting_events(df, fps)

    if batting_hand_override in ("left", "right"):
        batting_hand = batting_hand_override
    else:
        batting_hand = detect_batting_hand(df, events["STANCE"])

    return {
        "status": "success",
        "df": df,
        "csv_path": csv_path,
        "fps": fps,
        "batting_hand": batting_hand,
        "events": events,
    }


def run_batting_analysis(video_path: str,
                          output_dir: str = "output",
                          batting_hand_override: str = None,
                          seed_point: tuple = None,
                          seed_frame_index: int = 0,
                          extra_seeds: list = None,
                          precomputed: dict = None) -> dict:
    """
    Core batting orchestration loop — extracts landmarks, detects phase
    events, calculates all 5 batting metrics, generates an annotated
    video, and returns a unified payload in the same overall shape as
    orchestrator.run_complete_bowling_analysis (video_metadata,
    time_indices, biomechanical_metrics, annotated_video_output), so the
    existing PDF/history-saving code can be extended to handle either
    shape with minimal special-casing.

    precomputed: optional result dict already returned by
    extract_and_detect_batting_events, reused instead of re-running
    extraction — same pattern as the bowling pipeline, for callers (the
    Streamlit UI) that already ran stage 1+2 to show a confirmation step.
    """
    if precomputed is not None and precomputed.get("status") == "success":
        stage12 = precomputed
    else:
        stage12 = extract_and_detect_batting_events(
            video_path, output_dir=output_dir,
            batting_hand_override=batting_hand_override,
            seed_point=seed_point, seed_frame_index=seed_frame_index,
            extra_seeds=extra_seeds,
        )
    if stage12["status"] != "success":
        return stage12

    df = stage12["df"]
    fps = stage12["fps"]
    batting_hand = stage12["batting_hand"]
    events = stage12["events"]

    stance_frame = events["STANCE"]
    backlift_frame = events["BACKLIFT"]
    contact_frame = events["CONTACT"]

    contact_rows = df[df["frame"] == contact_frame]
    if contact_rows.empty:
        return {
            "status": "failed",
            "stage": "frame_extraction",
            "message": f"Contact frame {contact_frame} not found in landmark data.",
        }
    contact_row = contact_rows.iloc[0]

    try:
        head_movement = calculate_head_movement(df, stance_frame, contact_frame)
        front_foot = calculate_front_foot_alignment(contact_row, front_side=batting_hand)
        weight_transfer = calculate_weight_transfer(df, stance_frame, contact_frame, front_side=batting_hand)
        downswing_plane = calculate_downswing_plane(df, backlift_frame, contact_frame)
        top_elbow = calculate_top_elbow_angle(contact_row, top_hand_side=batting_hand)
    except Exception as e:
        monitoring.capture(e)
        return {"status": "failed", "stage": "metrics", "message": str(e)}

    raw_output_video = os.path.join(output_dir, "batting_annotated_raw.mp4")
    annotated_video_output = None
    try:
        from batting_video_overlay import render_batting_annotated_video
        import orchestrator as o
        render_batting_annotated_video(video_path, raw_output_video, df, events, batting_hand=batting_hand)
        annotated_video_output = o.transcode_to_h264(raw_output_video)
    except Exception as e:
        # Video annotation is a nice-to-have on top of the metrics, which
        # are the actual coaching value — a rendering failure shouldn't
        # take down the whole analysis. Same "don't fail outright"
        # reasoning as save_uploaded_video_capped's ffmpeg-missing path.
        monitoring.capture(e)

    def _val(d, key):
        return d.get(key)

    return {
        "status": "success",
        "analysis_type": "batting",
        "video_metadata": {
            "source_file": os.path.basename(video_path),
            "fps": fps,
            "total_frames": len(df),
        },
        "time_indices": {
            "stance_frame": stance_frame,
            "backlift_frame": backlift_frame,
            "contact_frame": contact_frame,
            "follow_through_frame": events["FOLLOW_THROUGH"],
            "contact_confidence": events.get("CONTACT_confidence", "high"),
            "contact_frame_auto_detected": events.get("CONTACT_auto_detected"),
            "stance_frame_auto_detected": events.get("STANCE_auto_detected"),
            "backlift_frame_auto_detected": events.get("BACKLIFT_auto_detected"),
        },
        "biomechanical_metrics": {
            "batting_hand_detected": batting_hand,
            "head_movement": {
                "value": _val(head_movement, "deviation_index"),
                "tier": head_movement.get("tier", "Unknown"),
                "status": head_movement.get("status", "error"),
            },
            "front_foot_alignment": {
                "degrees": _val(front_foot, "degrees"),
                "tier": front_foot.get("tier", "Unknown"),
                "status": front_foot.get("status", "error"),
            },
            "weight_transfer": {
                "percent": _val(weight_transfer, "percent"),
                "tier": weight_transfer.get("tier", "Unknown"),
                "status": weight_transfer.get("status", "error"),
            },
            "downswing_plane": {
                "degrees": _val(downswing_plane, "degrees"),
                "tier": downswing_plane.get("tier", "Unknown"),
                "status": downswing_plane.get("status", "error"),
            },
            "top_elbow_angle": {
                "degrees": _val(top_elbow, "degrees"),
                "tier": top_elbow.get("tier", "Unknown"),
                "status": top_elbow.get("status", "error"),
            },
        },
        "annotated_video_output": annotated_video_output,
    }
