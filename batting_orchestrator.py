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
from one phone.

DUAL CAMERA ANGLE (2026-08-03): a coach filming in the nets often can't
get a side-on shot at all (the net physically obstructs it), so this
module now auto-detects side-on vs. front-on/rear-on the same way
orchestrator.py already does for bowling (reusing
camera_angle_detection.py wholesale — it's already generic, not
bowling-specific). Unlike bowling, camera_angle here does NOT gate which
metrics run — every metric is always computed (see
batting_kinematics.py's module docstring for why weight_transfer/
downswing_plane specifically are still computed either way but flagged
as reduced-confidence under front_or_rear rather than silently hidden).
front_foot_alignment and the falling-over compound check are unaffected
by camera angle at all — they derive their own reference axis per clip
from body geometry (see batting_kinematics._derive_batting_axes).
"""

import os

import cv2
import pandas as pd

import camera_angle_detection as cad
import monitoring
import orchestrator as o
from main import extract_video_landmarks
from batting_events import detect_batting_hand, detect_batting_events
from batting_kinematics import (
    calculate_head_movement,
    calculate_front_foot_alignment,
    calculate_weight_transfer,
    calculate_downswing_plane,
    calculate_top_elbow_angle,
    calculate_front_knee_flexion,
    detect_falling_over_risk,
)

# Metrics whose formula assumes the batter's forward stride projects onto
# a real image axis (true for side-on filming; foreshortened into the
# depth axis for front-on/rear-on) — surfaced as a caveat, not gated, when
# the detected/confirmed camera angle is front_or_rear. See this module's
# docstring and batting_kinematics.py's for the full reasoning.
VIEW_SENSITIVE_METRIC_KEYS = ["batting_weight_transfer", "batting_downswing_plane"]


def extract_and_detect_batting_events(video_path: str,
                                       output_dir: str = "output",
                                       batting_hand_override: str = None,
                                       seed_point: tuple = None,
                                       seed_frame_index: int = 0,
                                       extra_seeds: list = None,
                                       camera_angle_override: str = None) -> dict:
    """
    STAGE 1+2: landmark extraction + leading-side detection + phase-event
    detection + camera-angle estimate — everything needed to show the
    coach a confirmable stance/backlift/contact/follow-through guess (and
    filming-angle guess) before the full metrics+video stage runs. See
    run_batting_analysis, which calls this internally and accepts its
    output via `precomputed` to avoid re-running it.

    camera_angle_override: 'side_on' | 'front_or_rear' | 'uncertain', when
    the coach has manually confirmed the filming angle. None (default)
    auto-detects, sampling several frames between STANCE and BACKLIFT
    (where the batter is set/composed, not yet deep in a fast swing) and
    taking a majority vote — same reasoning as
    orchestrator.extract_and_detect_events's identical bowling-side logic:
    verified there that a single arbitrary frame's incidental pose can
    misclassify an otherwise-consistent filming angle.

    Returns {"status": "success", "df": DataFrame, "csv_path": str,
    "fps": float, "batting_hand": str, "camera_angle": str,
    "angle_estimate": AngleEstimate, "events": dict} or
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

    angle_estimate = None
    if camera_angle_override in ("side_on", "front_or_rear", "uncertain"):
        camera_angle = camera_angle_override
    else:
        cap_probe = cv2.VideoCapture(video_path)
        probe_w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
        probe_h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
        cap_probe.release()
        lo, hi = events["STANCE"], max(events["STANCE"], events["BACKLIFT"])
        sample_frames = sorted(set(int(round(lo + f * (hi - lo))) for f in (0.0, 0.25, 0.5, 0.75, 1.0)))
        votes, ratios = {}, []
        for fidx in sample_frames:
            est = cad.estimate_camera_angle(df, fidx, probe_w, probe_h)
            if est.angle != "unavailable":
                votes[est.angle] = votes.get(est.angle, 0) + 1
            if est.ratio is not None:
                ratios.append(est.ratio)
        if votes:
            camera_angle = max(votes, key=votes.get)
            angle_estimate = cad.AngleEstimate(
                camera_angle, (ratios[len(ratios) // 2] if ratios else None),
                f"Auto-detected from {len(votes)} frame(s) sampled between stance and backlift."
            )
        else:
            camera_angle = "uncertain"
            angle_estimate = cad.AngleEstimate("uncertain", None, "Could not confidently sample any frame.")

    return {
        "status": "success",
        "df": df,
        "csv_path": csv_path,
        "fps": fps,
        "batting_hand": batting_hand,
        "camera_angle": camera_angle,
        "angle_estimate": angle_estimate,
        "events": events,
    }


def run_batting_analysis(video_path: str,
                          output_dir: str = "output",
                          batting_hand_override: str = None,
                          seed_point: tuple = None,
                          seed_frame_index: int = 0,
                          extra_seeds: list = None,
                          camera_angle_override: str = None,
                          shot_played: str = None,
                          ball_line: str = None,
                          precomputed: dict = None) -> dict:
    """
    Core batting orchestration loop — extracts landmarks, detects phase
    events, calculates all batting metrics, generates an annotated video,
    and returns a unified payload in the same overall shape as
    orchestrator.run_complete_bowling_analysis (video_metadata,
    time_indices, biomechanical_metrics, annotated_video_output), so the
    existing PDF/history-saving code can be extended to handle either
    shape with minimal special-casing.

    shot_played: coach-selected named shot (one of
    batting_kinematics.SHOT_TARGET_CENTERS_DEGREES's keys, or None) — see
    calculate_front_foot_alignment for why this needs a coach input
    rather than being detected (no ball-tracking data exists yet to infer
    the intended shot direction automatically).

    ball_line: coach-reported delivery line ("off"/"middle"/"leg") — feeds
    the falling-over compound-fault check only; unrelated to shot_played.

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
            extra_seeds=extra_seeds, camera_angle_override=camera_angle_override,
        )
    if stage12["status"] != "success":
        return stage12

    df = stage12["df"]
    fps = stage12["fps"]
    batting_hand = stage12["batting_hand"]
    camera_angle = stage12.get("camera_angle", "uncertain")
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
        front_foot = calculate_front_foot_alignment(df, stance_frame, contact_frame,
                                                      front_side=batting_hand, shot_played=shot_played)
        weight_transfer = calculate_weight_transfer(df, stance_frame, contact_frame, front_side=batting_hand)
        downswing_plane = calculate_downswing_plane(df, backlift_frame, contact_frame)
        top_elbow = calculate_top_elbow_angle(contact_row, top_hand_side=batting_hand)
        front_knee_flexion = calculate_front_knee_flexion(contact_row, front_side=batting_hand)
        xfactor_separation = o.calculate_hip_shoulder_separation(df, contact_frame)
        falling_over = detect_falling_over_risk(df, stance_frame, contact_frame,
                                                  front_side=batting_hand, ball_line=ball_line)
    except Exception as e:
        monitoring.capture(e)
        return {"status": "failed", "stage": "metrics", "message": str(e)}

    raw_output_video = os.path.join(output_dir, "batting_annotated_raw.mp4")
    annotated_video_output = None
    try:
        from batting_video_overlay import render_batting_annotated_video
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

    view_confidence_caveats = list(VIEW_SENSITIVE_METRIC_KEYS) if camera_angle == "front_or_rear" else []

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
        "camera_angle": camera_angle,
        "view_confidence_caveats": view_confidence_caveats,
        "shot_played": shot_played,
        "ball_line": ball_line,
        "falling_over_risk": {
            "flagged": falling_over.get("flagged", False),
            "reason": falling_over.get("reason"),
            "head_shift_pct": falling_over.get("head_shift_pct"),
            "foot_cross_pct": falling_over.get("foot_cross_pct"),
            "status": falling_over.get("status", "error"),
        },
        "biomechanical_metrics": {
            "batting_hand_detected": batting_hand,
            "head_movement": {
                "value": _val(head_movement, "deviation_index"),
                "tier": head_movement.get("tier", "Unknown"),
                "status": head_movement.get("status", "error"),
            },
            "front_foot_alignment": {
                "signed_degrees": _val(front_foot, "signed_degrees"),
                "side": front_foot.get("side"),
                "deviation_degrees": _val(front_foot, "deviation_degrees"),
                "target_shot": front_foot.get("target_shot"),
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
            "front_knee_flexion": {
                "degrees": _val(front_knee_flexion, "degrees"),
                "tier": front_knee_flexion.get("tier", "Unknown"),
                "status": front_knee_flexion.get("status", "error"),
            },
            "xfactor_separation": {
                "degrees": _val(xfactor_separation, "degrees"),
                "tier": xfactor_separation.get("tier", "Unknown"),
                "status": xfactor_separation.get("status", "error"),
            },
        },
        "annotated_video_output": annotated_video_output,
    }
