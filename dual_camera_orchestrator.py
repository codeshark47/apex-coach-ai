import os
import pandas as pd
from orchestrator import (
    run_complete_bowling_analysis,
    embedded_detect_events,
    calculate_hip_shoulder_separation,
    calculate_release_height_ratio_safe,
    generate_fail_safe_video,
    transcode_to_h264,
    detect_bowling_arm,
)
from main import extract_video_landmarks
from kinematics import calculate_knee_bracing, calculate_trunk_lean, calculate_head_stability


def run_dual_camera_analysis(side_on_path: str, rear_view_path: str, output_dir: str = "output",
                              bowling_arm_override: str = None,
                              side_seed_point: tuple = None, side_seed_frame_index: int = 0,
                              rear_seed_point: tuple = None, rear_seed_frame_index: int = 0,
                              side_extra_seeds: list = None, rear_extra_seeds: list = None) -> dict:
    """
    SaaS Architecture Dual Camera Engine.
    Processes side-on and rear streams independently to bypass manual sync issues.

    side_seed_point/rear_seed_point: optional coach click identifying the
    bowler in each stream's own reference frame — the two camera angles
    are separate videos, so each needs its own seed. See
    main.extract_video_landmarks for details.

    side_extra_seeds/rear_extra_seeds: optional additional (frame_index,
    point) re-confirmations later in each stream — same purpose as
    Single Camera mode's extra_seeds, passed straight through.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("[DUAL-CAM CORE] Extracting side-on tracking vectors...")
    side_csv = os.path.join(output_dir, "landmarks_side.csv")
    side_extraction = extract_video_landmarks(side_on_path, side_csv,
                                               seed_point=side_seed_point,
                                               seed_frame_index=side_seed_frame_index,
                                               extra_seeds=side_extra_seeds)
    if side_extraction["status"] == "error":
        return {"status": "failed", "stage": "side_camera", "message": side_extraction["error_message"]}

    side_df  = pd.read_csv(side_csv)
    side_fps = side_extraction["fps"]

    # BOWLING ARM AUTO-DETECTION — detected from the side-on stream (the
    # primary view for delivery mechanics), then applied consistently to
    # both camera streams so left-arm and right-arm bowlers are both
    # measured correctly, matching the fix already applied to Single
    # Camera mode.
    if bowling_arm_override in ("left", "right"):
        bowling_arm = bowling_arm_override
    else:
        bowling_arm = detect_bowling_arm(side_df)
    lead_side = "left" if bowling_arm == "right" else "right"

    side_events = embedded_detect_events(side_df, fps=side_fps, bowling_arm=bowling_arm,
                                         camera_angle="side_on")

    side_ffc_rows = side_df[side_df["frame"] == side_events["FFC"]]
    side_br_rows  = side_df[side_df["frame"] == side_events["BR"]]

    if side_ffc_rows.empty or side_br_rows.empty:
        return {"status": "failed", "stage": "side_frame_extraction", "message": "Kinematic frames missing from side-on stream."}

    knee_analysis     = calculate_knee_bracing(side_ffc_rows.iloc[0], lead_side=lead_side)
    knee_at_release   = calculate_knee_bracing(side_br_rows.iloc[0], lead_side=lead_side)
    lean_analysis     = calculate_trunk_lean(side_br_rows.iloc[0])
    release_height    = calculate_release_height_ratio_safe(side_br_rows.iloc[0], bowling_arm=bowling_arm,
                                                              reference_row=side_ffc_rows.iloc[0])

    # FFC-to-Release knee angle delta ("yielding knee" check from external
    # biomechanical audit) — same logic as Single Camera mode.
    knee_delta = None
    knee_delta_status = "unavailable"
    if knee_analysis.get("status") == "success" and knee_at_release.get("status") == "success":
        knee_delta = round(knee_at_release["degrees"] - knee_analysis["degrees"], 1)
        knee_delta_status = "yielding" if knee_delta < -5.0 else ("braced" if knee_delta >= 0 else "minor_yield")

    print("[DUAL-CAM CORE] Extracting rear-view tracking vectors...")
    rear_csv = os.path.join(output_dir, "landmarks_rear.csv")
    rear_extraction = extract_video_landmarks(rear_view_path, rear_csv,
                                               seed_point=rear_seed_point,
                                               seed_frame_index=rear_seed_frame_index,
                                               extra_seeds=rear_extra_seeds)
    if rear_extraction["status"] == "error":
        return {"status": "failed", "stage": "rear_camera", "message": rear_extraction["error_message"]}

    rear_df  = pd.read_csv(rear_csv)
    rear_fps = rear_extraction["fps"]
    rear_events = embedded_detect_events(rear_df, fps=rear_fps, bowling_arm=bowling_arm,
                                         camera_angle="front_or_rear")

    hip_separation = calculate_hip_shoulder_separation(rear_df, rear_events["FFC"])
    head_stability = calculate_head_stability(rear_df, rear_events["BFC"], rear_events["BR"])

    # SIDE-ON ANNOTATED VIDEO
    raw_video = os.path.join(output_dir, "annotated_raw.mp4")
    generate_fail_safe_video(side_on_path, raw_video, side_df, side_events, bowling_arm=bowling_arm)
    web_safe_video = transcode_to_h264(raw_video)

    # REAR-VIEW ANNOTATED VIDEO
    rear_raw_video = os.path.join(output_dir, "annotated_rear_raw.mp4")
    generate_fail_safe_video(rear_view_path, rear_raw_video, rear_df, rear_events, bowling_arm=bowling_arm)
    rear_web_safe_video = transcode_to_h264(rear_raw_video)

    def clean_numeric(val):
        import numpy as np
        if val is None or pd.isna(val): return None
        if isinstance(val, (np.float64, np.float32)): return float(val)
        return val

    trunk_lean_val  = clean_numeric(lean_analysis.get("trunk_lean_degrees") or lean_analysis.get("degrees"))
    knee_bracing_val = clean_numeric(knee_analysis.get("front_knee_angle") or knee_analysis.get("degrees"))

    return {
        "status": "success",
        "camera_mode": "dual",
        "bowling_arm_detected": bowling_arm,
        "video_metadata": {
            "source_file": os.path.basename(side_on_path),
            "fps": side_fps,
            "total_frames": len(side_df)
        },
        "time_indices": {
            "back_foot_contact_frame": int(side_events["BFC"]),
            "front_foot_contact_frame": int(side_events["FFC"]),
            "ball_release_frame": int(side_events["BR"]),
            "ball_release_confidence": side_events.get("BR_confidence", "high"),
            "ball_release_plausible_fraction": side_events.get("BR_plausible_fraction", 1.0),
        },
        "biomechanical_metrics": {
            "trunk_lean": {
                "degrees": trunk_lean_val,
                "tier": lean_analysis.get("tier", "Unknown"),
                "status": "success" if trunk_lean_val is not None else "error"
            },
            "front_knee_bracing": {
                "degrees": knee_bracing_val,
                "tier": knee_analysis.get("tier", "Unknown"),
                "status": "success" if knee_bracing_val is not None else "error",
                "degrees_at_release": knee_at_release.get("degrees"),
                "yield_delta_degrees": knee_delta,
                "yield_status": knee_delta_status
            },
            "release_height": {
                "ratio": clean_numeric(release_height.get("ratio")),
                "classification": release_height.get("classification", "Unknown"),
                "status": release_height.get("status", "error")
            },
            "hip_shoulder_separation": {
                "degrees": clean_numeric(hip_separation.get("degrees")),
                "tier": hip_separation.get("tier", "Unknown"),
                "status": hip_separation.get("status", "error")
            },
            "head_stability": {
                "value": clean_numeric(head_stability.get("deviation_index") or head_stability.get("value")),
                "tier": head_stability.get("tier", "Unknown"),
                "status": head_stability.get("status", "error")
            }
        },
        "annotated_video_output": web_safe_video,
        "rear_annotated_video_output": rear_web_safe_video
    }
