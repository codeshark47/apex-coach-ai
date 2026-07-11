import os
import pandas as pd
from orchestrator import (
    run_complete_bowling_analysis,
    embedded_detect_events,
    calculate_hip_shoulder_separation,
    calculate_release_height_ratio_safe,
    generate_fail_safe_video,
    transcode_to_h264,
)
from main import extract_video_landmarks
from kinematics import calculate_knee_bracing, calculate_trunk_lean, calculate_head_stability


def run_dual_camera_analysis(side_on_path: str, rear_view_path: str, output_dir: str = "output") -> dict:
    """
    SaaS Architecture Dual Camera Engine.
    Processes side-on and rear streams independently to bypass manual sync issues.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("[DUAL-CAM CORE] Extracting side-on tracking vectors...")
    side_csv = os.path.join(output_dir, "landmarks_side.csv")
    side_extraction = extract_video_landmarks(side_on_path, side_csv)
    if side_extraction["status"] == "error":
        return {"status": "failed", "stage": "side_camera", "message": side_extraction["error_message"]}

    side_df  = pd.read_csv(side_csv)
    side_fps = side_extraction["fps"]
    side_events = embedded_detect_events(side_df, fps=side_fps)

    side_ffc_rows = side_df[side_df["frame"] == side_events["FFC"]]
    side_br_rows  = side_df[side_df["frame"] == side_events["BR"]]

    if side_ffc_rows.empty or side_br_rows.empty:
        return {"status": "failed", "stage": "side_frame_extraction", "message": "Kinematic frames missing from side-on stream."}

    knee_analysis  = calculate_knee_bracing(side_ffc_rows.iloc[0])
    lean_analysis  = calculate_trunk_lean(side_br_rows.iloc[0])
    release_height = calculate_release_height_ratio_safe(side_br_rows.iloc[0])

    print("[DUAL-CAM CORE] Extracting rear-view tracking vectors...")
    rear_csv = os.path.join(output_dir, "landmarks_rear.csv")
    rear_extraction = extract_video_landmarks(rear_view_path, rear_csv)
    if rear_extraction["status"] == "error":
        return {"status": "failed", "stage": "rear_camera", "message": rear_extraction["error_message"]}

    rear_df  = pd.read_csv(rear_csv)
    rear_fps = rear_extraction["fps"]
    rear_events = embedded_detect_events(rear_df, fps=rear_fps)

    hip_separation = calculate_hip_shoulder_separation(rear_df, rear_events["FFC"])
    head_stability = calculate_head_stability(rear_df, rear_events["BFC"], rear_events["BR"])

    raw_video = os.path.join(output_dir, "annotated_raw.mp4")
    generate_fail_safe_video(side_on_path, raw_video, side_df, side_events)
    web_safe_video = transcode_to_h264(raw_video)

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
        "video_metadata": {
            "source_file": os.path.basename(side_on_path),
            "fps": side_fps,
            "total_frames": len(side_df)
        },
        "time_indices": {
            "back_foot_contact_frame": int(side_events["BFC"]),
            "front_foot_contact_frame": int(side_events["FFC"]),
            "ball_release_frame": int(side_events["BR"])
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
                "status": "success" if knee_bracing_val is not None else "error"
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
        "annotated_video_output": web_safe_video
    }
