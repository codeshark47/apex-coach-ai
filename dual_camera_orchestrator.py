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
    _nearest_complete_row,
    _find_grounded_reference_near,
)
from main import extract_video_landmarks
from kinematics import calculate_knee_bracing, calculate_trunk_lean, calculate_head_stability


def run_dual_camera_analysis(side_on_path: str, rear_view_path: str, output_dir: str = "output",
                              bowling_arm_override: str = None,
                              side_seed_point: tuple = None, side_seed_frame_index: int = 0,
                              rear_seed_point: tuple = None, rear_seed_frame_index: int = 0,
                              side_extra_seeds: list = None, rear_extra_seeds: list = None,
                              side_precomputed: dict = None, rear_precomputed: dict = None,
                              side_event_overrides: dict = None, rear_event_overrides: dict = None) -> dict:
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

    side_precomputed/rear_precomputed: optional result dicts already
    returned by orchestrator.extract_and_detect_events() for each stream —
    reuses them instead of re-running extraction, for the Streamlit UI
    flow that already ran that stage to show each stream's own BFC/FFC/BR
    confirmation before this function is called. Falls back to running
    extraction internally (the original behavior) when not given, so any
    other caller (CLI, tests) is unaffected.

    side_event_overrides/rear_event_overrides: optional coach-confirmed
    {"BFC": int, "FFC": int, "BR": int, "wrist_override_x": float|None,
    "wrist_override_y": float|None} per stream — see
    streamlit_app.render_stream_event_confirmation. BUG FIX (found during
    a full app audit): before this, Dual Camera went straight from upload
    to fully-automatic event detection with NO human review at all, while
    every real auto-detection failure mode this session found and fixed
    (wrong release frame despite "high confidence", FFC landing on an
    ordinary run-up stride, etc.) applies here exactly as much as it does
    to Single Camera — a human confirming the frame is what actually
    catches those, not a confidence score.

    Release height is computed from BOTH streams' wrist overrides (see
    the fallback logic below) — an earlier version of this function only
    ever computed it from the side stream with no way to correct a failed
    reading from the rear stream at all, found on real footage where the
    side stream's auto-tracked wrist failed and the coach had no way to
    fix it even though the rear stream was sitting right there.
    """
    os.makedirs(output_dir, exist_ok=True)

    if side_precomputed is not None and side_precomputed.get("status") == "success":
        side_df = side_precomputed["df"]
        side_fps = side_precomputed["fps"]
        bowling_arm = (bowling_arm_override if bowling_arm_override in ("left", "right")
                       else side_precomputed.get("bowling_arm") or "right")
        side_events = dict(side_precomputed["events"])
    else:
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

        side_events = embedded_detect_events(side_df, fps=side_fps, bowling_arm=bowling_arm,
                                             camera_angle="side_on")

    if side_event_overrides:
        side_events.update(side_event_overrides)

    lead_side = "left" if bowling_arm == "right" else "right"

    side_ffc_rows = side_df[side_df["frame"] == side_events["FFC"]]
    side_br_rows  = side_df[side_df["frame"] == side_events["BR"]]

    if side_ffc_rows.empty or side_br_rows.empty:
        return {"status": "failed", "stage": "side_frame_extraction", "message": "Kinematic frames missing from side-on stream."}

    knee_analysis     = calculate_knee_bracing(side_ffc_rows.iloc[0], lead_side=lead_side)
    knee_at_release   = calculate_knee_bracing(side_br_rows.iloc[0], lead_side=lead_side)
    lean_analysis     = calculate_trunk_lean(side_br_rows.iloc[0])
    side_height_ref = _find_grounded_reference_near(side_df, side_events["BR"], bowling_arm)
    if side_height_ref is None:
        side_height_ref = _nearest_complete_row(
            side_df, side_events["FFC"], ["NOSE_y", "LEFT_ANKLE_y", "RIGHT_ANKLE_y"]
        )
    # Coach-confirmed wrist/ball position, when given, overrides the
    # tracked landmark entirely — same reasoning as Single Camera (see
    # calculate_release_height_ratio_safe's docstring): verified real
    # MediaPipe mistracking during a fast, blurred release swing that no
    # automatic plausibility filter can catch.
    side_wrist_x = side_events.get("wrist_override_x")
    side_wrist_y = side_events.get("wrist_override_y")
    side_wrist_override_norm = (side_wrist_x, side_wrist_y) if side_wrist_x is not None and side_wrist_y is not None else None
    release_height    = calculate_release_height_ratio_safe(side_br_rows.iloc[0], bowling_arm=bowling_arm,
                                                              reference_row=side_height_ref,
                                                              wrist_override_norm=side_wrist_override_norm)

    # FFC-to-Release knee angle delta ("yielding knee" check from external
    # biomechanical audit) — same logic as Single Camera mode.
    knee_delta = None
    knee_delta_status = "unavailable"
    if knee_analysis.get("status") == "success" and knee_at_release.get("status") == "success":
        knee_delta = round(knee_at_release["degrees"] - knee_analysis["degrees"], 1)
        knee_delta_status = "yielding" if knee_delta < -5.0 else ("braced" if knee_delta >= 0 else "minor_yield")

    if rear_precomputed is not None and rear_precomputed.get("status") == "success":
        rear_df = rear_precomputed["df"]
        rear_fps = rear_precomputed["fps"]
        rear_events = dict(rear_precomputed["events"])
    else:
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

    if rear_event_overrides:
        rear_events.update(rear_event_overrides)

    hip_separation = calculate_hip_shoulder_separation(rear_df, rear_events["FFC"])
    head_stability = calculate_head_stability(rear_df, rear_events["BFC"], rear_events["BR"])

    # RELEASE HEIGHT FALLBACK — BUG FIX found on real footage: this metric
    # was ONLY ever computed from the side stream, with no way to correct
    # its wrist tracking from the rear stream at all. Release height only
    # needs vertical (y-axis) wrist/ankle/head positions, which are just
    # as valid from a rear-view stream as side-on (Single Camera mode
    # already computes this metric from a front/rear-only upload — nothing
    # about the calculation itself requires a side profile). If the side
    # stream's reading is unavailable (tracking failure on that specific
    # clip, or the coach never got a usable auto-track to correct), the
    # rear stream's own coach-corrected wrist point is now used instead of
    # just reporting "N/A" when a second, real, independently-tracked
    # camera angle was sitting right there with a possibly-good reading.
    rear_ffc_rows = rear_df[rear_df["frame"] == rear_events["FFC"]]
    rear_br_rows = rear_df[rear_df["frame"] == rear_events["BR"]]
    rear_release_height = {"status": "error", "ratio": None, "classification": "Not computed"}
    if not rear_br_rows.empty:
        rear_height_ref = _find_grounded_reference_near(rear_df, rear_events["BR"], bowling_arm)
        if rear_height_ref is None and not rear_ffc_rows.empty:
            rear_height_ref = _nearest_complete_row(
                rear_df, rear_events["FFC"], ["NOSE_y", "LEFT_ANKLE_y", "RIGHT_ANKLE_y"]
            )
        rear_wrist_x = rear_events.get("wrist_override_x")
        rear_wrist_y = rear_events.get("wrist_override_y")
        rear_wrist_override_norm = (rear_wrist_x, rear_wrist_y) if rear_wrist_x is not None and rear_wrist_y is not None else None
        rear_release_height = calculate_release_height_ratio_safe(rear_br_rows.iloc[0], bowling_arm=bowling_arm,
                                                                    reference_row=rear_height_ref,
                                                                    wrist_override_norm=rear_wrist_override_norm)

    release_height_source = "side"
    if release_height.get("status") != "success" and rear_release_height.get("status") == "success":
        release_height = rear_release_height
        release_height_source = "rear"

    # SIDE-ON ANNOTATED VIDEO
    raw_video = os.path.join(output_dir, "annotated_raw.mp4")
    generate_fail_safe_video(side_on_path, raw_video, side_df, side_events, bowling_arm=bowling_arm,
                              camera_angle="side_on")
    web_safe_video = transcode_to_h264(raw_video)

    # REAR-VIEW ANNOTATED VIDEO
    rear_raw_video = os.path.join(output_dir, "annotated_rear_raw.mp4")
    generate_fail_safe_video(rear_view_path, rear_raw_video, rear_df, rear_events, bowling_arm=bowling_arm,
                              camera_angle="front_or_rear")
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
            # Present when side_event_overrides was given (the Streamlit UI's
            # confirmation flow) — same (auto_guess, coach_confirmed) label
            # pairs Single Camera already logs, for Phase 2 training data.
            "ball_release_frame_auto_detected": side_events.get("BR_auto_detected"),
            "ball_release_auto_confidence": side_events.get("BR_auto_confidence"),
            "front_foot_contact_frame_auto_detected": side_events.get("FFC_auto_detected"),
            "back_foot_contact_frame_auto_detected": side_events.get("BFC_auto_detected"),
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
                "status": release_height.get("status", "error"),
                # "side" (preferred/default) or "rear" (used only because
                # the side stream's own reading failed and the rear
                # stream's coach-corrected wrist point produced a real one)
                "measured_from": release_height_source,
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
            },
            # BUG FIX (found during a broader audit): this used to only
            # exist at the TOP level of this return dict. Single Camera's
            # equivalent (orchestrator.py's run_complete_bowling_analysis)
            # nests it HERE, inside biomechanical_metrics — and
            # streamlit_app.py reads it via metrics.get("bowling_arm_detected")
            # where metrics IS biomechanical_metrics. For Dual Camera that
            # always silently returned None, meaning: (a) speed estimation
            # never received the already-confirmed bowling arm and fell
            # back to its own internal auto-detection instead — able to
            # disagree with the arm the rest of the report is built on —
            # and (b) the "Bowling arm (auto-detected/manually selected)"
            # caption never displayed at all. Kept at the top level too,
            # for anything else that might read it from there.
            "bowling_arm_detected": bowling_arm,
        },
        "annotated_video_output": web_safe_video,
        "rear_annotated_video_output": rear_web_safe_video
    }
