import numpy as np
import pandas as pd

def calculate_knee_bracing(row: pd.Series, lead_side: str = "left") -> dict:
    """
    Computes the 2D angle of the lead knee joint via Law of Cosines.
    lead_side: which side is the LEAD (front) leg — 'left' for a right-arm
    bowler's standard action, 'right' for a left-arm bowler (the lead leg
    is always opposite the bowling arm). Defaults to 'left' to preserve
    existing behavior for any caller not yet passing this explicitly.
    """
    try:
        side = "LEFT" if lead_side == "left" else "RIGHT"
        h = np.array([float(row[f"{side}_HIP_x"]), float(row[f"{side}_HIP_y"])])
        k = np.array([float(row[f"{side}_KNEE_x"]), float(row[f"{side}_KNEE_y"])])
        a = np.array([float(row[f"{side}_ANKLE_x"]), float(row[f"{side}_ANKLE_y"])])

        kh, ka = h - k, a - k
        denom = np.linalg.norm(kh) * np.linalg.norm(ka)
        if denom == 0 or np.isnan(denom):
            # None, not 0.0 — a fabricated 0-degree reading here would fold
            # into the "Critical" band and read as a real (terrible) result
            # instead of "we couldn't measure this." Same reasoning as the
            # trunk_lean/head_stability fix below: a placeholder number must
            # never look like a genuine measurement to classify()/data_quality.
            return {"degrees": None, "tier": "Tracking Drop", "status": "error"}

        cos_theta = np.dot(kh, ka) / denom
        angle = round(float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))), 1)

        if angle >= 165.0: tier = "Elite Rigid Extension"
        elif angle >= 145.0: tier = "Moderate Flexion"
        else: tier = "Collapsing Knee Joint"
        return {"degrees": angle, "tier": tier, "status": "success"}
    except Exception:
        return {"degrees": None, "tier": "Data Deficit", "status": "error"}

def calculate_trunk_lean(row: pd.Series) -> dict:
    """Measures dynamic lateral deviation of the spinal column away from vertical orientation."""
    try:
        mid_hip_x = (float(row["LEFT_HIP_x"]) + float(row["RIGHT_HIP_x"])) / 2
        mid_hip_y = (float(row["LEFT_HIP_y"]) + float(row["RIGHT_HIP_y"])) / 2
        mid_sh_x = (float(row["LEFT_SHOULDER_x"]) + float(row["RIGHT_SHOULDER_x"])) / 2
        mid_sh_y = (float(row["LEFT_SHOULDER_y"]) + float(row["RIGHT_SHOULDER_y"])) / 2
        
        dx = mid_sh_x - mid_hip_x
        dy = mid_hip_y - mid_sh_y  # Invert image coordinate axis to match standard cartesian space
        
        angle = round(float(np.degrees(np.arctan2(np.abs(dx), dy))), 1)
        if np.isnan(angle):
            # BUG FIX: was returning 0.0 here — indistinguishable from a
            # genuinely perfect 0-degree upright posture (the actual best
            # possible score for this metric). A real tracking failure was
            # silently being reported to the coach as an ideal result, and
            # data_quality.py's missing-metric check (which only looks for
            # None) couldn't catch it either since 0.0 isn't None. Verified
            # directly against a real low-confidence session where this
            # exact path fired and produced a fake "Optimal Upright Posture."
            return {"degrees": None, "tier": "Tracking Drop", "status": "error"}

        # FIX: a value above 90 degrees is physically nonsensical for "lean
        # from vertical" (it would mean the torso axis is pointing more
        # sideways/upside-down than upright), and indicates a tracking
        # error (e.g. shoulder/hip landmark swap or an odd momentary
        # detection) rather than a real extreme lean. Flag it as unreliable
        # instead of reporting a false, alarming number like 168 degrees.
        if angle > 90.0:
            return {"degrees": angle, "tier": "Tracking Unreliable (implausible angle)", "status": "error"}

        tier = "Optimal Upright Posture" if angle <= 8.0 else "Excessive Lateral Flexion"
        return {"degrees": angle, "tier": tier, "status": "success"}
    except Exception:
        return {"degrees": None, "tier": "Data Deficit", "status": "error"}

def calculate_head_stability(df: pd.DataFrame, start_frame: int, end_frame: int) -> dict:
    """
    Tracks lateral head-position stability during the delivery stride,
    normalized by shoulder width in the same window.

    BUG FIX (2026-08-05, found during a full audit of every biomechanical
    calculation after tonight's release-height fix): this used to measure
    raw std(nose_x) — a normalized-0-1 IMAGE coordinate — against a fixed
    absolute threshold (0.015), with no accounting for camera distance.
    The exact same real head movement produces a LARGER raw pixel
    deviation when the bowler is filmed closer to the camera than when
    filmed further away — pure framing, not real technique. Same class
    of bug as release_height's original body-height issue (a fixed
    threshold applied to a camera-distance-dependent raw measurement),
    just not discovered until this audit.

    Fix: divide each frame's nose_x by that SAME frame's shoulder width
    before computing the deviation — the same same-frame self-
    normalization batting_kinematics.calculate_weight_transfer already
    does correctly with stance width. A person's shoulder width in the
    image scales with camera distance in exactly the same way head
    movement does, so the ratio cancels that dependence out.

    The tier threshold below is a first-pass, explicitly PROVISIONAL
    estimate (a rough conversion of the old, itself-never-validated 0.015
    absolute threshold against a typical shoulder width) — there is not
    yet real multi-clip evidence to properly re-tune it, hence
    recalibration_pending=True on a successful reading. See
    orchestrator.calculate_release_height_ratio_safe's own
    recalibration_pending for the identical reasoning.
    """
    try:
        window = df[(df["frame"] >= start_frame) & (df["frame"] <= end_frame)]
        if window.empty:
            # None, not "0.00" — same fix as trunk_lean above: "0.00" sits
            # right inside this metric's own "Elite Fixed Gaze Focus" band,
            # so a tracking failure was silently reporting as a perfect
            # result instead of "unavailable."
            return {"deviation_index": None, "tier": "Window Missing", "status": "error", "recalibration_pending": False}

        required = ["NOSE_x", "LEFT_SHOULDER_x", "RIGHT_SHOULDER_x"]
        valid = window.dropna(subset=required)
        if len(valid) < 2:
            return {"deviation_index": None, "tier": "Tracking Limited", "status": "error", "recalibration_pending": False}

        shoulder_width = (valid["LEFT_SHOULDER_x"] - valid["RIGHT_SHOULDER_x"]).abs()
        # Guard against a degenerate/near-zero shoulder width (mistracked
        # landmarks collapsed together, or a genuinely edge-on camera
        # angle) blowing the ratio up to a huge, meaningless number —
        # same "None, not a fabricated number" discipline as everywhere
        # else in this codebase.
        MIN_SHOULDER_WIDTH = 0.02
        valid = valid[shoulder_width >= MIN_SHOULDER_WIDTH]
        if len(valid) < 2:
            return {"deviation_index": None, "tier": "Tracking Drop", "status": "error", "recalibration_pending": False}

        normalized_nose_x = valid["NOSE_x"] / (valid["LEFT_SHOULDER_x"] - valid["RIGHT_SHOULDER_x"]).abs()
        std_dev = round(float(np.std(normalized_nose_x.values)), 4)

        tier = "Elite Fixed Gaze Focus" if std_dev <= 0.08 else "Erratic Lateral Head Drift"
        return {"deviation_index": f"{std_dev}", "tier": tier, "status": "success", "recalibration_pending": True}
    except Exception:
        return {"deviation_index": None, "tier": "Data Deficit", "status": "error", "recalibration_pending": False}