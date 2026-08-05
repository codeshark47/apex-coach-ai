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

        # FIX (2026-08-06, real literature audit): this used to imply a
        # value judgment ("Elite Rigid Extension" vs "Collapsing Knee
        # Joint") using unsourced 165/145-degree thresholds — this metric
        # is now always-descriptive for pace (see
        # metric_ranges._ALWAYS_DESCRIPTIVE_METRICS) because real research
        # (Portus, Mason, Elliott, Pfitzner & Done, 2004) shows front knee
        # action at release is a real TECHNIQUE CLASSIFICATION (Extended-
        # Knee >=170deg vs Flexed-Knee <170deg), both legitimate at the
        # elite level — one companion analysis even found non-injured
        # bowlers had a MORE flexed knee than injured ones, the opposite
        # of what "Collapsing" implied. Also fixes a real, separate
        # inconsistency: this raw tier text didn't match the wrist-spin
        # override's own real 134/119-degree thresholds (metric_ranges.
        # SPIN_RANGE_OVERRIDES), so a wrist-spin bowler could see ZONE:
        # green next to a contradictory "Collapsing Knee Joint" descriptor.
        # The 170-degree split matches descriptive_note()'s real threshold
        # for both bowler types now — this raw field is a fallback label
        # only (metric_ranges.classify() is authoritative), but it should
        # never contradict the real data.
        tier = "Extended-Knee Technique" if angle >= 170.0 else "Flexed-Knee Technique"
        return {"degrees": angle, "tier": tier, "status": "success"}
    except Exception:
        return {"degrees": None, "tier": "Data Deficit", "status": "error"}

def calculate_trunk_lean(row: pd.Series) -> dict:
    """
    Measures forward (sagittal-plane) deviation of the shoulder-hip line
    from vertical, at ball release — NOT lateral flexion (side-to-side
    bend), a different anatomical plane covered by separate real research
    (spinal lateral flexion / spondylolysis risk, e.g. Senington, Lee &
    Williams) that this 2D side-on formula was never measuring or citing.

    DIRECTION FIX (2026-08-06, real literature audit): this used to score
    MORE forward lean as WORSE ("Optimal Upright Posture" <= 8 degrees,
    "Excessive Lateral Flexion" above it — the "Lateral" in that old label
    was itself wrong, see above). Real research (Elliott, Foster & Gray,
    1986; Portus, Mason, Elliott, Pfitzner & Done, 2004; Worthington, King
    & Ranson, 2013a) consistently finds MORE forward trunk flexion at
    release correlates with FASTER ball release speed, not less — see
    metric_ranges.RANGES["trunk_lean"]'s comment for the real sourced
    numbers this tier text now matches (mean ~20.5 degrees in elite male
    fast bowlers, Felton, Lister, Worthington & King, 2018/19).
    """
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

        # Matches metric_ranges.RANGES["trunk_lean"]'s real green floor
        # (13 degrees, mean-1SD of Felton et al.'s elite sample) — this raw
        # tier string is a fallback label only (metric_ranges.classify()
        # is the authoritative source of truth for the report/PDF/UI), but
        # it should never contradict that real data the way the old
        # "Optimal <= 8 degrees" cutoff did.
        tier = "Effective Forward Drive" if angle >= 13.0 else "Insufficient Forward Lean"
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