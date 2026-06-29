# reference_ranges.py
# Biomechanical reference range engine — CBC-style zone classification

BIOMECHANICAL_REFERENCE_RANGES = {
    "front_knee_bracing": {
        "unit": "degrees",
        "optimal_low": 160, "optimal_high": 180,
        "acceptable_low": 145, "acceptable_high": 160,
        "critical_threshold": 145,
        "critical_direction": "below",
        "note": "Below 145° indicates collapsing front leg. 160–180° is functional to elite range."
    },
    "trunk_lean": {
        "unit": "degrees",
        "optimal_low": 0, "optimal_high": 20,
        "acceptable_low": 20, "acceptable_high": 35,
        "critical_threshold": 35,
        "critical_direction": "above",
        "note": "Some bowlers bowl effectively at 25–30°. Only flag above 35° as requiring intervention. Above 45° is likely a camera angle artifact — flag as measurement warning."
    },
    "hip_shoulder_separation": {
        "unit": "degrees",
        "optimal_low": 25, "optimal_high": 50,
        "acceptable_low": 15, "acceptable_high": 25,
        "critical_threshold": 15,
        "critical_direction": "below",
        "note": "Below 15° is blocked rotation — severe power loss. 25–50° is where pace generation lives. If reading is below 5°, flag as possible camera angle error."
    },
    "release_height": {
        "unit": "ratio",
        "optimal_low": 0.85, "optimal_high": 1.05,
        "acceptable_low": 0.75, "acceptable_high": 0.85,
        "critical_threshold": 0.75,
        "critical_direction": "below",
        "note": "Ratio above 1.05 means wrist above head — valid for high-arm bowlers. Ratio above 1.15 is a measurement error — flag it."
    },
    "head_stability": {
        "unit": "std_dev",
        "optimal_low": 0.0, "optimal_high": 0.02,
        "acceptable_low": 0.02, "acceptable_high": 0.05,
        "critical_threshold": 0.05,
        "critical_direction": "above",
        "note": "Standard deviation of nose Y position across delivery stride. Lower is more stable. Above 0.05 introduces accuracy and balance risk."
    }
}

def classify_with_range(metric_name: str, value: float) -> dict:
    """
    Classifies a biomechanical metric value against its reference range.
    Returns zone (optimal/acceptable/critical/warning), label, range display,
    and a coaching note. Never returns a binary pass/fail.
    """
    if value is None:
        return {
            "zone": "unknown",
            "label": "No data",
            "optimal_range": "N/A",
            "acceptable_range": "N/A",
            "measured": None,
            "coaching_note": "Metric could not be calculated. Check camera angle and landmark visibility."
        }

    ranges = BIOMECHANICAL_REFERENCE_RANGES.get(metric_name)
    if not ranges:
        return {
            "zone": "unknown",
            "label": "No reference range defined",
            "measured": value
        }

    opt_low = ranges["optimal_low"]
    opt_high = ranges["optimal_high"]
    acc_low = ranges["acceptable_low"]
    acc_high = ranges["acceptable_high"]
    threshold = ranges["critical_threshold"]
    direction = ranges["critical_direction"]
    unit = ranges["unit"]

    # Measurement sanity checks before classification
    measurement_warning = None
    if metric_name == "trunk_lean" and value > 45:
        measurement_warning = "Value exceeds 45° — possible camera angle artifact. Verify video angle before prescribing corrections."
    if metric_name == "hip_shoulder_separation" and value < 5:
        measurement_warning = "Value below 5° — possible rear-view camera limitation affecting measurement accuracy."
    if metric_name == "release_height" and value > 1.15:
        measurement_warning = "Ratio exceeds 1.15 — measurement error likely. Check landmark tracking quality."

    # Zone classification
    if opt_low <= value <= opt_high:
        zone = "optimal"
        label = "Optimal range"
        action = "Maintain current mechanics."
    elif acc_low <= value <= acc_high or (
        direction == "below" and acc_low <= value < opt_low
    ) or (
        direction == "above" and opt_high < value <= acc_high
    ):
        zone = "acceptable"
        label = "Acceptable — monitor"
        action = "Within functional range. Monitor across multiple deliveries before intervening."
    else:
        zone = "critical"
        label = "Outside safe range"
        action = "Coaching intervention recommended. However, if this is an established action and bowler is injury-free, compare across 3+ deliveries before correcting."

    return {
        "zone": zone,
        "label": label,
        "optimal_range": f"{opt_low}–{opt_high} {unit}",
        "acceptable_range": f"{acc_low}–{acc_high} {unit}",
        "measured": value,
        "unit": unit,
        "coaching_note": ranges["note"],
        "recommended_action": action,
        "measurement_warning": measurement_warning
    }