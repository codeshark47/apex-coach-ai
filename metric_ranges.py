"""
metric_ranges.py

Single canonical source of truth for biomechanical reference ranges.

Phase 1 had this defined TWICE with slightly different values:
  - inline `ranges` dict in streamlit_app.py (UI display)
  - hardcoded "Optimal Range" column in generate_pdf_report() (PDF display)

Both the sidebar UI and the PDF report should import from HERE ONLY.
Do not redefine ranges anywhere else.

Tier logic:
  - "higher_better": green is the top band, red is below amber (e.g. knee bracing)
  - "lower_better":  green is the bottom band, red is above amber (e.g. head stability)
  - "band":          green is a middle band, red is outside amber on either side
                      (used only where the underlying metric can meaningfully be
                      too high AND too low — currently unused, reserved)

Each entry gives explicit numeric boundaries. No metric is scored without an
explicit boundary defined here — if a metric key isn't in RANGES, classify()
raises, on purpose, rather than silently guessing a tier.
"""

from dataclasses import dataclass

TIER_COLORS = {
    "green": "#00C853",
    "amber": "#FFB300",
    "red": "#FF3D3D",
    "unknown": "#94A3B8",
}

TIER_COLORS_PDF = {
    # softer background fills for reportlab table cells (hex, no alpha)
    "green": "#D9F7E4",
    "amber": "#FFF3D6",
    "red": "#FDE0E0",
    "unknown": "#EDF2F7",
}


@dataclass(frozen=True)
class MetricRange:
    label: str
    unit: str
    kind: str          # "higher_better" | "lower_better" | "band"
    green: tuple        # (low, high) inclusive band classified as green
    amber: tuple         # (low, high) — lower amber band for all kinds
    display_optimal: str  # human string for the "optimal range" column
    amber_high: tuple = None  # (low, high) upper amber band — only used by "band" kind


RANGES = {
    "front_knee_bracing": MetricRange(
        label="Lead Knee Bracing",
        unit="°",
        kind="higher_better",
        green=(160.0, 180.0),
        amber=(145.0, 160.0),
        display_optimal="160–180°",
    ),
    "hip_shoulder_separation": MetricRange(
        label="Hip-Shoulder Separation",
        unit="°",
        kind="band",
        green=(25.0, 50.0),
        amber=(15.0, 25.0),
        amber_high=(50.0, 65.0),
        # FIX: was kind="higher_better", which silently classified ANY value
        # above the green ceiling (including physically implausible readings
        # like 84 degrees) as "green" since that logic only ever checks for
        # values being too LOW, never too HIGH. Hip-shoulder separation has a
        # real anatomical ceiling — converted to "band" so values above 65
        # correctly flag as red (critical/likely tracking error) instead of
        # silently passing as optimal.
        display_optimal="25–50°",
    ),
    "trunk_lean": MetricRange(
        label="Trunk Lean",
        unit="°",
        kind="lower_better",
        green=(0.0, 20.0),
        amber=(20.0, 35.0),
        display_optimal="0–20°",
    ),
    "release_height": MetricRange(
        label="Release Height",
        unit="%",
        kind="band",
        green=(0.85, 1.05),
        amber=(0.75, 0.85),          # lower bound: under-extension
        amber_high=(1.05, 1.15),     # upper bound: over-extension/lunge
        # NOTE: the 1.05-1.15 amber / >1.15 red upper thresholds are an
        # engineering default (symmetric margin to the lower bound), not a
        # cited clinical/biomechanics source. Replace with a real reference
        # if one becomes available. Do not describe this as validated.
        display_optimal="85–105%",
    ),
    "head_stability": MetricRange(
        label="Head Stability",
        unit="",
        kind="lower_better",
        green=(0.0, 0.02),
        amber=(0.02, 0.05),
        display_optimal="0.00–0.02",
    ),
}


def classify(metric_key: str, value) -> str:
    """
    Returns one of "green", "amber", "red", "unknown".
    "unknown" only fires when value is None/NaN — never fabricated.
    """
    if metric_key not in RANGES:
        raise KeyError(
            f"No reference range defined for metric '{metric_key}'. "
            f"Add it to RANGES in metric_ranges.py before scoring it."
        )
    if value is None:
        return "unknown"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if v != v:  # NaN check without importing math/numpy
        return "unknown"

    r = RANGES[metric_key]
    g_lo, g_hi = r.green
    a_lo, a_hi = r.amber

    if g_lo <= v <= g_hi:
        return "green"

    if r.kind == "higher_better":
        # amber sits below green; red is below amber
        if a_lo <= v < g_lo:
            return "amber"
        return "red" if v < a_lo else "green"  # values above g_hi still fine
    elif r.kind == "lower_better":
        # amber sits above green; red is above amber
        if g_hi < v <= a_hi:
            return "amber"
        return "red" if v > a_hi else "green"  # values below g_lo still fine
    elif r.kind == "band":
        # green is a middle band; amber/red exist on BOTH sides
        if v < g_lo:
            if a_lo <= v < g_lo:
                return "amber"
            return "red"
        # v > g_hi (since v == green range already returned above)
        if r.amber_high is None:
            raise ValueError(f"'{metric_key}' is kind='band' but has no amber_high defined.")
        ah_lo, ah_hi = r.amber_high
        if ah_lo < v <= ah_hi:
            return "amber"
        return "red"
    else:
        raise ValueError(f"Unsupported range kind '{r.kind}' for '{metric_key}'")


def all_metric_keys():
    return list(RANGES.keys())


def describe_range(metric_key: str) -> str:
    """
    Human-readable description of a metric's zones, generated from RANGES —
    used by the Gemini coaching prompt so it can never hardcode a second,
    driftable copy of these numbers.
    """
    r = RANGES[metric_key]

    def fv(v):
        return f"{v * 100:.0f}%" if r.unit == "%" else f"{v}{r.unit}"

    if r.kind == "higher_better":
        return (f"- {r.label}: Optimal {fv(r.green[0])}-{fv(r.green[1])} | "
                f"Acceptable {fv(r.amber[0])}-{fv(r.amber[1])} | "
                f"Critical below {fv(r.amber[0])}")
    elif r.kind == "lower_better":
        return (f"- {r.label}: Optimal {fv(r.green[0])}-{fv(r.green[1])} | "
                f"Acceptable {fv(r.green[1])}-{fv(r.amber[1])} | "
                f"Critical above {fv(r.amber[1])}")
    elif r.kind == "band":
        ah = r.amber_high
        return (f"- {r.label}: Optimal {fv(r.green[0])}-{fv(r.green[1])} | "
                f"Acceptable {fv(r.amber[0])}-{fv(r.amber[1])} (low) or "
                f"{fv(ah[0])}-{fv(ah[1])} (high) | "
                f"Critical below {fv(r.amber[0])} or above {fv(ah[1])}")
    else:
        raise ValueError(f"Unsupported kind '{r.kind}' for '{metric_key}'")


def format_value(metric_key: str, value) -> str:
    """Human-readable value + unit, matching describe_range's convention."""
    r = RANGES[metric_key]
    if r.unit == "%":
        return f"{float(value) * 100:.0f}%"
    return f"{value}{r.unit}"


def measurement_warning(metric_key: str, value) -> str:
    """
    Flags a value so extreme it's more likely a camera-angle/tracking
    artifact than a real reading. Was previously trapped in a separate,
    unsynced module (reference_ranges.py) used only by the PDF report —
    consolidated here so it's available anywhere without re-duplicating
    the thresholds a third time. Returns None when nothing is flagged.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    if metric_key == "trunk_lean" and v > 45:
        return "Value exceeds 45° — possible camera angle artifact. Verify video angle before prescribing corrections."
    if metric_key == "hip_shoulder_separation" and v < 5:
        return "Value below 5° — possible rear-view camera limitation affecting measurement accuracy."
    if metric_key == "release_height" and v > 1.15:
        return "Ratio exceeds 1.15 — measurement error likely. Check landmark tracking quality."
    return None


def extract_metric_value(metrics: dict, metric_key: str):
    """
    Single shared lookup: maps a metric_ranges key to the numeric value
    orchestrator's metrics dict actually stores it under. Both the PDF
    table and the data-quality check use this — do not duplicate this
    mapping anywhere else.
    """
    head_value = metrics.get("head_stability", {}).get("value")
    if head_value is None:
        head_value = metrics.get("head_stability", {}).get("deviation_index")

    lookup = {
        "front_knee_bracing": metrics.get("front_knee_bracing", {}).get("degrees"),
        "hip_shoulder_separation": metrics.get("hip_shoulder_separation", {}).get("degrees"),
        "trunk_lean": metrics.get("trunk_lean", {}).get("degrees"),
        "release_height": metrics.get("release_height", {}).get("ratio"),
        "head_stability": head_value,
    }
    return lookup.get(metric_key)
