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
        # BUG FIX (was kind="band"): a low release ratio is a real coaching
        # concern (a "low-sling" action releasing well below head height,
        # losing the leverage a higher release gives), but a HIGH ratio is
        # not the symmetric opposite fault — it's not an established
        # biomechanics concern at all. Tall bowlers and bowlers with a
        # pronounced leap/jump in their action (verified directly on a real
        # session: a genuinely excellent, athletic leaping delivery)
        # routinely and legitimately release well above head height —
        # that's a documented trait of some elite fast bowlers, not a
        # fault. "band" was flagging that as "Critical" purely from an
        # unjustified symmetric margin around the lower bound (the
        # previous NOTE here already admitted this wasn't from any cited
        # source). A separate, genuine implausibility ceiling already
        # exists elsewhere (calculate_release_height_ratio_safe rejects
        # anything above 1.30 as a likely tracking/geometry error before
        # it ever reaches this classification) — that's the right place
        # for a hard limit, not a "technique fault" label at 115%.
        kind="higher_better",
        green=(0.85, 1.30),
        amber=(0.75, 0.85),          # genuine concern: under-extension/low-sling
        display_optimal="85%+",
    ),
    "head_stability": MetricRange(
        label="Head Stability",
        unit="",
        kind="lower_better",
        green=(0.0, 0.02),
        amber=(0.02, 0.05),
        display_optimal="0.00–0.02",
    ),

    # --- BATTING METRICS (2026-08-03) ---
    # Namespaced with a "batting_" prefix, deliberately, so they can never
    # collide with a bowling key and can never accidentally show up in the
    # bowling coaching report via all_metric_keys() (see that function's
    # docstring — it explicitly returns ONLY the original 5 bowling keys,
    # unchanged, specifically so adding these below carries zero risk to
    # the existing bowling narrative/PDF). See batting_kinematics.py for
    # the honesty note on how these thresholds were derived (real
    # coaching principles, not yet tuned against real batting footage the
    # way the bowling ranges above were).
    "batting_head_movement": MetricRange(
        label="Head Movement (Stance to Contact)",
        unit="",
        kind="lower_better",
        green=(0.0, 0.02),
        amber=(0.02, 0.05),
        display_optimal="0.00–0.02",
    ),
    "batting_front_foot_alignment": MetricRange(
        label="Front Foot Alignment",
        unit="°",
        kind="lower_better",
        green=(0.0, 20.0),
        amber=(20.0, 35.0),
        display_optimal="0–20°",
    ),
    "batting_weight_transfer": MetricRange(
        label="Weight Transfer Onto Front Foot",
        unit="%",
        kind="higher_better",
        green=(40.0, 200.0),
        amber=(20.0, 40.0),
        display_optimal="40%+",
    ),
    "batting_downswing_plane": MetricRange(
        label="Downswing Plane (Straight Bat)",
        unit="°",
        kind="band",
        green=(10.0, 35.0),
        amber=(5.0, 10.0),
        amber_high=(35.0, 50.0),
        display_optimal="10–35°",
    ),
    "batting_top_elbow_angle": MetricRange(
        label="Top-Elbow Angle At Contact",
        unit="°",
        kind="band",
        green=(100.0, 160.0),
        amber=(85.0, 100.0),
        amber_high=(160.0, 175.0),
        display_optimal="100–160°",
    ),
}

# Explicit, hardcoded lists (not "everything in RANGES") so adding either
# sport's metrics can never silently change what the OTHER sport's
# coaching report/PDF iterates over — see all_metric_keys()/
# all_batting_metric_keys() below.
_BOWLING_METRIC_KEYS = [
    "front_knee_bracing", "hip_shoulder_separation", "trunk_lean",
    "release_height", "head_stability",
]
_BATTING_METRIC_KEYS = [
    "batting_head_movement", "batting_front_foot_alignment",
    "batting_weight_transfer", "batting_downswing_plane", "batting_top_elbow_angle",
]


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
    """Bowling's 5 metric keys ONLY — explicitly hardcoded (not
    list(RANGES.keys())) so adding batting_* entries to RANGES can never
    silently pull batting metrics into the bowling coaching report/PDF.
    See all_batting_metric_keys() for the batting equivalent."""
    return list(_BOWLING_METRIC_KEYS)


def all_batting_metric_keys():
    return list(_BATTING_METRIC_KEYS)


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
        # BUG FIX: this used to render the green band as CLOSED
        # ("Optimal 85%-130%"), implying a ceiling — but classify() for
        # "higher_better" has no ceiling on purpose (see release_height's
        # own comment: a high ratio from a tall/leaping bowler is a real,
        # legitimate trait, not a fault). A value like 145% used to show
        # as "OPTIMAL" right next to a range string that visibly excluded
        # it — self-contradictory in the same report row. Open-ended,
        # matching what display_optimal already correctly says.
        return (f"- {r.label}: Optimal {fv(r.green[0])}+ | "
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
    # NOTE: release_height intentionally has NO warning threshold here.
    # It used to fire at v > 1.15 — stale from before release_height's
    # RANGES entry was fixed to kind="higher_better" with no ceiling (a
    # tall/leaping bowler's high ratio is real, not a fault). That left
    # a value classify() correctly calls "green" showing this warning
    # ("measurement error likely") in the same report row. The genuine
    # implausibility ceiling for this metric lives upstream, in
    # calculate_release_height_ratio_safe (rejects > 1.30 before a value
    # ever reaches here) — this function must not re-impose a second,
    # lower, contradictory one.
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


def extract_batting_metric_value(metrics: dict, metric_key: str):
    """Batting equivalent of extract_metric_value — maps a batting_*
    metric_ranges key to where batting_orchestrator.run_batting_analysis
    actually stores the value in its biomechanical_metrics dict."""
    lookup = {
        "batting_head_movement": metrics.get("head_movement", {}).get("value"),
        "batting_front_foot_alignment": metrics.get("front_foot_alignment", {}).get("degrees"),
        "batting_weight_transfer": metrics.get("weight_transfer", {}).get("percent"),
        "batting_downswing_plane": metrics.get("downswing_plane", {}).get("degrees"),
        "batting_top_elbow_angle": metrics.get("top_elbow_angle", {}).get("degrees"),
    }
    return lookup.get(metric_key)
