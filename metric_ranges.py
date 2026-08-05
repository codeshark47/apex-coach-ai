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
    # Distinct from "unknown" (value missing/NaN) — "descriptive" means the
    # value IS real, but no validated green/amber/red benchmark exists for
    # this bowler_type yet (see SPIN_RANGE_OVERRIDES). A neutral info-blue,
    # not gray, so it doesn't read as "we failed to measure this."
    "descriptive": "#3B82F6",
}

TIER_COLORS_PDF = {
    # softer background fills for reportlab table cells (hex, no alpha)
    "green": "#D9F7E4",
    "amber": "#FFF3D6",
    "red": "#FDE0E0",
    "unknown": "#EDF2F7",
    "descriptive": "#DCEEFB",
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
    # BUG FIX (2026-08-03, found while restyling the batting UI): format_value/
    # describe_range used to multiply EVERY unit="%" metric's raw value by 100,
    # which is only correct for a metric stored as a 0-1 fraction (release_height,
    # e.g. 0.85 -> "85%"). batting_weight_transfer's calculate_weight_transfer
    # already returns a 0-100+ number (e.g. 52.0 meaning 52%) — blindly
    # multiplying that turned it into "5200%". already_percent=True opts a
    # metric out of that extra *100, for metrics already stored in percent units.
    already_percent: bool = False


RANGES = {
    # FIX (2026-08-06, real literature audit): these bounds (160-180/
    # 145-160) had NO documented source anywhere in this project — traced
    # via `git log -S` back to the very first commit ("Add files via
    # upload"), predating every citation discipline built since. A real
    # audit found the actual, heavily-cited primary research (Portus,
    # Mason, Elliott, Pfitzner & Done, 2004, Sports Biomechanics 3(2))
    # classifies front knee action at release into Extended-Knee (>=170
    # degrees) vs Flexed-Knee (<170 degrees) TECHNIQUES — both are real,
    # legitimate elite techniques, not a higher-is-better performance
    # scale. A companion analysis from the same research group even found
    # non-trunk-injured bowlers had a MORE flexed knee at release than
    # injured bowlers — the opposite direction from what this range used
    # to imply with "Elite Rigid Extension" (green) vs "Collapsing Knee
    # Joint" (red). These numbers are now DEAD for classification (see
    # _ALWAYS_DESCRIPTIVE_METRICS below — classify() never reaches them
    # for bowler_type=pace) — kept only so label/unit/display_optimal
    # still resolve for any legacy caller. Do not classify against them.
    "front_knee_bracing": MetricRange(
        label="Lead Knee Bracing",
        unit="°",
        kind="higher_better",
        green=(160.0, 180.0),
        amber=(145.0, 160.0),
        display_optimal="160–180°",
    ),
    # FIX (2026-08-06, same audit): also unsourced from the original
    # commit. Real research (Senington, Lee & Williams, J Sports Sciences
    # — 35 elite fast bowlers, mean 33.0 +/- SD 21.6 degrees) shows this
    # varies enormously by bowling action TYPE (front-on/side-on/mixed),
    # not by skill — a front-on bowler legitimately shows far less
    # separation than a mixed-action bowler by design. No universal
    # "optimal zone" exists independent of action type. Same as above:
    # numbers below are DEAD for classification (_ALWAYS_DESCRIPTIVE_METRICS),
    # kept only for label/unit/display_optimal.
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
    # FIX (2026-08-06): direction REVERSED and bounds re-sourced. This used
    # to score MORE forward lean as worse (kind="lower_better", green=0-20,
    # "Optimal Upright Posture") — backwards from real research. Multiple
    # real studies (Elliott, Foster & Gray, 1986; Portus, Mason, Elliott,
    # Pfitzner & Done, 2004; Worthington, King & Ranson, 2013a) find MORE
    # forward trunk flexion at release correlates with FASTER ball release
    # speed — Worthington et al.'s regression names it one of four
    # technique parameters explaining 74% of ball-speed variance. Felton,
    # Lister, Worthington & King (2018/19, J Sports Sciences — real Vicon
    # data, 20 elite male fast bowlers) measured upper trunk angle at
    # release at 159.5 +/- SD 7.8 degrees on an anatomical-180-degrees-
    # neutral convention, i.e. ~20.5 degrees of forward flexion. Bounds
    # below use mean -/+ 1 SD (~13-28 degrees) as the green band — kept
    # WIDE (not a tight cutoff) because that source measures upper-trunk-
    # relative-to-lower-trunk, while this app's formula measures the whole
    # shoulder-hip line relative to true vertical; the two conventions
    # likely track closely but are not proven identical. No established
    # upper ceiling exists in the literature for "too much" forward lean —
    # same reasoning as release_height's own no-ceiling fix below, so this
    # is "higher_better" like that metric, not "band".
    "trunk_lean": MetricRange(
        label="Trunk Lean",
        unit="°",
        kind="higher_better",
        green=(13.0, 30.0),
        amber=(5.0, 13.0),
        display_optimal="13°+ forward lean at release",
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
        # unjustified symmetric margin around the lower bound. A separate,
        # genuine implausibility ceiling already exists elsewhere
        # (calculate_release_height_ratio_safe rejects anything above 1.30
        # as a likely tracking/geometry error before it ever reaches this
        # classification) — that's the right place for a hard limit, not
        # a "technique fault" label.
        #
        # BOUNDS RE-SOURCED (2026-08-06, real literature audit): the old
        # 85%/75% bounds had no documented source (same "Add files via
        # upload" origin as the other unsourced ranges). Real data: Felton,
        # Lister, Worthington & King (2018/19, J Sports Sciences — real
        # Vicon data, 20 elite male fast bowlers) measured release height
        # at 112.8% +/- SD 4.1% of TRUE STANDING HEIGHT (floor to crown).
        # This app's ratio uses a DIFFERENT baseline (nose-to-ankle
        # segment-sum, not full stature — MediaPipe has no reliable
        # head-top landmark), so their number does not transfer directly;
        # converted using real anthropometric ratios (standing eye/nose
        # height ~93.4% of stature; ankle height ~7.9% of stature, both
        # population averages, not bowling-specific) so nose-to-ankle ~=
        # 85.5% of true stature. Worked through with real units (180cm
        # bowler, their 112.8% = a genuine 203cm release point = 188.8cm
        # above the ankle = 122.7% of an ~153.9cm nose-to-ankle span) —
        # mean converts to ~122.7%, SD to ~4.8% in THIS app's units.
        # Green = mean -/+ 1 SD (~118-128%); amber = mean -2SD to -1SD
        # (~108-118%, a real "notably below elite" concern zone, same
        # role the old 75-85% amber played). This stacks a population
        # anthropometric approximation on top of a real elite-bowler
        # dataset — a real, documented engineering judgment call, not a
        # fabricated number, but flagged here as such.
        kind="higher_better",
        green=(1.18, 1.28),
        amber=(1.08, 1.18),
        display_optimal="118%+",
    ),
    "head_stability": MetricRange(
        label="Head Stability",
        unit="",
        kind="lower_better",
        # BUG FIX (2026-08-05): kinematics.calculate_head_stability used
        # to measure raw std(nose_x) — a camera-distance-dependent number,
        # same class of bug release_height's body-height baseline had —
        # against these same 0.02/0.05 bounds. Now normalized by same-
        # frame shoulder width (matching calculate_weight_transfer's
        # stance-width pattern), so the underlying scale changed
        # entirely. These bounds are a PROVISIONAL first-pass estimate
        # (a rough conversion of the old, itself-never-validated
        # thresholds), not yet checked against real multi-clip data —
        # see calculate_head_stability's recalibration_pending flag,
        # same reasoning as release_height's.
        green=(0.0, 0.08),
        amber=(0.08, 0.15),
        display_optimal="0.00–0.08",
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
        # BUG FIX (2026-08-05): this reuses kinematics.calculate_head_
        # stability wholesale (see that function), which was just fixed
        # to normalize by same-frame shoulder width instead of measuring
        # raw camera-distance-dependent pixel deviation — same class of
        # bug release_height's body-height baseline had. These bounds
        # moved to match that new scale; still a PROVISIONAL first-pass
        # estimate, not yet checked against real multi-clip data.
        green=(0.0, 0.08),
        amber=(0.08, 0.15),
        display_optimal="0.00–0.08",
    ),
    "batting_front_foot_alignment": MetricRange(
        # REDESIGNED (2026-08-03, coach requirement: front-foot alignment
        # should be judged relative to the SHOT BEING PLAYED, e.g. a cover
        # drive's toe should point between mid-off and extra-cover, not
        # dead straight down the pitch) — this metric is no longer a raw
        # angle-from-vertical; it's the DEVIATION (in degrees) between the
        # front foot's actual direction and the target direction for the
        # coach-selected shot (or dead-straight, if no shot is selected).
        # See batting_kinematics.calculate_front_foot_alignment and its
        # SHOT_TARGET_CENTERS_DEGREES table.
        label="Front Foot Alignment (vs. Shot Target)",
        unit="°",
        kind="lower_better",
        green=(0.0, 15.0),
        amber=(15.0, 30.0),
        display_optimal="0–15° off target",
    ),
    "batting_weight_transfer": MetricRange(
        label="Weight Transfer Onto Front Foot",
        unit="%",
        kind="higher_better",
        green=(40.0, 200.0),
        amber=(20.0, 40.0),
        display_optimal="40%+",
        already_percent=True,  # calculate_weight_transfer already returns e.g. 52.0 meaning 52%
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
    "batting_front_knee_flexion": MetricRange(
        # NEW (2026-08-03), added alongside dual camera-angle support:
        # most reliable from side-on footage (see
        # batting_kinematics.calculate_front_knee_flexion) — same
        # Law-of-Cosines formula and reasoning as bowling's proven
        # front_knee_bracing metric above, applied to the batter's lead
        # leg at contact.
        label="Front Knee Flexion At Contact",
        unit="°",
        kind="band",
        green=(100.0, 170.0),
        amber=(85.0, 100.0),
        amber_high=(170.0, 178.0),
        display_optimal="100–170°",
    ),
    "batting_xfactor_separation": MetricRange(
        # NEW (2026-08-03): reuses orchestrator.calculate_hip_shoulder_separation
        # wholesale (same computation bowling already relies on) — most
        # reliable from front-on/rear-on footage, per that function's own
        # documented geometry (front/rear preserves hip-shoulder rotation
        # signal that side-on foreshortens). Same green/amber bands as
        # bowling's hip_shoulder_separation since it's the identical
        # underlying measurement, just applied at batting's contact frame
        # instead of bowling's front-foot-contact frame.
        label="Hip-Shoulder Separation (X-Factor)",
        unit="°",
        kind="band",
        green=(25.0, 50.0),
        amber=(15.0, 25.0),
        amber_high=(50.0, 65.0),
        display_optimal="25–50°",
    ),
}

# --- SPIN-BOWLING SUPPORT (2026-08-04) ---
#
# bowler_type: None (default) | "pace" | "finger_spin" | "wrist_spin".
# None and "pace" are equivalent — both mean "classify against RANGES
# above, exactly as every existing caller already does." Every existing
# call site that doesn't pass bowler_type keeps working identically; this
# is purely additive.
#
# RULE (do not violate): a spin bowler_type NEVER silently falls back to
# a pace-calibrated band for a metric it doesn't have its own entry for.
# Evaluated and rejected doing that during design — a real spinner's
# normal, correct technique could get falsely flagged against numbers
# calibrated for fast bowlers (concretely: the front-knee-bracing
# research below shows wrist-spinners' own natural range sits partly
# below pace's "amber/red" cutoffs). Any (metric, bowler_type) pair not
# listed in SPIN_RANGE_OVERRIDES classifies as "descriptive" instead —
# the real measured value, shown with no invented pass/fail verdict.
#
# Real literature audit (full sourcing in project history — do not add an
# entry here without an equally real, correctly-mapped source):
#   - front_knee_bracing / wrist_spin: Goswami, Srivastava & Rajpoot
#     (2016), "A Biomechanical Analysis of Spin Bowling in Cricket",
#     European Journal of Physical Education and Sport Science 2(6).
#     5 real leg-spin bowlers, 30 deliveries, joint angles at ball release
#     via video analysis. All 5 bowlers were right-handed, so — same
#     lead-side convention kinematics.py already uses (front/lead leg is
#     opposite the bowling arm) — their "Knee Joint LEFT" column is the
#     front/lead knee: mean 162.4 +/- 13.3 deg, observed range 134-187.
#     (Their "Knee Joint RIGHT" column, 131.6 +/- 9.8, is the TRAILING
#     leg — a real, checked mix-up an external AI suggestion made when
#     proposing a range for this metric; the trailing leg is naturally
#     more flexed than the bracing leg, which is exactly the 131.6 vs
#     162.4 split observed.) N=5 at interuniversity level — a real
#     starting reference, not a definitive international-elite standard;
#     used here as the full observed range rather than a tighter mean+/-SD
#     band, given how small the sample is.
#   - Everything else (hip_shoulder_separation, release_height,
#     trunk_lean, head_stability) for EITHER spin type, and everything
#     for finger_spin including front_knee_bracing: no real, correctly-
#     mapped source was found. Real finger-spin research exists (Chin et
#     al. 2009; a 23-bowler Loughborough kinematics study) but reports
#     correlation/variance-explained statistics for pelvis and hip
#     rotation, not a validated target angle band for any metric this
#     app currently computes — that's a real, separate metric worth
#     building later, not a substitute range for an existing one.
SPIN_RANGE_OVERRIDES = {
    ("front_knee_bracing", "wrist_spin"): MetricRange(
        label="Lead Knee Bracing",
        unit="°",
        kind="higher_better",
        green=(134.0, 187.0),
        amber=(119.0, 134.0),
        display_optimal="134–187°",
    ),
}

_SPIN_BOWLER_TYPES = ("finger_spin", "wrist_spin")

# REAL LITERATURE AUDIT (2026-08-06): these two metrics have no universal
# validated pass/fail band for ANY bowler_type, including pace — unlike
# trunk_lean/release_height/head_stability, which DO have a real pace band
# (see RANGES above), just none yet for spin. front_knee_bracing (pace) is
# a real TECHNIQUE CLASSIFICATION (Extended-Knee vs Flexed-Knee, Portus et
# al. 2004), not a higher-is-better scale — both are legitimate elite
# techniques. hip_shoulder_separation varies by bowling action TYPE (front-
# on/side-on/mixed), not skill (Senington, Lee & Williams). Always
# "descriptive" in classify() UNLESS a specific (metric, bowler_type) pair
# has a real override in SPIN_RANGE_OVERRIDES (front_knee_bracing/
# wrist_spin does).
_ALWAYS_DESCRIPTIVE_METRICS = ("front_knee_bracing", "hip_shoulder_separation")


def has_validated_range(metric_key: str, bowler_type: str = None) -> bool:
    """
    True if this (metric_key, bowler_type) pair has a real, validated
    green/amber/red band to classify or draw a chart band against — False
    means classify() returns "descriptive" for any real value. Single
    source of truth for every gauge/chart/table that draws a green band,
    so one can never be drawn for a metric+type combination that doesn't
    have a real one (front_knee_bracing/hip_shoulder_separation for ANY
    bowler_type; any metric with no SPIN_RANGE_OVERRIDES entry for a spin
    bowler_type).
    """
    if metric_key in _ALWAYS_DESCRIPTIVE_METRICS:
        return (metric_key, bowler_type) in SPIN_RANGE_OVERRIDES
    if bowler_type in _SPIN_BOWLER_TYPES:
        return (metric_key, bowler_type) in SPIN_RANGE_OVERRIDES
    return True


def descriptive_note(metric_key: str, value=None, bowler_type: str = None) -> str:
    """
    Real, honest text to show in place of an invented green/amber/red
    verdict whenever has_validated_range()/classify() says "descriptive".
    front_knee_bracing and hip_shoulder_separation get real, sourced,
    metric-specific text (a technique classification, not a generic "no
    data for your style" disclaimer) since they're always-descriptive
    regardless of bowler_type — see _ALWAYS_DESCRIPTIVE_METRICS above.
    Everything else falls back to the generic spin-specific message.
    """
    if metric_key == "front_knee_bracing":
        if value is not None:
            try:
                v = float(value)
                if v == v:  # not NaN
                    technique = "Extended-Knee" if v >= 170.0 else "Flexed-Knee"
                    return (
                        f"{technique} technique at release — both are real, legitimate "
                        f"elite techniques (Portus, Mason, Elliott, Pfitzner & Done, 2004), "
                        f"not a pass/fail scale."
                    )
            except (TypeError, ValueError):
                pass
        return (
            "Front knee action at release is a real technique CLASSIFICATION "
            "(Extended-Knee vs Flexed-Knee — Portus et al. 2004), not a pass/fail scale."
        )
    if metric_key == "hip_shoulder_separation":
        return (
            "Varies substantially by bowling action type (front-on/side-on/mixed), not "
            "by skill — elite bowlers average ~33° with a wide spread (±22°) across "
            "action types (Senington, Lee & Williams). No universal target exists."
        )
    label = RANGES[metric_key].label
    if bowler_type in _SPIN_BOWLER_TYPES:
        return (f"No validated {bowler_type.replace('_', '-')} benchmark yet for {label} — "
                f"reported as a measurement only, not a pass/fail zone.")
    return f"No validated benchmark for {label} yet — reported as a measurement only."


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
    "batting_front_knee_flexion", "batting_xfactor_separation",
]


def classify(metric_key: str, value, bowler_type: str = None) -> str:
    """
    Returns one of "green", "amber", "red", "unknown", "descriptive".
    "unknown" fires when value is None/NaN — never fabricated.
    "descriptive" fires whenever has_validated_range() says this
    (metric_key, bowler_type) pair has no real validated band — either
    because bowler_type is a spin type with no SPIN_RANGE_OVERRIDES entry,
    or because the metric itself has no universal band for ANY bowler_type
    (front_knee_bracing/hip_shoulder_separation — see
    _ALWAYS_DESCRIPTIVE_METRICS). Never a silent fallback to pace's band.
    See the module-level comment above SPIN_RANGE_OVERRIDES and
    _ALWAYS_DESCRIPTIVE_METRICS for the full reasoning and sourcing.
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

    if not has_validated_range(metric_key, bowler_type):
        return "descriptive"
    r = SPIN_RANGE_OVERRIDES.get((metric_key, bowler_type), RANGES[metric_key])
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


def describe_range(metric_key: str, bowler_type: str = None) -> str:
    """
    Human-readable description of a metric's zones, generated from RANGES —
    used by the Gemini coaching prompt so it can never hardcode a second,
    driftable copy of these numbers.

    Whenever has_validated_range() says this (metric_key, bowler_type) pair
    has no real band — a spin bowler_type with no SPIN_RANGE_OVERRIDES
    entry, OR a metric with no universal band at all (front_knee_bracing/
    hip_shoulder_separation, see _ALWAYS_DESCRIPTIVE_METRICS) — returns the
    real, sourced descriptive_note() text instead of describing a band
    that doesn't exist. The prompt must never present a fast-bowling-
    calibrated range as if it applied to a spinner, and must never present
    front_knee_bracing/hip_shoulder_separation as a pass/fail scale (see
    classify()'s "descriptive" tier).
    """
    if not has_validated_range(metric_key, bowler_type):
        label = RANGES[metric_key].label
        return f"- {label}: {descriptive_note(metric_key, None, bowler_type)}"

    r = SPIN_RANGE_OVERRIDES.get((metric_key, bowler_type), RANGES[metric_key])

    def fv(v):
        if r.unit != "%":
            return f"{v}{r.unit}"
        return f"{v:.0f}%" if r.already_percent else f"{v * 100:.0f}%"

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
        v = float(value)
        return f"{v:.0f}%" if r.already_percent else f"{v * 100:.0f}%"
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
        # REDESIGNED (2026-08-03): front_foot_alignment now stores its
        # classifiable number under "deviation_degrees" (degrees off the
        # shot-relative target), not the old "degrees" (raw angle from
        # vertical) — see batting_kinematics.calculate_front_foot_alignment.
        "batting_front_foot_alignment": metrics.get("front_foot_alignment", {}).get("deviation_degrees"),
        "batting_weight_transfer": metrics.get("weight_transfer", {}).get("percent"),
        "batting_downswing_plane": metrics.get("downswing_plane", {}).get("degrees"),
        "batting_top_elbow_angle": metrics.get("top_elbow_angle", {}).get("degrees"),
        "batting_front_knee_flexion": metrics.get("front_knee_flexion", {}).get("degrees"),
        "batting_xfactor_separation": metrics.get("xfactor_separation", {}).get("degrees"),
    }
    return lookup.get(metric_key)
