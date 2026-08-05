import os
import re
from typing import Dict, Any

import metric_ranges as mr
import monitoring

ZONE_LABELS = {
    "green": "OPTIMAL", "amber": "ACCEPTABLE", "red": "CRITICAL", "unknown": "NO DATA",
    # No validated benchmark exists for this bowler_type/metric pair — see
    # metric_ranges.classify()'s "descriptive" tier. Never CRITICAL/OPTIMAL.
    "descriptive": "DESCRIPTIVE (no benchmark yet for this bowling style)",
}


def _strip_section_header(text: str) -> str:
    """
    Removes Gemini's restated section headers regardless of exact
    punctuation/prefix — LLMs don't reliably reproduce an exact requested
    heading string every call, so exact string-replace was brittle (it
    previously let a leftover "PRESCRIBED DRILLS:" line survive into the
    drills list as a fake fourth "drill").
    """
    text = re.sub(r'^\s*SECTION\s*1\s*[-—:]*\s*BIOMECHANICAL NARRATIVE ASSESSMENT:?\s*',
                  '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*SECTION\s*2\s*[-—:]*\s*PRESCRIBED DRILLS:?\s*',
                  '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*(SECTION\s*2\s*[-—:]*\s*)?PRESCRIBED DRILLS:?\s*$',
                  '', text, flags=re.IGNORECASE | re.MULTILINE)
    return text.strip()


def generate_biomechanical_coaching_report(result_payload: Dict[str, Any]) -> dict:
    """
    Production AI Coaching Agent.
    Extracts all 5 biomechanical metrics and sends them to Gemini for analysis.
    Returns a structured dict with narrative_analysis and prescribed_drills.
    Never uses fake defaults. Never silently fails.
    """
    # 1. TIME INDICES AND METADATA
    events = result_payload.get("time_indices", {})
    bfc = events.get("back_foot_contact_frame", "Unknown")
    ffc = events.get("front_foot_contact_frame", "Unknown")
    br = events.get("ball_release_frame", "Unknown")

    video_meta = result_payload.get("video_metadata", {})
    fps = video_meta.get("fps", 30)
    source_file = video_meta.get("source_file", "Unknown")
    total_frames = video_meta.get("total_frames", 0)

    # 2. EXTRACT ALL 5 METRICS
    metrics = result_payload.get("biomechanical_metrics", {})

    # bowler_type: None/"pace" (default) | "finger_spin" | "wrist_spin" — set
    # by the coach's sidebar selection, carried through result_payload the
    # same way bowling_arm_detected already is. Drives which metrics get a
    # real green/amber/red verdict vs. a "descriptive" (no benchmark yet)
    # one — see metric_ranges.SPIN_RANGE_OVERRIDES for what's real so far.
    bowler_type = result_payload.get("bowler_type")

    knee_data = metrics.get("front_knee_bracing", {})
    knee_angle = knee_data.get("degrees")
    knee_descriptor = knee_data.get("tier", "Unknown")

    lean_data = metrics.get("trunk_lean", {})
    trunk_lean = lean_data.get("degrees")
    lean_descriptor = lean_data.get("tier", "Unknown")

    hip_data = metrics.get("hip_shoulder_separation", {})
    hip_sep = hip_data.get("degrees")
    hip_descriptor = hip_data.get("tier", "Unknown")

    release_data = metrics.get("release_height", {})
    release_ratio = release_data.get("ratio")
    release_descriptor = release_data.get("classification") or release_data.get("tier") or "Unknown"
    release_recalibration_pending = bool(release_data.get("recalibration_pending"))

    head_data = metrics.get("head_stability", {})
    head_val = head_data.get("value")
    if head_val is None:
        head_val = head_data.get("deviation_index")
    head_descriptor = head_data.get("classification") or head_data.get("tier") or "Unknown"
    head_recalibration_pending = bool(head_data.get("recalibration_pending"))

    # --- SINGLE SOURCE OF TRUTH ---
    # ZONE below comes from metric_ranges.py — the SAME classifier used by
    # the sidebar UI and the PDF report. Previously this prompt used
    # orchestrator's own separate "tier"/"classification" field for
    # CRITICAL/ACCEPTABLE decisions, which could disagree with what the UI
    # and PDF showed (confirmed: a value the PDF correctly flagged CRITICAL
    # was narrated by Gemini as fine, because orchestrator's tier string for
    # it didn't say "critical"). The descriptor is kept as supplementary
    # color commentary only — Gemini is instructed to make urgency/drill
    # decisions strictly from ZONE.
    knee_zone = ZONE_LABELS[mr.classify("front_knee_bracing", knee_angle, bowler_type)]
    lean_zone = ZONE_LABELS[mr.classify("trunk_lean", trunk_lean, bowler_type)]
    hip_zone = ZONE_LABELS[mr.classify("hip_shoulder_separation", hip_sep, bowler_type)]
    release_zone = ZONE_LABELS[mr.classify("release_height", release_ratio, bowler_type)]
    head_zone = ZONE_LABELS[mr.classify("head_stability", head_val, bowler_type)]

    # 3. VALIDATE — block only if majority of metrics are missing
    missing = []
    if knee_angle is None: missing.append("front_knee_bracing")
    if trunk_lean is None: missing.append("trunk_lean")
    if hip_sep is None: missing.append("hip_shoulder_separation")
    if release_ratio is None: missing.append("release_height")
    if head_val is None: missing.append("head_stability")

    missing_note = ""
    if missing:
        missing_note = (
            f"\nNOTE: The following metrics could not be calculated: {', '.join(missing)}. "
            f"Acknowledge this in your narrative. Do not fabricate values for them.\n"
        )
        if len(missing) >= 4:
            return _error_state(
                f"Too many metrics missing: {', '.join(missing)}. "
                f"Check landmark tracking quality and camera angle."
            )

    # 4. FORMAT HELPER
    def fmt(val, unit="°"):
        if val is None:
            return "No Data (N/A)"
        return f"{round(float(val), 2)}{unit}"

    release_display = fmt(
        round(release_ratio * 100, 1) if release_ratio is not None else None,
        "%"
    )

    # Reference ranges generated FROM metric_ranges.py — not a separate
    # hardcoded copy — so this can never drift out of sync with the UI/PDF
    # again (this used to be a third, independently-typed copy of these
    # numbers, and it had gone stale relative to the other two).
    reference_ranges_block = "\n".join(mr.describe_range(k, bowler_type) for k in mr.all_metric_keys())

    bowler_type_label = {
        "finger_spin": "a finger-spin bowler (off-spin / left-arm orthodox)",
        "wrist_spin": "a wrist-spin bowler (leg-spin / left-arm wrist-spin)",
    }.get(bowler_type, "a fast (pace) bowler")

    # 5. BUILD PROMPT
    prompt = f"""
You are the lead biomechanics analyst at a national cricket academy.

Analyze the following bowling tracking data. This delivery is from {bowler_type_label}.

VIDEO METADATA:
- Source file: {source_file}
- Frame rate: {fps} FPS
- Total frames analyzed: {total_frames}
- Back Foot Contact (BFC): Frame {bfc}
- Front Foot Contact (FFC): Frame {ffc}
- Ball Release (BR): Frame {br}

BIOMECHANICAL MEASUREMENTS (ZONE is the authoritative classification for every
decision below — base all urgency and drill choices on ZONE, not on the
descriptor in parentheses, which is supplementary color commentary only):
1. Lead Knee Bracing Angle: {fmt(knee_angle)} — ZONE: {knee_zone} (descriptor: {knee_descriptor})
2. Trunk Lean Deflection: {fmt(trunk_lean)} — ZONE: {lean_zone} (descriptor: {lean_descriptor})
3. Hip-Shoulder Separation: {fmt(hip_sep)} — ZONE: {hip_zone} (descriptor: {hip_descriptor})
4. Release Height Ratio: {release_display} — ZONE: {release_zone} (descriptor: {release_descriptor}){" [RECALIBRATION PENDING - see rule below]" if release_recalibration_pending else ""}
5. Head Stability Variance: {fmt(head_val, "")} — ZONE: {head_zone} (descriptor: {head_descriptor}){" [RECALIBRATION PENDING - see rule below]" if head_recalibration_pending else ""}
{missing_note}
REFERENCE RANGES (CBC-style classification — authoritative, matches the UI and PDF report exactly):
{reference_ranges_block}

COACHING PHILOSOPHY:
- Some bowlers have unconventional but effective actions built through years of muscle memory.
- Do not recommend correcting a metric whose ZONE is ACCEPTABLE if the bowler appears injury-free.
- Only prescribe drills for metrics whose ZONE is CRITICAL, or metrics showing severe technical blocks (like 'Blocked rotation' or extreme outliers).
- If trunk lean exceeds 45 degrees, note that the absolute measurement may be exaggerated by a 2D camera angle artifact, but still comment on managing lateral torque.
- CRITICAL INTERVENTION RULE: If Hip-Shoulder Separation ZONE is CRITICAL, this represents a critical developmental floor error where hips and shoulders fire simultaneously. Treat this as a high-priority CRITICAL coaching opportunity. Prescribe actionable drills to build rotational separation.
- DESCRIPTIVE ZONE RULE: a ZONE of "DESCRIPTIVE (no benchmark yet for this bowling style)" means there is currently no validated pass/fail range for this metric for this bowler's style — this is common for spin bowlers, since most published research on spin bowling reports what correlates with performance, not a validated target angle. Report the number as neutral, informational context only (e.g. "for reference, X was measured at..."). NEVER call it optimal, acceptable, or critical, and NEVER prescribe a drill based on a DESCRIPTIVE metric alone.
- RECALIBRATION-PENDING RULE: if ANY metric above is marked "[RECALIBRATION PENDING]", its underlying measurement was just corrected to fix a real false-reading bug, but the OPTIMAL/ACCEPTABLE/CRITICAL bands it's compared against were tuned for the OLD measurement and have not been re-validated for the new one yet. You may still report the number and its ZONE as useful, directional information, but explicitly note in the narrative that this specific reading is provisional pending re-validation, and do NOT prescribe a drill based on this metric alone even if its ZONE reads CRITICAL.

Your task is to produce a two-section technical coaching report.
Separate the two sections with exactly one line containing only: ---

SECTION 1 — BIOMECHANICAL NARRATIVE ASSESSMENT:
Write 4-5 sentences analyzing the full kinetic chain from BFC through ball release.
Reference each metric by name and its ZONE (OPTIMAL/ACCEPTABLE/CRITICAL/DESCRIPTIVE) only — do NOT restate the
exact numeric value yourself. The precise figures are already shown in the table directly above this
narrative in the final report; a value you restate from memory risks not matching the table exactly,
which has happened before and undermines trust in the whole report. Explicitly call out any missing data (N/A) without fabricating values.
Clearly distinguish between what requires immediate correction (CRITICAL zone), what requires monitoring (ACCEPTABLE zone), and what has no established benchmark yet (DESCRIPTIVE zone, per the rule above).

SECTION 2 — PRESCRIBED DRILLS:
Provide exactly 3 drills targeting the weakest CRITICAL zone or technically blocked metric. If no metric is CRITICAL (common for a spin bowler whose metrics are mostly DESCRIPTIVE), say so plainly instead of inventing a drill for a DESCRIPTIVE or ACCEPTABLE metric.
Format each drill exactly as a single line without extra line breaks:
DRILL NAME: explaining what it corrects and how to perform it.
"""

    # 6. CALL GEMINI API
    try:
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return _error_state("GEMINI_API_KEY not found in environment. Check your .env file.")

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=8000
            )
        )

        if not response or not response.text:
            return _error_state("Gemini returned an empty response.")

        raw_text = response.text.strip()

        # 7. PARSE RESPONSE INTO STRUCTURED DICT
        if "---" in raw_text:
            parts = raw_text.split("---", maxsplit=1)
            narrative = _strip_section_header(parts[0].strip())
            drills_block = _strip_section_header(parts[1].strip())

            drills = []
            for line in drills_block.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Skip any leftover restated header line, case/punctuation-insensitive
                if re.match(r'^(section\s*2\s*[-—:]*\s*)?prescribed\s+drills:?$',
                            line, re.IGNORECASE):
                    continue
                line = line.lstrip("-•*0123456789. ").strip()
                if len(line) > 10:
                    drills.append(line)
            drills = drills[:3]  # never surface more than the 3 the prompt asked for
        else:
            narrative = _strip_section_header(raw_text)
            drills = []

        return {
            "narrative_analysis": narrative,
            "prescribed_drills": drills
        }

    except Exception as e:
        monitoring.capture(e)
        return _error_state(f"Gemini API call failed: {str(e)}")


def generate_batting_coaching_report(result_payload: Dict[str, Any]) -> dict:
    """
    Batting equivalent of generate_biomechanical_coaching_report — same
    machinery (ZONE from metric_ranges.py as the single source of truth,
    same missing-data gate, same two-section parse), new persona and new
    metric set (head movement, front-foot alignment, weight transfer,
    downswing plane, top-elbow angle) instead of bowling's 5.
    """
    events = result_payload.get("time_indices", {})
    stance = events.get("stance_frame", "Unknown")
    backlift = events.get("backlift_frame", "Unknown")
    contact = events.get("contact_frame", "Unknown")

    video_meta = result_payload.get("video_metadata", {})
    fps = video_meta.get("fps", 30)
    source_file = video_meta.get("source_file", "Unknown")
    total_frames = video_meta.get("total_frames", 0)

    metrics = result_payload.get("biomechanical_metrics", {})
    batting_hand = metrics.get("batting_hand_detected", "Unknown")
    camera_angle = result_payload.get("camera_angle", "uncertain")
    view_caveats = result_payload.get("view_confidence_caveats", [])
    shot_played = result_payload.get("shot_played")
    ball_line = result_payload.get("ball_line")
    falling_over = result_payload.get("falling_over_risk", {})

    head_data = metrics.get("head_movement", {})
    head_val = head_data.get("value")
    head_descriptor = head_data.get("tier", "Unknown")

    foot_data = metrics.get("front_foot_alignment", {})
    foot_val = foot_data.get("deviation_degrees")
    foot_signed = foot_data.get("signed_degrees")
    foot_side = foot_data.get("side")
    foot_descriptor = foot_data.get("tier", "Unknown")

    weight_data = metrics.get("weight_transfer", {})
    weight_val = weight_data.get("percent")
    weight_descriptor = weight_data.get("tier", "Unknown")

    swing_data = metrics.get("downswing_plane", {})
    swing_val = swing_data.get("degrees")
    swing_descriptor = swing_data.get("tier", "Unknown")

    elbow_data = metrics.get("top_elbow_angle", {})
    elbow_val = elbow_data.get("degrees")
    elbow_descriptor = elbow_data.get("tier", "Unknown")

    knee_data = metrics.get("front_knee_flexion", {})
    knee_val = knee_data.get("degrees")
    knee_descriptor = knee_data.get("tier", "Unknown")

    xfactor_data = metrics.get("xfactor_separation", {})
    xfactor_val = xfactor_data.get("degrees")
    xfactor_descriptor = xfactor_data.get("tier", "Unknown")

    head_zone = ZONE_LABELS[mr.classify("batting_head_movement", head_val)]
    foot_zone = ZONE_LABELS[mr.classify("batting_front_foot_alignment", foot_val)]
    weight_zone = ZONE_LABELS[mr.classify("batting_weight_transfer", weight_val)]
    swing_zone = ZONE_LABELS[mr.classify("batting_downswing_plane", swing_val)]
    elbow_zone = ZONE_LABELS[mr.classify("batting_top_elbow_angle", elbow_val)]
    knee_zone = ZONE_LABELS[mr.classify("batting_front_knee_flexion", knee_val)]
    xfactor_zone = ZONE_LABELS[mr.classify("batting_xfactor_separation", xfactor_val)]

    # front_foot_alignment's deviation is intentionally None for shots in
    # batting_kinematics.NOT_APPLICABLE_SHOTS (back-foot/horizontal-bat/
    # unorthodox shots, e.g. a hook or reverse sweep) — that's a real,
    # deliberate "this metric doesn't apply", not a tracking failure, and
    # must not count toward "missing data" or the too-many-missing gate.
    foot_not_applicable = foot_descriptor == "Not Applicable For This Shot"

    missing = []
    if head_val is None: missing.append("head_movement")
    if foot_val is None and not foot_not_applicable: missing.append("front_foot_alignment")
    if weight_val is None: missing.append("weight_transfer")
    if swing_val is None: missing.append("downswing_plane")
    if elbow_val is None: missing.append("top_elbow_angle")
    if knee_val is None: missing.append("front_knee_flexion")
    if xfactor_val is None: missing.append("xfactor_separation")

    missing_note = ""
    if missing:
        missing_note = (
            f"\nNOTE: The following metrics could not be calculated: {', '.join(missing)}. "
            f"Acknowledge this in your narrative. Do not fabricate values for them.\n"
        )
        # Threshold scaled up from the original 5-metric set's ">= 4" (80%
        # missing) to keep roughly the same bar now that there are 7
        # metrics — front_knee_flexion and xfactor_separation are most
        # reliable from opposite camera angles (see
        # batting_orchestrator.py's docstring), so it's expected and fine
        # for ONE of them to occasionally read as weaker than the other,
        # not grounds alone to fail the whole report.
        if len(missing) >= 5:
            return _error_state(
                f"Too many metrics missing: {', '.join(missing)}. "
                f"Check landmark tracking quality and camera angle."
            )

    def fmt(val, unit="°"):
        if val is None:
            return "No Data (N/A)"
        return f"{round(float(val), 2)}{unit}"

    reference_ranges_block = "\n".join(mr.describe_range(k) for k in mr.all_batting_metric_keys())

    camera_angle_note = {
        "side_on": "Filmed side-on.",
        "front_or_rear": (
            "Filmed front-on/rear-on (camera roughly behind the stumps or bowler's end). "
            + (f"Treat {', '.join(view_caveats)} with extra caution — these two are most reliable "
               "from side-on footage and can be less precise from this angle." if view_caveats else "")
        ),
        "uncertain": "Filming angle could not be confidently determined.",
        "unavailable": "Filming angle could not be determined (insufficient landmark data).",
    }.get(camera_angle, "Filming angle unknown.")

    foot_direction_note = ""
    if foot_signed is not None:
        shot_label = shot_played.replace("_", " ").title() if shot_played else "no specific shot selected"
        foot_direction_note = (
            f" Front foot pointed {abs(foot_signed)}° toward the {foot_side.lower()} "
            f"of straight down the pitch (shot the coach identified: {shot_label})."
        )

    falling_over_block = ""
    if falling_over.get("status") == "success":
        if falling_over.get("flagged"):
            falling_over_block = (
                f"\nCOMPOUND TECHNICAL FAULT DETECTED — FALLING OVER / PLAYING AROUND THE FRONT PAD:\n"
                f"{falling_over.get('reason')} (head drift {falling_over.get('head_shift_pct')}% of stance "
                f"width, front-foot cross {falling_over.get('foot_cross_pct')}% of stance width, both toward "
                f"the {ball_line} side on a {ball_line}-stump line delivery). This MUST be called out as a "
                f"critical finding in your narrative and should be the top-priority drill if it's the most "
                f"severe fault present — this is a specific, named coaching fault, not a generic head-position "
                f"or foot-alignment comment.\n"
            )
        else:
            falling_over_block = "\nCompound falling-over/scissor-foot check: not triggered on this delivery.\n"

    prompt = f"""
You are the lead batting technique analyst at a national cricket academy — a batting coach with
decades of experience developing players from junior academy level through to elite performance,
fluent in classical technique (MCC-style coaching principles: balance, head position, straight bat)
as well as modern high-performance batting analysis.

Analyze the following batting tracking data.

VIDEO METADATA:
- Source file: {source_file}
- Frame rate: {fps} FPS
- Total frames analyzed: {total_frames}
- Batting hand detected (leading side): {batting_hand}
- Camera angle: {camera_angle_note}
- Ball line reported by coach: {ball_line or "not specified"}
- Stance: Frame {stance}
- Backlift (top of swing): Frame {backlift}
- Point of Contact: Frame {contact}

BIOMECHANICAL MEASUREMENTS (ZONE is the authoritative classification for every
decision below — base all urgency and drill choices on ZONE, not on the
descriptor in parentheses, which is supplementary color commentary only):
1. Head Movement (Stance to Contact): {fmt(head_val, "")} — ZONE: {head_zone} (descriptor: {head_descriptor})
2. Front Foot Alignment (deviation from shot-relative target): {fmt(foot_val)} — ZONE: {foot_zone} (descriptor: {foot_descriptor}).{foot_direction_note}
3. Weight Transfer Onto Front Foot: {fmt(weight_val, "%")} — ZONE: {weight_zone} (descriptor: {weight_descriptor})
4. Downswing Plane (Straight Bat): {fmt(swing_val)} — ZONE: {swing_zone} (descriptor: {swing_descriptor})
5. Top-Elbow Angle At Contact: {fmt(elbow_val)} — ZONE: {elbow_zone} (descriptor: {elbow_descriptor})
6. Front Knee Flexion At Contact: {fmt(knee_val)} — ZONE: {knee_zone} (descriptor: {knee_descriptor})
7. Hip-Shoulder Separation (X-Factor): {fmt(xfactor_val)} — ZONE: {xfactor_zone} (descriptor: {xfactor_descriptor})
{missing_note}{falling_over_block}
REFERENCE RANGES (authoritative, matches the UI and PDF report exactly):
{reference_ranges_block}

IMPORTANT CONTEXT: front_foot_alignment, downswing_plane, and top_elbow_angle are all derived from
body-pose landmarks only (no bat-tracking sensor exists yet) — downswing_plane specifically uses
the midpoint of both wrists as a proxy for the bat handle's path, not the bat's actual face angle.
Do not overstate precision on these three; you may comment on them but frame borderline (ACCEPTABLE
zone) readings with appropriate caution rather than absolute certainty. Front Knee Flexion is most
reliable from side-on footage; Hip-Shoulder Separation (X-Factor) is most reliable from front-on/
rear-on footage — one of the two reading as N/A or borderline is expected depending on camera angle,
not necessarily a tracking failure.

COACHING PHILOSOPHY:
- Every batter has an individual technique built through years of practice — do not recommend
  changing something that is merely unorthodox if it is not in a CRITICAL zone.
- A CONFIRMED compound fault (the falling-over/scissor-foot check above, if triggered) always takes
  priority over any single metric being CRITICAL — it describes a specific, named technical fault a
  real coach would call out by name, not a number in isolation.
- Only prescribe drills for metrics whose ZONE is CRITICAL, or a technically severe/obvious fault.
- Prioritize the fundamentals in this order when multiple metrics are CRITICAL: the falling-over
  compound fault first (if triggered), then head position (a stable head over the ball is the
  foundation everything else is built on), then weight transfer, then front foot alignment, then
  downswing plane, then front-knee flexion / hip-shoulder separation, then top-elbow control — a fix
  higher in this order often naturally improves the ones below it, so lead with the most foundational
  fault.
- Drills must be REAL, standard batting coaching drills a club or academy coach would recognize
  (e.g. shadow batting in front of a mirror, throwdowns/side-arm feeds at a specific line and length,
  a target cone drill for front-foot direction, a resistance-band or towel drill for downswing path,
  a "top-hand only" shadow shot drill for elbow control, an off-stump guard drill for the falling-over
  fault) — not vague, generic advice like "practice more."

Your task is to produce a two-section technical coaching report.
Separate the two sections with exactly one line containing only: ---

SECTION 1 — BATTING TECHNIQUE NARRATIVE ASSESSMENT:
Write 4-5 sentences analyzing the full movement from stance through contact.
Reference each metric by name and its ZONE (OPTIMAL/ACCEPTABLE/CRITICAL) only — do NOT restate the
exact numeric value yourself. The precise figures are already shown in the table directly above this
narrative in the final report; a value you restate from memory risks not matching the table exactly,
which undermines trust in the whole report. Explicitly call out any missing data (N/A) without
fabricating values. Clearly distinguish between what requires immediate correction (CRITICAL zone)
versus what requires monitoring (ACCEPTABLE zone).

SECTION 2 — PRESCRIBED DRILLS:
Provide exactly 3 specific, real batting drills targeting the weakest CRITICAL zone(s), as an expert
batting coach would prescribe them — name the drill, then explain exactly what it corrects and how to
perform it (reps/sets or a concrete setup where useful).
Format each drill exactly as a single line without extra line breaks:
DRILL NAME: explaining what it corrects and how to perform it.
"""

    try:
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return _error_state("GEMINI_API_KEY not found in environment. Check your .env file.")

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=8000
            )
        )

        if not response or not response.text:
            return _error_state("Gemini returned an empty response.")

        raw_text = response.text.strip()

        if "---" in raw_text:
            parts = raw_text.split("---", maxsplit=1)
            narrative = _strip_section_header(parts[0].strip())
            drills_block = _strip_section_header(parts[1].strip())

            drills = []
            for line in drills_block.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if re.match(r'^(section\s*2\s*[-—:]*\s*)?prescribed\s+drills:?$',
                            line, re.IGNORECASE):
                    continue
                line = line.lstrip("-•*0123456789. ").strip()
                if len(line) > 10:
                    drills.append(line)
            drills = drills[:3]
        else:
            narrative = _strip_section_header(raw_text)
            drills = []

        return {
            "narrative_analysis": narrative,
            "prescribed_drills": drills
        }

    except Exception as e:
        monitoring.capture(e)
        return _error_state(f"Gemini API call failed: {str(e)}")


def _error_state(message: str) -> dict:
    """Returns a clean structured error dict. Never silently fails."""
    return {
        "narrative_analysis": f"⚠️ ANALYSIS ERROR: {message}",
        "prescribed_drills": [],
        "error": True
    }
