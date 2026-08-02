import os
import re
from typing import Dict, Any

import metric_ranges as mr
import monitoring

ZONE_LABELS = {"green": "OPTIMAL", "amber": "ACCEPTABLE", "red": "CRITICAL", "unknown": "NO DATA"}


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

    head_data = metrics.get("head_stability", {})
    head_val = head_data.get("value")
    if head_val is None:
        head_val = head_data.get("deviation_index")
    head_descriptor = head_data.get("classification") or head_data.get("tier") or "Unknown"

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
    knee_zone = ZONE_LABELS[mr.classify("front_knee_bracing", knee_angle)]
    lean_zone = ZONE_LABELS[mr.classify("trunk_lean", trunk_lean)]
    hip_zone = ZONE_LABELS[mr.classify("hip_shoulder_separation", hip_sep)]
    release_zone = ZONE_LABELS[mr.classify("release_height", release_ratio)]
    head_zone = ZONE_LABELS[mr.classify("head_stability", head_val)]

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
    reference_ranges_block = "\n".join(mr.describe_range(k) for k in mr.all_metric_keys())

    # 5. BUILD PROMPT
    prompt = f"""
You are the lead biomechanics analyst at a national cricket fast-bowling academy.

Analyze the following fast-bowling tracking data.

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
4. Release Height Ratio: {release_display} — ZONE: {release_zone} (descriptor: {release_descriptor})
5. Head Stability Variance: {fmt(head_val, "")} — ZONE: {head_zone} (descriptor: {head_descriptor})
{missing_note}
REFERENCE RANGES (CBC-style classification — authoritative, matches the UI and PDF report exactly):
{reference_ranges_block}

COACHING PHILOSOPHY:
- Some bowlers have unconventional but effective actions built through years of muscle memory.
- Do not recommend correcting a metric whose ZONE is ACCEPTABLE if the bowler appears injury-free.
- Only prescribe drills for metrics whose ZONE is CRITICAL, or metrics showing severe technical blocks (like 'Blocked rotation' or extreme outliers).
- If trunk lean exceeds 45 degrees, note that the absolute measurement may be exaggerated by a 2D camera angle artifact, but still comment on managing lateral torque.
- CRITICAL INTERVENTION RULE: If Hip-Shoulder Separation ZONE is CRITICAL, this represents a critical developmental floor error where hips and shoulders fire simultaneously. Treat this as a high-priority CRITICAL coaching opportunity. Prescribe actionable drills to build rotational separation.

Your task is to produce a two-section technical coaching report.
Separate the two sections with exactly one line containing only: ---

SECTION 1 — BIOMECHANICAL NARRATIVE ASSESSMENT:
Write 4-5 sentences analyzing the full kinetic chain from BFC through ball release.
Reference each metric by name and its ZONE (OPTIMAL/ACCEPTABLE/CRITICAL) only — do NOT restate the
exact numeric value yourself. The precise figures are already shown in the table directly above this
narrative in the final report; a value you restate from memory risks not matching the table exactly,
which has happened before and undermines trust in the whole report. Explicitly call out any missing data (N/A) without fabricating values.
Clearly distinguish between what requires immediate correction (CRITICAL zone) versus what requires monitoring (ACCEPTABLE zone).

SECTION 2 — PRESCRIBED DRILLS:
Provide exactly 3 drills targeting the weakest CRITICAL zone or technically blocked metric.
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

    head_data = metrics.get("head_movement", {})
    head_val = head_data.get("value")
    head_descriptor = head_data.get("tier", "Unknown")

    foot_data = metrics.get("front_foot_alignment", {})
    foot_val = foot_data.get("degrees")
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

    head_zone = ZONE_LABELS[mr.classify("batting_head_movement", head_val)]
    foot_zone = ZONE_LABELS[mr.classify("batting_front_foot_alignment", foot_val)]
    weight_zone = ZONE_LABELS[mr.classify("batting_weight_transfer", weight_val)]
    swing_zone = ZONE_LABELS[mr.classify("batting_downswing_plane", swing_val)]
    elbow_zone = ZONE_LABELS[mr.classify("batting_top_elbow_angle", elbow_val)]

    missing = []
    if head_val is None: missing.append("head_movement")
    if foot_val is None: missing.append("front_foot_alignment")
    if weight_val is None: missing.append("weight_transfer")
    if swing_val is None: missing.append("downswing_plane")
    if elbow_val is None: missing.append("top_elbow_angle")

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

    def fmt(val, unit="°"):
        if val is None:
            return "No Data (N/A)"
        return f"{round(float(val), 2)}{unit}"

    reference_ranges_block = "\n".join(mr.describe_range(k) for k in mr.all_batting_metric_keys())

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
- Stance: Frame {stance}
- Backlift (top of swing): Frame {backlift}
- Point of Contact: Frame {contact}

BIOMECHANICAL MEASUREMENTS (ZONE is the authoritative classification for every
decision below — base all urgency and drill choices on ZONE, not on the
descriptor in parentheses, which is supplementary color commentary only):
1. Head Movement (Stance to Contact): {fmt(head_val, "")} — ZONE: {head_zone} (descriptor: {head_descriptor})
2. Front Foot Alignment: {fmt(foot_val)} — ZONE: {foot_zone} (descriptor: {foot_descriptor})
3. Weight Transfer Onto Front Foot: {fmt(weight_val, "%")} — ZONE: {weight_zone} (descriptor: {weight_descriptor})
4. Downswing Plane (Straight Bat): {fmt(swing_val)} — ZONE: {swing_zone} (descriptor: {swing_descriptor})
5. Top-Elbow Angle At Contact: {fmt(elbow_val)} — ZONE: {elbow_zone} (descriptor: {elbow_descriptor})
{missing_note}
REFERENCE RANGES (authoritative, matches the UI and PDF report exactly):
{reference_ranges_block}

IMPORTANT CONTEXT: front_foot_alignment, downswing_plane, and top_elbow_angle are all derived from
body-pose landmarks only (no bat-tracking sensor exists yet) — downswing_plane specifically uses
the midpoint of both wrists as a proxy for the bat handle's path, not the bat's actual face angle.
Do not overstate precision on these three; you may comment on them but frame borderline (ACCEPTABLE
zone) readings with appropriate caution rather than absolute certainty.

COACHING PHILOSOPHY:
- Every batter has an individual technique built through years of practice — do not recommend
  changing something that is merely unorthodox if it is not in a CRITICAL zone.
- Only prescribe drills for metrics whose ZONE is CRITICAL, or a technically severe/obvious fault.
- Prioritize the fundamentals in this order when multiple metrics are CRITICAL: head position first
  (a stable head over the ball is the foundation everything else is built on), then weight transfer,
  then front foot alignment, then downswing plane, then top-elbow control — a fix higher in this
  order often naturally improves the ones below it, so lead with the most foundational fault.
- Drills must be REAL, standard batting coaching drills a club or academy coach would recognize
  (e.g. shadow batting in front of a mirror, throwdowns/side-arm feeds at a specific line and length,
  a target cone drill for front-foot direction, a resistance-band or towel drill for downswing path,
  a "top-hand only" shadow shot drill for elbow control) — not vague, generic advice like "practice more."

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
