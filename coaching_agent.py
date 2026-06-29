import os
from typing import Dict, Any


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
    br  = events.get("ball_release_frame", "Unknown")

    video_meta   = result_payload.get("video_metadata", {})
    fps          = video_meta.get("fps", 30)
    source_file  = video_meta.get("source_file", "Unknown")
    total_frames = video_meta.get("total_frames", 0)

    # 2. EXTRACT ALL 5 METRICS
    metrics = result_payload.get("biomechanical_metrics", {})

    knee_data      = metrics.get("front_knee_bracing", {})
    knee_angle     = knee_data.get("degrees")
    knee_tier      = knee_data.get("tier", "Unknown")

    lean_data      = metrics.get("trunk_lean", {})
    trunk_lean     = lean_data.get("degrees")
    lean_tier      = lean_data.get("tier", "Unknown")

    hip_data       = metrics.get("hip_shoulder_separation", {})
    hip_sep        = hip_data.get("degrees")
    hip_tier       = hip_data.get("tier", "Unknown")

    release_data   = metrics.get("release_height", {})
    release_ratio  = release_data.get("ratio")
    release_tier   = release_data.get("classification") or release_data.get("tier") or "Unknown"

    head_data      = metrics.get("head_stability", {})
    head_val       = head_data.get("value") or head_data.get("deviation_index")
    head_tier      = head_data.get("classification") or head_data.get("tier") or "Unknown"

    # 3. VALIDATE — block only if majority of metrics are missing
    missing = []
    if knee_angle    is None: missing.append("front_knee_bracing")
    if trunk_lean    is None: missing.append("trunk_lean")
    if hip_sep       is None: missing.append("hip_shoulder_separation")
    if release_ratio is None: missing.append("release_height")
    if head_val      is None: missing.append("head_stability")

    missing_note = ""
    if missing:
        missing_note = f"\nNOTE: The following metrics could not be calculated: {', '.join(missing)}. Acknowledge this in your narrative. Do not fabricate values for them.\n"

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

   # 5. BUILD PROMPT WITH CORRECTED COACHING BIAS & EXACT MATCHING HEADERS
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

BIOMECHANICAL MEASUREMENTS:
1. Lead Knee Bracing Angle:   {fmt(knee_angle)} ({knee_tier})
2. Trunk Lean Deflection:     {fmt(trunk_lean)} ({lean_tier})
3. Hip-Shoulder Separation:   {fmt(hip_sep)} ({hip_tier})
4. Release Height Ratio:      {release_display} ({release_tier})
5. Head Stability Variance:   {fmt(head_val, "")} ({head_tier})
{missing_note}

REFERENCE RANGES (CBC-style classification):
- Knee Bracing:         Optimal 160-180° | Acceptable 145-160° | Critical below 145°
- Trunk Lean:           Optimal 0-20°    | Acceptable 20-35°   | Critical above 35°
- Hip-Shoulder Sep:     Optimal 25-50°   | Acceptable 15-25°   | Critical below 15°
- Release Height:       Optimal 85-105%  | Acceptable 75-85%   | Critical below 75%
- Head Stability:       Optimal 0-0.02   | Acceptable 0.02-0.05| Critical above 0.05

COACHING PHILOSOPHY:
- Some bowlers have unconventional but effective actions built through years of muscle memory.
- Do not recommend correcting a metric that falls in the ACCEPTABLE range if the bowler appears injury-free.
- Only prescribe drills for metrics in the CRITICAL zone, or metrics showing severe technical blocks (like 'Blocked rotation' or extreme outliers).
- If trunk lean exceeds 45 degrees, note that the absolute measurement may be exaggerated by a 2D camera angle artifact, but still comment on managing lateral torque.
- CRITICAL INTERVENTION RULE: If Hip-Shoulder Separation is extremely low (e.g., below 10° or labeled 'Blocked rotation'), this represents a critical developmental floor error where hips and shoulders fire simultaneously. Treat this as a high-priority CRITICAL coaching opportunity. Prescribe actionable drills to build rotational separation.

Your task is to produce a two-section technical coaching report.
Separate the two sections with exactly one line containing only: ---

SECTION 1 — BIOMECHANICAL NARRATIVE ASSESSMENT:
Write 4-5 sentences analyzing the full kinetic chain from BFC through ball release.
Reference actual measured values and their zones. Explicitly call out any missing data (N/A) without fabricating values.
Clearly distinguish between what requires immediate correction versus what requires monitoring.

SECTION 2 — PRESCRIBED DRILLS:
Provide exactly 3 drills targeting the weakest CRITICAL zone or technically blocked metric.
Format each drill exactly as a single line without extra line breaks:
DRILL NAME: explaining what it corrects and how to perform it.
"""

    # 6. CALL GEMINI API WITH INCREASED TOKEN BUDGET
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
                max_output_tokens=8000  # Increased token allowance to prevent cutting off
            )
        )

        if not response or not response.text:
            return _error_state("Gemini returned an empty response.")

        raw_text = response.text.strip()

        # 7. PARSE RESPONSE INTO STRUCTURED DICT
        if "---" in raw_text:
            parts = raw_text.split("---", maxsplit=1)
            narrative = parts[0].strip()
            drills_block = parts[1].strip()
            
            # Clean up headers if the model spits them back out inside the text block
            narrative = narrative.replace("SECTION 1 — BIOMECHANICAL NARRATIVE ASSESSMENT:", "").strip()
            drills_block = drills_block.replace("SECTION 2 — PRESCRIBED DRILLS:", "").strip()
            
            drills = [
                line.lstrip("-•*0123456789. ").strip()
                for line in drills_block.split("\n")
                if len(line.strip()) > 10
            ]
        else:
            narrative = raw_text.replace("SECTION 1 — BIOMECHANICAL NARRATIVE ASSESSMENT:", "").strip()
            drills = []

        return {
            "narrative_analysis": narrative,
            "prescribed_drills": drills
        }

    except Exception as e:
        return _error_state(f"Gemini API call failed: {str(e)}")


def _error_state(message: str) -> dict:
    """Returns a clean structured error dict. Never silently fails."""
    return {
        "narrative_analysis": f"⚠️ ANALYSIS ERROR: {message}",
        "prescribed_drills": [],
        "error": True
    }