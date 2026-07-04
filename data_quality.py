"""
data_quality.py

Guards against a real failure mode: when MediaPipe pose tracking degrades on
a clip (motion blur, occlusion, bowler leaving frame, oblique camera angle),
some metrics come back as None/NaN — but OTHERS in the same run can come
back as numbers that are mathematically computed yet not physically
trustworthy (e.g. a trunk lean angle near 90 degrees, driven by the same
corrupted landmarks).

Without a guard, the app would:
  1. Color-code a garbage number as "Critical" (red) as if it were real, and
  2. Feed it to the Gemini narrative, which writes confident coaching
     language about a problem that likely never happened.

This module doesn't try to invalidate individual implausible values (that
would require physiologically-validated bounds we don't have a source for —
exactly the kind of fabricated threshold Claude was told to avoid). Instead
it uses an objective, unfakeable signal: how many of the 5 metrics failed to
compute at all in this same run. If tracking broke down badly enough to lose
several metrics, the ones that "succeeded" numerically are not trustworthy
either, since they came from the same degraded landmark stream.
"""

import metric_ranges as mr

LOW_CONFIDENCE_THRESHOLD = 3  # out of 5 metrics — tune based on real-world data, not guessed science


def assess_quality(metrics: dict) -> dict:
    """
    Returns:
      {
        "confidence": "high" | "low",
        "missing_metrics": [list of metric_ranges keys that came back None/NaN],
        "missing_count": int,
      }
    """
    missing = []
    for key in mr.all_metric_keys():
        value = mr.extract_metric_value(metrics, key)
        if mr.classify(key, value) == "unknown":
            missing.append(key)

    confidence = "low" if len(missing) >= LOW_CONFIDENCE_THRESHOLD else "high"

    return {
        "confidence": confidence,
        "missing_metrics": missing,
        "missing_count": len(missing),
    }
