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

# Batting's own threshold (2026-08-15): scaled proportionally to keep
# roughly the same ~60% bar bowling uses, now that there are 7 metrics
# instead of 5 (3/5 = 60% -> 4/7 = 57%, close enough not to need its own
# tuning pass) — NOT the same number as coaching_agent.py's separate
# >=5-of-7 gate, which decides whether to skip the AI narrative entirely;
# this one decides whether to show the coach a caution banner alongside
# metrics that DID compute. Different question, deliberately different
# (lower) bar — see this module's docstring for why even "successful"
# numbers aren't trustworthy once tracking has degraded this much.
BATTING_LOW_CONFIDENCE_THRESHOLD = 4  # out of 7 metrics


def assess_quality(metrics: dict, metric_keys=None, value_extractor=None, threshold: int = None) -> dict:
    """
    metric_keys / value_extractor: override the bowling defaults (
    mr.all_metric_keys / mr.extract_metric_value) to run this same check
    for batting — pass mr.all_batting_metric_keys() and
    mr.extract_batting_metric_value, plus threshold=
    BATTING_LOW_CONFIDENCE_THRESHOLD. Added 2026-08-15: batting shipped
    2026-08-03 with no equivalent of this guard at all — its 7 metric
    cards would render normally even when tracking had degraded badly
    enough that several of them silently came back unavailable, exactly
    the failure mode this module exists to catch for bowling.

    Returns:
      {
        "confidence": "high" | "low",
        "missing_metrics": [list of metric_ranges keys that came back None/NaN],
        "missing_count": int,
      }
    """
    keys = metric_keys if metric_keys is not None else mr.all_metric_keys()
    extractor = value_extractor if value_extractor is not None else mr.extract_metric_value
    thresh = threshold if threshold is not None else LOW_CONFIDENCE_THRESHOLD

    missing = []
    for key in keys:
        value = extractor(metrics, key)
        if mr.classify(key, value) == "unknown":
            missing.append(key)

    confidence = "low" if len(missing) >= thresh else "high"

    return {
        "confidence": confidence,
        "missing_metrics": missing,
        "missing_count": len(missing),
    }
