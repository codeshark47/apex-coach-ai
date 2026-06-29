import pandas as pd
import math
from event_engine import detect_events

# Load data
df = pd.read_csv("output/landmarks.csv")
df = df.dropna()

# -----------------------------------
# GET EVENTS FROM EVENT ENGINE
# -----------------------------------

events = detect_events()

ffc_frame = events["FFC"]
br_frame = events["BR"]

# -----------------------------------
# KNEE ANGLE AT FFC
# -----------------------------------

ffc_row = df.loc[ffc_frame]

def calculate_angle(ax, ay, bx, by, cx, cy):

    ab = (ax - bx, ay - by)
    cb = (cx - bx, cy - by)

    dot = ab[0] * cb[0] + ab[1] * cb[1]

    mag_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    mag_cb = math.sqrt(cb[0]**2 + cb[1]**2)

    if mag_ab == 0 or mag_cb == 0:
        return 0

    cosine = dot / (mag_ab * mag_cb)
    cosine = max(-1, min(1, cosine))

    return math.degrees(math.acos(cosine))

knee_angle = calculate_angle(
    ffc_row["LEFT_HIP_x"],
    ffc_row["LEFT_HIP_y"],
    ffc_row["LEFT_KNEE_x"],
    ffc_row["LEFT_KNEE_y"],
    ffc_row["LEFT_ANKLE_x"],
    ffc_row["LEFT_ANKLE_y"]
)

# -----------------------------------
# TRUNK LEAN
# -----------------------------------

br_row = df.loc[br_frame]

shoulder_x = (
    br_row["LEFT_SHOULDER_x"] +
    br_row["RIGHT_SHOULDER_x"]
) / 2

shoulder_y = (
    br_row["LEFT_SHOULDER_y"] +
    br_row["RIGHT_SHOULDER_y"]
) / 2

hip_x = (
    br_row["LEFT_HIP_x"] +
    br_row["RIGHT_HIP_x"]
) / 2

hip_y = (
    br_row["LEFT_HIP_y"] +
    br_row["RIGHT_HIP_y"]
) / 2

dx = shoulder_x - hip_x
dy = shoulder_y - hip_y

trunk_angle = abs(
    math.degrees(
        math.atan2(dx, -dy)
    )
)

# -----------------------------------
# REPORT
# -----------------------------------

print("\n")
print("=" * 50)
print("FAST BOWLING ANALYSIS REPORT")
print("=" * 50)

print(f"\nFront Foot Contact Frame: {ffc_frame}")
print(f"Ball Release Frame: {br_frame}")

print(f"\nFront Knee Angle: {knee_angle:.2f}°")
print(f"Trunk Lean Angle: {trunk_angle:.2f}°")

print("\nSTRENGTHS:")

if knee_angle >= 160:
    print("✓ Excellent front leg bracing")

if trunk_angle < 10:
    print("✓ Excellent trunk alignment")

print("\nAREAS TO IMPROVE:")

if knee_angle < 140:
    print("✗ Improve front leg bracing")

if trunk_angle >= 20:
    print("✗ Reduce excessive trunk lean")

print("\nRECOMMENDED DRILLS:")

if knee_angle < 140:
    print("- Front leg bracing drill")

if trunk_angle >= 20:
    print("- Core stability drill")

if knee_angle >= 160 and trunk_angle < 10:
    print("- Maintain current mechanics")

print("\nAnalysis Complete.")
print("=" * 50)