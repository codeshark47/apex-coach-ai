import pandas as pd
import math
from event_engine import detect_events

# Load landmark data
df = pd.read_csv("output/landmarks.csv")
df = df.dropna()

# -----------------------------------
# GET EVENTS
# -----------------------------------

events = detect_events()

ffc_frame = events["FFC"]
br_frame = events["BR"]

# -----------------------------------
# FRONT KNEE ANGLE
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
# HIP-SHOULDER SEPARATION
# -----------------------------------

sx1 = br_row["LEFT_SHOULDER_x"]
sy1 = br_row["LEFT_SHOULDER_y"]

sx2 = br_row["RIGHT_SHOULDER_x"]
sy2 = br_row["RIGHT_SHOULDER_y"]

shoulder_angle = math.degrees(
    math.atan2(sy2 - sy1, sx2 - sx1)
)

hx1 = br_row["LEFT_HIP_x"]
hy1 = br_row["LEFT_HIP_y"]

hx2 = br_row["RIGHT_HIP_x"]
hy2 = br_row["RIGHT_HIP_y"]

hip_angle = math.degrees(
    math.atan2(hy2 - hy1, hx2 - hx1)
)

separation = abs(shoulder_angle - hip_angle)

if separation > 180:
    separation = 360 - separation

# -----------------------------------
# SCORING
# -----------------------------------

score = 0

# Front leg score
if knee_angle >= 160:
    score += 3
elif knee_angle >= 140:
    score += 2
else:
    score += 1

# Trunk score
if trunk_angle <= 10:
    score += 3
elif trunk_angle <= 20:
    score += 2
else:
    score += 1

# Separation score
if separation >= 30:
    score += 4
elif separation >= 20:
    score += 3
elif separation >= 10:
    score += 2
else:
    score += 1

# -----------------------------------
# OUTPUT
# -----------------------------------

print(f"\nOverall Bowling Score: {score}/10")

if score >= 8:
    print("Elite bowling mechanics.")

elif score >= 6:
    print("Good bowling mechanics with minor improvements needed.")

else:
    print("Significant biomechanical improvements recommended.")