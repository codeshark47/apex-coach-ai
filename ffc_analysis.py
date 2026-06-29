import pandas as pd
import math
from event_engine import detect_events

# Load landmark data
df = pd.read_csv("output/landmarks.csv")
df = df.dropna()

# Get Front Foot Contact from Event Engine
events = detect_events()
ffc_frame = events["FFC"]

print(f"\nEstimated Front Foot Contact Frame: {ffc_frame}")

# Get data for FFC frame
row = df.loc[ffc_frame]

def calculate_angle(ax, ay, bx, by, cx, cy):

    ab = (ax - bx, ay - by)
    cb = (cx - bx, cy - by)

    dot = ab[0] * cb[0] + ab[1] * cb[1]

    mag_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    mag_cb = math.sqrt(cb[0]**2 + cb[1]**2)

    if mag_ab == 0 or mag_cb == 0:
        return None

    cosine = dot / (mag_ab * mag_cb)
    cosine = max(-1, min(1, cosine))

    angle = math.degrees(math.acos(cosine))

    return angle


# Front leg angle
angle = calculate_angle(
    row["LEFT_HIP_x"],
    row["LEFT_HIP_y"],

    row["LEFT_KNEE_x"],
    row["LEFT_KNEE_y"],

    row["LEFT_ANKLE_x"],
    row["LEFT_ANKLE_y"]
)

print(f"\nFront Knee Angle at FFC: {angle:.2f} degrees")

# Coaching feedback

if angle >= 160:
    print("\nAssessment:")
    print("Excellent front leg bracing.")

elif angle >= 140:
    print("\nAssessment:")
    print("Good front leg stability.")

elif angle >= 120:
    print("\nAssessment:")
    print("Moderate front knee collapse detected.")

else:
    print("\nAssessment:")
    print("Significant front knee collapse detected.")

print("\nRecommended Coaching Focus:")

if angle < 140:
    print("- Front leg bracing drills")
    print("- Medicine ball block drill")
    print("- Bound and hold drill")

else:
    print("- Maintain current front leg mechanics")