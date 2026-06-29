import pandas as pd
import math
from event_engine import detect_events

# Load data
df = pd.read_csv("output/landmarks.csv")
df = df.dropna()

# Get Ball Release frame from Event Engine
events = detect_events()
BR_FRAME = events["BR"]

row = df.loc[BR_FRAME]

# -------------------------
# SHOULDER LINE ANGLE
# -------------------------

sx1 = row["LEFT_SHOULDER_x"]
sy1 = row["LEFT_SHOULDER_y"]

sx2 = row["RIGHT_SHOULDER_x"]
sy2 = row["RIGHT_SHOULDER_y"]

shoulder_angle = math.degrees(
    math.atan2(sy2 - sy1, sx2 - sx1)
)

# -------------------------
# HIP LINE ANGLE
# -------------------------

hx1 = row["LEFT_HIP_x"]
hy1 = row["LEFT_HIP_y"]

hx2 = row["RIGHT_HIP_x"]
hy2 = row["RIGHT_HIP_y"]

hip_angle = math.degrees(
    math.atan2(hy2 - hy1, hx2 - hx1)
)

# -------------------------
# SEPARATION
# -------------------------

separation = abs(shoulder_angle - hip_angle)

# Normalize value

if separation > 180:
    separation = 360 - separation

print(f"\nBall Release Frame: {BR_FRAME}")
print(f"Shoulder Angle: {shoulder_angle:.2f}°")
print(f"Hip Angle: {hip_angle:.2f}°")

print(f"\nHip-Shoulder Separation: {separation:.2f}°")

print("\nAssessment:")

if separation < 10:
    print("Low separation detected.")
    print("Potential reduction in pace generation.")

elif separation < 30:
    print("Moderate hip-shoulder separation.")
    print("Good rotational mechanics.")

elif separation < 50:
    print("Excellent separation.")
    print("High pace generation potential.")

else:
    print("Very high separation.")
    print("Monitor workload and injury risk.")

print("\nCoaching Focus:")

if separation < 20:
    print("- Medicine ball rotational throws")
    print("- Hip-shoulder dissociation drills")
    print("- Separation loading drills")

else:
    print("- Maintain current rotational mechanics")