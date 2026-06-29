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

# Shoulder coordinates
left_x = row["LEFT_SHOULDER_x"]
left_y = row["LEFT_SHOULDER_y"]

right_x = row["RIGHT_SHOULDER_x"]
right_y = row["RIGHT_SHOULDER_y"]

# Shoulder line angle
dx = right_x - left_x
dy = right_y - left_y

angle = abs(math.degrees(math.atan2(dy, dx)))

print(f"\nBall Release Frame: {BR_FRAME}")
print(f"Shoulder Alignment Angle: {angle:.2f} degrees")

print("\nAssessment:")

if angle < 10:
    print("Shoulders are well aligned.")

elif angle < 20:
    print("Minor shoulder tilt detected.")

else:
    print("Significant shoulder tilt detected.")

print("\nCoaching Focus:")

if angle >= 20:
    print("- Improve shoulder stability")
    print("- Strengthen upper back muscles")
    print("- Alignment drills")

else:
    print("- Maintain current shoulder alignment")