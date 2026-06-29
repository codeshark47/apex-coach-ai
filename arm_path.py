import pandas as pd
import numpy as np
from event_engine import detect_events

# Load landmark data
df = pd.read_csv("output/landmarks.csv")
df = df.dropna()

# Get Ball Release frame from Event Engine
events = detect_events()
br_frame = events["BR"]

shoulder = np.array([
    df.loc[br_frame, "RIGHT_SHOULDER_x"],
    df.loc[br_frame, "RIGHT_SHOULDER_y"]
])

elbow = np.array([
    df.loc[br_frame, "RIGHT_ELBOW_x"],
    df.loc[br_frame, "RIGHT_ELBOW_y"]
])

wrist = np.array([
    df.loc[br_frame, "RIGHT_WRIST_x"],
    df.loc[br_frame, "RIGHT_WRIST_y"]
])

upper_arm = shoulder - elbow
forearm = wrist - elbow

cosine = np.dot(upper_arm, forearm) / (
    np.linalg.norm(upper_arm) *
    np.linalg.norm(forearm)
)

# Prevent numerical errors
cosine = np.clip(cosine, -1.0, 1.0)

angle = np.degrees(np.arccos(cosine))

print("\nBOWLING ARM ANALYSIS")
print("--------------------------------")

print(f"Ball Release Frame: {br_frame}")
print(f"Elbow Angle at Release: {angle:.2f}°")

if angle > 160:
    print("\nAssessment:")
    print("Excellent arm extension.")

elif angle > 140:
    print("\nAssessment:")
    print("Moderate elbow flexion.")

else:
    print("\nAssessment:")
    print("Excessive elbow flexion detected.")

print("\nCoaching Focus:")

if angle < 140:
    print("- Improve arm extension drills")
    print("- High-arm bowling drills")

else:
    print("- Maintain current bowling action")