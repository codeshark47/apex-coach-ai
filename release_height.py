import pandas as pd
from event_engine import detect_events

# Load landmark data
df = pd.read_csv("output/landmarks.csv")
df = df.dropna()

# Get Ball Release frame from Event Engine
events = detect_events()
br_frame = events["BR"]

wrist_y = df.loc[br_frame, "RIGHT_WRIST_y"]

# MediaPipe coordinates:
# smaller y = higher position

release_score = 1 - wrist_y

print("\nRELEASE HEIGHT ANALYSIS")
print("--------------------------------")

print(f"Ball Release Frame: {br_frame}")
print(f"Normalized Release Height: {release_score:.3f}")

print("\nAssessment:")

if release_score > 0.60:
    print("Excellent high release point.")

elif release_score > 0.45:
    print("Moderate release height.")

else:
    print("Low release point detected.")

print("\nCoaching Focus:")

if release_score < 0.45:
    print("- Improve front leg bracing")
    print("- Improve trunk extension")
    print("- High-arm release drills")

else:
    print("- Maintain current release mechanics")