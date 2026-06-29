import pandas as pd
import math
from event_engine import detect_events

# Load landmark data
df = pd.read_csv("output/landmarks.csv")
df = df.dropna()

# Get events from Event Engine
events = detect_events()

start = events["FFC"]      # Front Foot Contact
end = events["BR"]         # Ball Release

nose_x = df["NOSE_x"][start:end]
nose_y = df["NOSE_y"][start:end]

movement = []

for i in range(len(nose_x) - 1):

    dx = nose_x.iloc[i + 1] - nose_x.iloc[i]
    dy = nose_y.iloc[i + 1] - nose_y.iloc[i]

    dist = math.sqrt(dx**2 + dy**2)
    movement.append(dist)

avg_move = sum(movement) / len(movement)

print("\nHEAD STABILITY ANALYSIS")
print("--------------------------------")

print(f"Front Foot Contact Frame: {start}")
print(f"Ball Release Frame      : {end}")

print(f"\nAverage Head Movement: {avg_move:.4f}")

if avg_move < 0.01:
    print("\nAssessment:")
    print("Excellent head stability.")

elif avg_move < 0.02:
    print("\nAssessment:")
    print("Good head stability.")

else:
    print("\nAssessment:")
    print("Excessive head movement detected.")

print("\nCoaching Focus:")

if avg_move >= 0.02:
    print("- Improve postural control")
    print("- Balance drills")
    print("- Core stability exercises")

else:
    print("- Maintain current mechanics")