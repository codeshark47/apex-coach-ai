import pandas as pd

# Load data
df = pd.read_csv("output/landmarks.csv")
df = df.dropna()

# Use RIGHT_WRIST height
# Small y value = hand high
# Large y value = hand low

wrist_y = df["RIGHT_WRIST_y"]

# Highest hand position during delivery
release_frame = wrist_y.idxmin()

print("\nEstimated Ball Release Frame:")
print(release_frame)

print("\nRight Wrist Height:")
print(wrist_y[release_frame])