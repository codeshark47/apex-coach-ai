import pandas as pd

df = pd.read_csv("output/landmarks.csv")

# Remove rows with missing data
df = df.dropna()

# Use left ankle vertical position
# Smaller Y value = foot higher
# Larger Y value = foot lower (touching ground)

ankle_y = df["LEFT_ANKLE_y"]

ffc_frame = ankle_y.idxmax()

print("\nEstimated Front Foot Contact Frame:")
print(ffc_frame)

print("\nAnkle Y Position:")
print(ankle_y[ffc_frame])