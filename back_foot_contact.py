import pandas as pd

# Load data
df = pd.read_csv("output/landmarks.csv")
df = df.dropna()

# Front Foot Contact
ffc_frame = df["LEFT_ANKLE_y"].idxmax()

# Use RIGHT ankle horizontal movement
ankle_x = df["RIGHT_ANKLE_x"]

# Calculate frame-to-frame movement
velocity = ankle_x.diff().abs()

# Search only before FFC
search_start = max(0, ffc_frame - 40)
search_end = max(0, ffc_frame - 5)

window = velocity.iloc[search_start:search_end]

# Find minimum velocity (foot planted)
bfc_frame = window.idxmin()

print("\nEstimated Back Foot Contact Frame:")
print(bfc_frame)

delivery_stride = ffc_frame - bfc_frame

print(f"\nDelivery Stride Frames: {delivery_stride}")

if delivery_stride < 5:
    print("\nAssessment:")
    print("Very short delivery stride.")

elif delivery_stride <= 20:
    print("\nAssessment:")
    print("Normal delivery stride.")

else:
    print("\nAssessment:")
    print("Long delivery stride.")