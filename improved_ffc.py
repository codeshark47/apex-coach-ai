import pandas as pd

# Load landmarks
df = pd.read_csv("output/landmarks.csv")
df = df.dropna()

# Front foot coordinates
x = df["LEFT_ANKLE_x"]
y = df["LEFT_ANKLE_y"]

# Calculate frame-to-frame movement
velocity = x.diff().abs()

# Search in second half of action
start = int(len(df) * 0.30)
end = int(len(df) * 0.80)

window = df.iloc[start:end]

best_frame = None
lowest_score = 999

for i in range(start + 1, end):

    # Foot near ground
    ground_score = 1 - y.iloc[i]

    # Foot almost stationary
    move_score = velocity.iloc[i]

    score = move_score + ground_score

    if score < lowest_score:
        lowest_score = score
        best_frame = i

print("\nImproved Front Foot Contact Frame:")
print(best_frame)

print("\nLeft Ankle Y:")
print(y.iloc[best_frame])