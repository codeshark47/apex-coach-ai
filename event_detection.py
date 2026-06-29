import pandas as pd

df = pd.read_csv("output/landmarks.csv")
df = df.dropna()

# ---------------------------
# BALL RELEASE
# ---------------------------

br_frame = df["RIGHT_WRIST_y"].idxmin()

# ---------------------------
# FRONT FOOT CONTACT
# ---------------------------

ffc_start = max(0, br_frame - 40)
ffc_end = max(0, br_frame - 10)

print("FFC search window:", ffc_start, "to", ffc_end)

ffc_window = df.iloc[ffc_start:ffc_end]

if len(ffc_window) == 0:
    print("FFC window is empty!")
    exit()

ffc_frame = ffc_window["LEFT_ANKLE_y"].idxmax()

# ---------------------------
# BACK FOOT CONTACT
# ---------------------------

bfc_start = max(0, ffc_frame - 40)
bfc_end = max(0, ffc_frame - 10)

print("BFC search window:", bfc_start, "to", bfc_end)

bfc_window = df.iloc[bfc_start:bfc_end]

if len(bfc_window) == 0:
    print("BFC window is empty!")
    bfc_frame = 0
else:
    bfc_frame = bfc_window["RIGHT_ANKLE_y"].idxmax()

# ---------------------------
# RESULTS
# ---------------------------

print("\nEVENT DETECTION")
print("--------------------------------")
print(f"Back Foot Contact : {bfc_frame}")
print(f"Front Foot Contact: {ffc_frame}")
print(f"Ball Release      : {br_frame}")

print("\nSequence Validation:")

if bfc_frame < ffc_frame < br_frame:
    print("PASS ✓")
else:
    print("FAIL ✗")