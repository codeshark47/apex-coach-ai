import cv2
import os
from event_engine import detect_events

# ---------------------------------------
# GET EVENTS FROM CENTRAL EVENT ENGINE
# ---------------------------------------

events = detect_events()

ffc_frame = events["FFC"]
br_frame = events["BR"]

print(f"FFC Frame: {ffc_frame}")
print(f"Ball Release Frame: {br_frame}")

# ---------------------------------------
# OPEN VIDEO
# ---------------------------------------

# IMPORTANT:
# Always use the latest uploaded video

cap = cv2.VideoCapture("input/input_video.mp4")

if not cap.isOpened():
    print("Could not open video.")
    exit()

# ---------------------------------------
# CREATE OUTPUT FOLDER
# ---------------------------------------

os.makedirs("output/key_frames", exist_ok=True)

# Remove old frames if they exist

ffc_path = "output/key_frames/FFC.jpg"
br_path = "output/key_frames/Ball_Release.jpg"

if os.path.exists(ffc_path):
    os.remove(ffc_path)

if os.path.exists(br_path):
    os.remove(br_path)

# ---------------------------------------
# EXTRACT FRAMES
# ---------------------------------------

frame_num = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    # Save Front Foot Contact frame
    if frame_num == ffc_frame:

        cv2.imwrite(
            "output/key_frames/FFC.jpg",
            frame
        )

        print("Saved FFC.jpg")

    # Save Ball Release frame
    if frame_num == br_frame:

        cv2.imwrite(
            "output/key_frames/Ball_Release.jpg",
            frame
        )

        print("Saved Ball_Release.jpg")

    frame_num += 1

cap.release()

print("Done.")