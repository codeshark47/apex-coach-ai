import cv2
import mediapipe as mp
import os

from event_engine import detect_events

# ---------------------------------------
# LOAD EVENTS
# ---------------------------------------

events = detect_events()

ffc = events["FFC"]
br = events["BR"]

# ---------------------------------------
# MEDIAPIPE SETUP
# ---------------------------------------

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_draw = mp.solutions.drawing_utils

# ---------------------------------------
# OPEN VIDEO
# ---------------------------------------

cap = cv2.VideoCapture("input/input_video.mp4")

if not cap.isOpened():
    print("ERROR: Could not open input video.")
    exit()

# ---------------------------------------
# VIDEO PROPERTIES
# ---------------------------------------

fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ---------------------------------------
# CREATE OUTPUT FOLDER
# ---------------------------------------

os.makedirs("output", exist_ok=True)

# ---------------------------------------
# CREATE VIDEO WRITER
# ---------------------------------------

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    "output/annotated_bowling.mp4",
    fourcc,
    fps,
    (width, height)
)

# ---------------------------------------
# PROCESS VIDEO
# ---------------------------------------

frame_num = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    # Draw pose landmarks
    if results.pose_landmarks:

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # Front Foot Contact label
    if frame_num == ffc:

        cv2.putText(
            frame,
            "FRONT FOOT CONTACT",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

    # Ball Release label
    if frame_num == br:

        cv2.putText(
            frame,
            "BALL RELEASE",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # Frame number
    cv2.putText(
        frame,
        f"Frame: {frame_num}",
        (50, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    out.write(frame)

    frame_num += 1

# ---------------------------------------
# CLEANUP
# ---------------------------------------

cap.release()
out.release()

print("Annotated video saved successfully.")
