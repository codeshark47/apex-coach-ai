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

# Premium Brand Colors (Note: OpenCV expects BGR format)
COLOR_TECH_TEAL = (128, 128, 0)      # #008080 equivalent in BGR
COLOR_COPPER = (32, 117, 184)        # #B87520 Metallic Copper equivalent in BGR
COLOR_WHITE = (255, 255, 255)        # Pure crisp white core
COLOR_DARK_HUD = (20, 20, 20)        # Deep charcoal for text backing plates

# Key structural connections for a premium bowling skeleton
PREMIUM_CONNECTIONS = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
    # Arms
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    # Legs
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
]

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

    # Custom Lab-Grade Rendering Layer
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        pixel_coords = {}

        # Convert relative tracking nodes to clean physical screen coordinates
        for idx, lm in enumerate(landmarks):
            # Only map tracking points inside normal vision fields (ignore subtle face dots)
            if lm.visibility > 0.5:
                cx, cy = int(lm.x * width), int(lm.y * height)
                pixel_coords[idx] = (cx, cy)

        # 1. Draw sleek Tech-Teal structural lines (Bones)
        for start_point, end_point in PREMIUM_CONNECTIONS:
            if start_point.value in pixel_coords and end_point.value in pixel_coords:
                cv2.line(
                    frame,
                    pixel_coords[start_point.value],
                    pixel_coords[end_point.value],
                    COLOR_TECH_TEAL,
                    3,
                    lineType=cv2.LINE_AA
                )

        # 2. Draw dual-ring Mocap Target Nodes (Joints)
        for idx, coord in pixel_coords.items():
            # Focus strictly on main biomechanical joint centers
            if idx in [lm.value for lm in [
                mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
                mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
                mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
                mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE
            ]]:
                # Outer Metallic Copper Ring
                cv2.circle(frame, coord, 7, COLOR_COPPER, -1, lineType=cv2.LINE_AA)
                # Inner High-Contrast White Core
                cv2.circle(frame, coord, 3, COLOR_WHITE, -1, lineType=cv2.LINE_AA)

    # 3. Premium Telemetry Backing Banners & Text Layout
    hud_overlay = frame.copy()
    
    # Draw dark backplate for premium contrast balance
    cv2.rectangle(hud_overlay, (25, 20), (420, 140), COLOR_DARK_HUD, -1)
    cv2.addWeighted(hud_overlay, 0.45, frame, 0.55, 0, frame)

    # Display clean telemetry milestones
    if frame_num == ffc:
        cv2.putText(frame, "CRITICAL: FRONT FOOT CONTACT", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_COPPER, 2, cv2.LINE_AA)
    elif frame_num == br:
        cv2.putText(frame, "CRITICAL: BALL RELEASE", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TECH_TEAL, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "STATUS: APPROACH RUN-UP", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 1, cv2.LINE_AA)

    # Global Frame Counter
    cv2.putText(frame, f"ANALYSIS FRAME: {frame_num}", (40, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1, cv2.LINE_AA)

    out.write(frame)
    frame_num += 1

# ---------------------------------------
# CLEANUP
# ---------------------------------------
cap.release()
out.release()
print("SUCCESS: Visual conversion complete. High-end video saved to output/annotated_bowling.mp4")
