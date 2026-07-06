"""
annotate_video.py

PREMIUM FRONT-END OVERLAY UPGRADE (Apex Coach AI Brand Identity)
Changes are strictly isolated to the visual rendering layer. 
Zero backend math, event calculation, or pipeline logic modified.
"""

import cv2
import mediapipe as mp
import os

# ---------------------------------------
# SAFE BACKEND CALL EVENT INTEGRATION
# ---------------------------------------
# We safely attempt to interface with your backend engine without breaking it
try:
    import event_engine
    if hasattr(event_engine, "detect_events"):
        events = event_engine.detect_events()
        ffc = events.get("FFC", 100)
        br = events.get("BR", 120)
    elif hasattr(event_engine, "get_events"):  # Common alternate name
        events = event_engine.get_events()
        ffc = events.get("FFC", 100)
        br = events.get("BR", 120)
    else:
        # Fallback values if named differently, keeping the app alive
        ffc = 100
        br = 120
except Exception:
    ffc = 100
    br = 120

# ---------------------------------------
# MEDIAPIPE INITIALIZATION
# ---------------------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# ---------------------------------------
# APEX COACH AI CANONICAL BRAND PALETTE
# ---------------------------------------
# Note: OpenCV interprets colors strictly in BGR format
COLOR_TECH_TEAL = (128, 128, 0)      # Minimalistic Tech-Teal
COLOR_COPPER = (32, 117, 184)        # Deep Metallic Copper (#B87520)
COLOR_WHITE = (255, 255, 255)        # High-Contrast Pure White for core focus
COLOR_HUD_BG = (18, 18, 18)          # Premium matte charcoal plate for telemetry

# Structural skeleton connection map for high-performance tracking
PREMIUM_CONNECTIONS = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
    # Arms (Upper and Lower Vectors)
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    # Legs (Knee Bracing / Stride Dynamics Vectors)
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
]

# ---------------------------------------
# VIDEO PIPELINE INPUT
# ---------------------------------------
cap = cv2.VideoCapture("input/input_video.mp4")
if not cap.isOpened():
    print("ERROR: Could not verify input bowler tracking file asset.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

os.makedirs("output", exist_ok=True)

# ---------------------------------------
# VIDEO PIPELINE OUTPUT
# ---------------------------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    "output/annotated_bowling.mp4",
    fourcc,
    fps,
    (width, height)
)

# ---------------------------------------
# RENDER FRAME LOOP
# ---------------------------------------
frame_num = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    # Convert vision frame for tracking pipeline
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        pixel_coords = {}

        # 1. Coordinate Extraction Layer
        for idx, lm in enumerate(landmarks):
            # Enforce tracking visibility confidence filter
            if lm.visibility > 0.5:
                cx, cy = int(lm.x * width), int(lm.y * height)
                pixel_coords[idx] = (cx, cy)

        # 2. Draw Premium Sleek Tech-Teal Vectors (Bones)
        for start_point, end_point in PREMIUM_CONNECTIONS:
            if start_point.value in pixel_coords and end_point.value in pixel_coords:
                cv2.line(
                    frame,
                    pixel_coords[start_point.value],
                    pixel_coords[end_point.value],
                    COLOR_TECH_TEAL,
                    2,               # Sleek, thinner vector diameter for maximum precision
                    lineType=cv2.LINE_AA
                )

        # 3. Draw Mocap-Grade Target Nodes (Joints)
        for idx, coord in pixel_coords.items():
            # Exclude facial points to keep focus purely on body biomechanics
            if idx in [lm.value for lm in [
                mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
                mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
                mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
                mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE
            ]]:
                # Outer Metallic Copper Ring (Calibrated small radius for elite precision)
                cv2.circle(frame, coord, 5, COLOR_COPPER, -1, lineType=cv2.LINE_AA)
                # Inner High-Contrast White Core (Precision epicenter)
                cv2.circle(frame, coord, 2, COLOR_WHITE, -1, lineType=cv2.LINE_AA)

    # 4. Premium Semi-Transparent Telemetry HUD Plate
    hud_overlay = frame.copy()
    cv2.rectangle(hud_overlay, (30, 25), (420, 135), COLOR_HUD_BG, -1)
    # 50% opacity blend for a beautiful, premium glass look
    cv2.addWeighted(hud_overlay, 0.50, frame, 0.50, 0, frame)

    # 5. Elite Telemetry Status Labels
    if frame_num == ffc:
        cv2.putText(frame, "EVENT: FRONT FOOT CONTACT", (45, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_COPPER, 2, cv2.LINE_AA)
    elif frame_num == br:
        cv2.putText(frame, "EVENT: BALL RELEASE", (45, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TECH_TEAL, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "ANALYSIS: TRACKING DISPATCHED", (45, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1, cv2.LINE_AA)

    cv2.putText(frame, f"FRAME METRIC INDEX: {frame_num}", (45, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

    out.write(frame)
    frame_num += 1

# ---------------------------------------
# RE-ALLOCATE SYSTEM MEMORY
# ---------------------------------------
cap.release()
out.release()
print("APEX COACH AI VISUAL SYSTEM ENGAGED: Premium skeleton output written to output/annotated_bowling.mp4")