import cv2
import pandas as pd
import numpy as np
import os
import imageio

def draw_biomechanical_overlays(video_path: str, csv_path: str, time_indices: dict, output_mp4_path: str = "output/verified_delivery.mp4") -> str:
    if not os.path.exists(video_path) or not os.path.exists(csv_path):
        return ""

    df = pd.read_csv(csv_path)
    cap = cv2.VideoCapture(video_path)
    
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    os.makedirs(os.path.dirname(output_mp4_path), exist_ok=True)
    
    frame_num = 0
    frames_list = []

    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame_data = df[df["frame"] == frame_num]
        
        if not frame_data.empty:
            row = frame_data.iloc[0]
            
            # Point-by-point verification wrapper to exclude NaN value errors
            def get_joint_pixels(col_x, col_y):
                try:
                    val_x = row[col_x]
                    val_y = row[col_y]
                    if pd.isna(val_x) or pd.isna(val_y):
                        return None
                    return (int(val_x * width), int(val_y * height))
                except:
                    return None

            # Get joints safely
            l_sh = get_joint_pixels("LEFT_SHOULDER_x", "LEFT_SHOULDER_y")
            r_sh = get_joint_pixels("RIGHT_SHOULDER_x", "RIGHT_SHOULDER_y")
            l_hip = get_joint_pixels("LEFT_HIP_x", "LEFT_HIP_y")
            r_hip = get_joint_pixels("RIGHT_HIP_x", "RIGHT_HIP_y")

            # 1. DRAW SPINE VECTOR (Cyan)
            if l_sh and r_sh and l_hip and r_hip:
                sh_mid = (int((l_sh[0] + r_sh[0]) / 2), int((l_sh[1] + r_sh[1]) / 2))
                hip_mid = (int((l_hip[0] + r_hip[0]) / 2), int((l_hip[1] + r_hip[1]) / 2))
                cv2.line(frame, hip_mid, sh_mid, (255, 255, 0), 4)

                # 2. DRAW ARMS (Blue)
                l_wrist = get_joint_pixels("LEFT_WRIST_x", "LEFT_WRIST_y")
                r_wrist = get_joint_pixels("RIGHT_WRIST_x", "RIGHT_WRIST_y")
                if l_wrist: 
                    cv2.line(frame, sh_mid, l_wrist, (255, 0, 0), 3)
                if r_wrist: 
                    cv2.line(frame, sh_mid, r_wrist, (255, 0, 0), 3)

                # 3. DRAW LEGS (Green)
                l_knee = get_joint_pixels("LEFT_KNEE_x", "LEFT_KNEE_y")
                l_ankle = get_joint_pixels("LEFT_ANKLE_x", "LEFT_ANKLE_y")
                if l_knee:
                    cv2.line(frame, hip_mid, l_knee, (0, 255, 0), 3)
                    if l_ankle: 
                        cv2.line(frame, l_knee, l_ankle, (0, 255, 0), 3)

                r_knee = get_joint_pixels("RIGHT_KNEE_x", "RIGHT_KNEE_y")
                r_ankle = get_joint_pixels("RIGHT_ANKLE_x", "RIGHT_ANKLE_y")
                if r_knee:
                    cv2.line(frame, hip_mid, r_knee, (0, 255, 0), 3)
                    if r_ankle: 
                        cv2.line(frame, r_knee, r_ankle, (0, 255, 0), 3)

        # Draw timeline overlays at explicit event locations
        if isinstance(time_indices, dict) and frame_num in time_indices.values():
            cv2.rectangle(frame, (10, 10), (350, 50), (0, 0, 0), -1)
            cv2.putText(frame, "CRITICAL KINEMATIC EVENT", (20, 38), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_num += 1

    cap.release()
    
    if frames_list:
        imageio.mimwrite(output_mp4_path, frames_list, fps=fps, codec='libx264', pixelformat='yuv420p', quality=6)
        
    return output_mp4_path