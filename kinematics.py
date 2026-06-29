import numpy as np
import pandas as pd

def calculate_knee_bracing(row: pd.Series) -> dict:
    """Computes the 2D angle of the lead knee joint at FFC via Law of Cosines."""
    try:
        h = np.array([float(row["LEFT_HIP_x"]), float(row["LEFT_HIP_y"])])
        k = np.array([float(row["LEFT_KNEE_x"]), float(row["LEFT_KNEE_y"])])
        a = np.array([float(row["LEFT_ANKLE_x"]), float(row["LEFT_ANKLE_y"])])
        
        kh, ka = h - k, a - k
        denom = np.linalg.norm(kh) * np.linalg.norm(ka)
        if denom == 0 or np.isnan(denom):
            return {"degrees": 0.0, "tier": "Tracking Drop", "status": "error"}
            
        cos_theta = np.dot(kh, ka) / denom
        angle = round(float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))), 1)
        
        if angle >= 165.0: tier = "Elite Rigid Extension"
        elif angle >= 145.0: tier = "Moderate Flexion"
        else: tier = "Collapsing Knee Joint"
        return {"degrees": angle, "tier": tier, "status": "success"}
    except Exception:
        return {"degrees": 0.0, "tier": "Data Deficit", "status": "error"}

def calculate_hip_shoulder_separation(row: pd.Series) -> dict:
    """Calculates alignment variance between hip and shoulder transverse planes at foot plant."""
    try:
        s_vec = np.array([float(row["LEFT_SHOULDER_x"] - row["RIGHT_SHOULDER_x"]), float(row["LEFT_SHOULDER_y"] - row["RIGHT_SHOULDER_y"])])
        h_vec = np.array([float(row["LEFT_HIP_x"] - row["RIGHT_HIP_x"]), float(row["LEFT_HIP_y"] - row["RIGHT_HIP_y"])])
        
        denom = np.linalg.norm(s_vec) * np.linalg.norm(h_vec)
        if denom == 0 or np.isnan(denom):
            return {"degrees": 0.0, "tier": "Side-On Profile/Drop", "status": "error"}
            
        cos_theta = np.dot(s_vec, h_vec) / denom
        angle = round(float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))), 1)
        
        if angle >= 28.0: tier = "Optimal Elastic Stretch"
        elif angle >= 15.0: tier = "Moderate Rotational Separation"
        else: tier = "Block Rotation (Low Power Potential)"
        return {"degrees": angle, "tier": tier, "status": "success"}
    except Exception:
        return {"degrees": 0.0, "tier": "Data Deficit", "status": "error"}

def calculate_trunk_lean(row: pd.Series) -> dict:
    """Measures dynamic lateral deviation of the spinal column away from vertical orientation."""
    try:
        mid_hip_x = (float(row["LEFT_HIP_x"]) + float(row["RIGHT_HIP_x"])) / 2
        mid_hip_y = (float(row["LEFT_HIP_y"]) + float(row["RIGHT_HIP_y"])) / 2
        mid_sh_x = (float(row["LEFT_SHOULDER_x"]) + float(row["RIGHT_SHOULDER_x"])) / 2
        mid_sh_y = (float(row["LEFT_SHOULDER_y"]) + float(row["RIGHT_SHOULDER_y"])) / 2
        
        dx = mid_sh_x - mid_hip_x
        dy = mid_hip_y - mid_sh_y  # Invert image coordinate axis to match standard cartesian space
        
        angle = round(float(np.degrees(np.arctan2(np.abs(dx), dy))), 1)
        if np.isnan(angle):
            return {"degrees": 0.0, "tier": "Tracking Drop", "status": "error"}
            
        tier = "Optimal Upright Posture" if angle <= 8.0 else "Excessive Lateral Flexion"
        return {"degrees": angle, "tier": tier, "status": "success"}
    except Exception:
        return {"degrees": 0.0, "tier": "Data Deficit", "status": "error"}

def calculate_release_height(row: pd.Series) -> dict:
    """Measures absolute release height of the bowling wrist relative to standing body baseline."""
    try:
        ankle_y = (float(row["LEFT_ANKLE_y"]) + float(row["RIGHT_ANKLE_y"])) / 2
        nose_y = float(row["NOSE_y"])
        wrist_y = float(row["RIGHT_WRIST_y"]) # Default tracking to right arm action
        
        body_length = ankle_y - nose_y
        if body_length <= 0 or np.isnan(body_length):
            return {"percentage": "0.0%", "tier": "Tracking Drop", "status": "error"}
            
        # Distance from floor level (ankle) up to wrist elevation point
        wrist_height = ankle_y - wrist_y
        ratio = round(float((wrist_height / body_length) * 100), 1)
        
        tier = "High-Release Leverage" if ratio >= 105.0 else "Low-Sling Action"
        return {"percentage": f"{ratio}%", "tier": tier, "status": "success"}
    except Exception:
        return {"percentage": "0.0%", "tier": "Data Deficit", "status": "error"}

def calculate_head_stability(df: pd.DataFrame, start_frame: int, end_frame: int) -> dict:
    """Tracks absolute trajectory path deviation of the head cluster during the delivery stride."""
    try:
        window = df[(df["frame"] >= start_frame) & (df["frame"] <= end_frame)]
        if window.empty:
            return {"deviation_index": "0.00", "tier": "Window Missing", "status": "error"}
            
        nose_x = window["NOSE_x"].dropna().values
        if len(nose_x) < 2:
            return {"deviation_index": "0.00", "tier": "Tracking Limited", "status": "error"}
            
        # Standard deviation calculates absolute displacement variance across frames
        std_dev = round(float(np.std(nose_x)), 4)
        
        tier = "Elite Fixed Gaze Focus" if std_dev <= 0.015 else "Erratic Lateral Head Drift"
        return {"deviation_index": f"{std_dev}", "tier": tier, "status": "success"}
    except Exception:
        return {"deviation_index": "0.00", "tier": "Data Deficit", "status": "error"}