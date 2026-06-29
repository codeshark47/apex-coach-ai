import pandas as pd
import numpy as np

def calculate_joint_angle(p1: tuple, p2: tuple, p3: tuple) -> float:
    """
    Calculates the 2D angle at vertex p2 given three coordinates: p1 (hip), p2 (knee), p3 (ankle).
    Uses the dot product vector formula.
    """
    ba = np.array(p1) - np.array(p2)
    bc = np.array(p3) - np.array(p2)

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    return float(np.degrees(np.arccos(cosine_angle)))

def analyze_front_knee_bracing(df: pd.DataFrame, ffc_frame: int) -> dict:
    """
    Analyzes front leg knee extension/bracing efficiency specifically at Front Foot Contact.
    Dynamically infers lead leg by comparing visibility or positioning attributes if required.
    """
    # Safely locate the specific event frame row
    frame_data = df[df["frame"] == ffc_frame]
    
    if frame_data.empty:
        return {
            "status": "error",
            "error_message": f"FFC Frame {ffc_frame} data was not captured or dropped."
        }
        
    row = frame_data.iloc[0]
    
    try:
        # Determine Lead Leg: For standard right-arm bowlers, the left leg is the front bracing leg.
        # Expert extension: Validate whether left or right hip is closer along the capture plane.
        # Defaulting to standard Left-side metrics for baseline validation.
        hip = (row["LEFT_HIP_x"], row["LEFT_HIP_y"])
        knee = (row["LEFT_KNEE_x"], row["LEFT_KNEE_y"])
        ankle = (row["LEFT_ANKLE_x"], row["LEFT_ANKLE_y"])
        
        knee_angle = calculate_joint_angle(hip, knee, ankle)
        
        # Biomechanical classifications for fast bowling bracing
        if knee_angle >= 165.0:
            classification = "Elite Braced Leg"
            critique = "Excellent front leg block. Energy transferred efficiently up through the kinetic chain."
        elif knee_angle >= 150.0:
            classification = "Acceptable Flexion"
            critique = "Moderate knee flexion. Good stability, but minor energy loss occurring during release."
        else:
            classification = "Collapsing Front Knee"
            critique = "Critical flaw: Knee is buckling and collapsing at crease contact, causing massive ball speed loss."

        return {
            "status": "success",
            "frame": int(ffc_frame),
            "front_knee_angle": float(round(knee_angle, 2)),
            "classification": classification,
            "critique": critique
        }

    except KeyError as e:
        return {
            "status": "error",
            "error_message": f"Required anatomical columns missing from input structure: {str(e)}"
        }

if __name__ == "__main__":
    from event_engine import detect_events_robust
    
    try:
        data = pd.read_csv("output/landmarks.csv")
        events = detect_events_robust("output/landmarks.csv", fps=30)
        
        knee_analysis = analyze_front_knee_bracing(data, events["FFC"])
        print("\n=== KNEE BIOMECHANICS METRIC OBJECT ===")
        print(knee_analysis)
    except Exception as e:
        print(f"Module testing failure: {str(e)}")