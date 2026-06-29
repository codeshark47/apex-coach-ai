import pandas as pd
import numpy as np

def detect_events_robust(csv_path: str, fps: int = 30) -> dict:
    """
    Kinematic event engine for fast-bowling analysis.
    Extracts critical bowling phases based on velocity changes and spatial contracts.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing critical landmark file: {csv_path}")

    clean_df = df.dropna().reset_index(drop=True)
    if clean_df.empty:
        raise ValueError("Landmark data file contains no valid coordinates.")

    frames = clean_df["frame"].astype(int).values
    
    # Left Ankle (FFC Detection)
    l_ankle_x = clean_df["LEFT_ANKLE_x"].values
    l_ankle_y = clean_df["LEFT_ANKLE_y"].values
    
    # Right Wrist (Ball Release Detection)
    r_wrist_x = clean_df["RIGHT_WRIST_x"].values
    r_wrist_y = clean_df["RIGHT_WRIST_y"].values

    # --------------------------------------------------
    # PHASE 1: BALL RELEASE (BR)
    # --------------------------------------------------
    apex_idx = np.argmin(r_wrist_y)
    wrist_vx = np.diff(r_wrist_x)
    
    window_limit = min(len(clean_df) - 1, apex_idx + int(fps * 0.4))
    search_window = wrist_vx[apex_idx:window_limit]
    
    if len(search_window) > 0:
        br_idx = apex_idx + np.argmax(search_window)
    else:
        br_idx = apex_idx

    # --------------------------------------------------
    # PHASE 2: FRONT FOOT CONTACT (FFC)
    # --------------------------------------------------
    ankle_v = np.sqrt(np.diff(l_ankle_x)**2 + np.diff(l_ankle_y)**2)
    
    # FIXED: Look backward a maximum of 0.7 seconds from release to find the actual delivery stride
    lookback_limit = max(int(br_idx - (fps * 0.7)), 0)
    pre_release_v = ankle_v[lookback_limit:br_idx]
    
    if len(pre_release_v) > 0:
        # FFC is the lowest velocity point right at impact within the stride window
        ffc_idx = lookback_limit + np.argmin(pre_release_v)
    else:
        ffc_idx = max(0, br_idx - int(fps * 0.4))

    # --------------------------------------------------
    # PHASE 3: BACK FOOT CONTACT (BFC)
    # --------------------------------------------------
    # FIXED: BFC must happen up to 0.4 seconds logically before FFC
    bfc_lookback = max(0, ffc_idx - int(fps * 0.4))
    r_ankle_y = clean_df["RIGHT_ANKLE_y"].values[bfc_lookback:ffc_idx]
    
    if len(r_ankle_y) > 0:
        bfc_idx = bfc_lookback + np.argmin(r_ankle_y)
    else:
        bfc_idx = max(0, ffc_idx - int(fps * 0.2))

    return {
        "BFC": int(frames[bfc_idx]),
        "FFC": int(frames[ffc_idx]),
        "BR": int(frames[br_idx])
    }

if __name__ == "__main__":
    try:
        events = detect_events_robust("output/landmarks.csv", fps=30)
        print("\n=== PRODUCTION EVENT ENGINE STATE ===")
        print(f"Back Foot Contact Frame  : {events['BFC']}")
        print(f"Front Foot Contact Frame : {events['FFC']}")
        print(f"Ball Release Frame       : {events['BR']}")
    except Exception as e:
        print(f"Engine Fault: {str(e)}")