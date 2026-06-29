import subprocess

print("\n========== CRICKET AI ANALYSIS ==========\n")

# ---------------------------------------
# STEP 1
# ---------------------------------------

print("1. Extracting landmarks...")
subprocess.run(["py", "-3.12", "main.py"])

# ---------------------------------------
# STEP 2
# ---------------------------------------

print("\n2. Detecting key frames...")
subprocess.run(["py", "-3.12", "extract_key_frames.py"])

# ---------------------------------------
# STEP 3
# ---------------------------------------

print("\n3. Front Foot Contact analysis...")
subprocess.run(["py", "-3.12", "ffc_analysis.py"])

# ---------------------------------------
# STEP 4
# ---------------------------------------

print("\n4. Trunk Lean analysis...")
subprocess.run(["py", "-3.12", "trunk_lean.py"])

# ---------------------------------------
# STEP 5
# ---------------------------------------

print("\n5. Shoulder Alignment analysis...")
subprocess.run(["py", "-3.12", "shoulder_alignment.py"])

# ---------------------------------------
# STEP 6
# ---------------------------------------

print("\n6. Hip-Shoulder Separation analysis...")
subprocess.run(["py", "-3.12", "hip_shoulder_separation.py"])

# ---------------------------------------
# STEP 7
# ---------------------------------------

print("\n7. Release Height analysis...")
subprocess.run(["py", "-3.12", "release_height.py"])

# ---------------------------------------
# STEP 8
# ---------------------------------------

print("\n8. Overall Scoring...")
subprocess.run(["py", "-3.12", "score.py"])

# ---------------------------------------
# STEP 9
# ---------------------------------------

print("\n9. Generating annotated video...")
subprocess.run(["py", "-3.12", "annotate_video.py"])

# ---------------------------------------
# STEP 10
# ---------------------------------------

print("\n10. Generating PDF report...")
subprocess.run(["py", "-3.12", "pdf_report.py"])

# ---------------------------------------

print("\n===================================")
print("ANALYSIS COMPLETE")
print("Check output folder for reports and annotated video.")
print("===================================\n")