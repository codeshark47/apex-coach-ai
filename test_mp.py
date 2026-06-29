import mediapipe as mp

print("MediaPipe location:")
print(mp.__file__)

print("\nChecking for solutions:")
print(hasattr(mp, "solutions"))

print("\nAll attributes:")
print(dir(mp))