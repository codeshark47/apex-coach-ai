"""
ball_tracking/visualize_trajectory.py

Standalone verification tool for track_ball_from_seed.py — draws the
tracked trail (line + points) onto the actual source video and writes an
annotated copy, so a real run can be SEEN, not just summarized as
numbers. Deliberately NOT wired into streamlit_app.py (see the project
memory on ball-tracking strategy — the live app stays isolated from this
work until tracking is reliable across more than one benchmark clip, not
just this one).

Only draws the trail across the frames actually spanned by real tracked
points — never extends a line past the last real detection, since that
would visually imply tracking continued when it didn't.

Usage:
    python ball_tracking/visualize_trajectory.py <video_path> <seed_frame> <seed_x> <seed_y> <model_path> [max_frames_forward]
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO

from ball_tracking.track_ball_from_seed import track_ball_from_seed


def render_trajectory_video(video_path: str, seed_frame: int, seed_xy: tuple,
                             model_path: str, output_path: str,
                             max_frames_forward: int = 60) -> dict:
    """
    Runs track_ball_from_seed and writes an annotated copy of video_path
    to output_path with the real tracked trail drawn on it (yellow
    connecting line, red dots per confirmed point, a green ring on the
    current frame's point). Returns the raw track_ball_from_seed result
    dict so the caller can also inspect/print it.
    """
    model = YOLO(model_path)
    result = track_ball_from_seed(video_path, seed_frame, seed_xy, model,
                                   max_frames_forward=max_frames_forward)
    if result["status"] != "success":
        return result

    points = result["points"]
    pts_by_frame = {p[0]: (p[1], p[2]) for p in points}
    first_f, last_f = points[0][0], points[-1][0]

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    trail = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in pts_by_frame:
            trail.append(pts_by_frame[idx])
        # Only annotate within (and briefly after, for a beat of context)
        # the real tracked window — an unannotated frame honestly shows
        # "we have no data here" instead of implying otherwise.
        if first_f <= idx <= last_f + 5:
            for i in range(1, len(trail)):
                cv2.line(frame, (int(trail[i - 1][0]), int(trail[i - 1][1])),
                          (int(trail[i][0]), int(trail[i][1])), (0, 255, 255), 2)
            for px, py in trail:
                cv2.circle(frame, (int(px), int(py)), 4, (0, 0, 255), -1)
            if idx in pts_by_frame:
                cx, cy = pts_by_frame[idx]
                cv2.circle(frame, (int(cx), int(cy)), 12, (0, 255, 0), 2)
        writer.write(frame)
        idx += 1
    writer.release()
    cap.release()

    return result


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print(__doc__)
        sys.exit(1)
    video_path = sys.argv[1]
    seed_frame = int(sys.argv[2])
    seed_xy = (float(sys.argv[3]), float(sys.argv[4]))
    model_path = sys.argv[5]
    max_frames_forward = int(sys.argv[6]) if len(sys.argv) > 6 else 60
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectory_verify_output.mp4")

    result = render_trajectory_video(video_path, seed_frame, seed_xy, model_path, out_path, max_frames_forward)
    print("status:", result["status"])
    if result["status"] == "success":
        points = result["points"]
        print(f"tracked {len(points)} points, frames {points[0][0]}-{points[-1][0]}")
        print("wrote", out_path)
