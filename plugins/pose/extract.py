"""
Video (or another estimator's output) -> frame payloads for the pose add-on (docs/POSE.md).

    --video clip.mp4          MediaPipe's pose landmarker on every frame (pip install mediapipe opencv-python;
                              --landmarker pose_landmarker_full.task from Google's MediaPipe model page).
                              World landmarks in metres; up to --people performers per frame
    --openpose DIR            OpenPose's --write_json output (BODY_25), one file per frame. OpenPose is 2D:
                              joints are image coordinates scaled by --scale (pseudo-metres) at z=0 unless the
                              files carry 3D keypoints (pose_keypoints_3d)
    --frames-jsonl FILE       already in this format (e.g. from WHAM or an SMPL-X fit): re-tracked and re-rotated

Writes --out frames.jsonl, and with --images DIR the video's frames as 000012.jpg for review and vision DPO.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pose_core as P


def from_mediapipe(video: str, landmarker: str, people: int, images: Path | None, every: int):
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        raise SystemExit("needs mediapipe and opencv-python: pip install mediapipe opencv-python")
    from mediapipe.tasks.python import vision, BaseOptions
    opts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=landmarker),
                                        running_mode=vision.RunningMode.VIDEO, num_poses=people)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i % every == 0:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 1000 / fps))
                dets = []
                for world, norm2d in zip(res.pose_world_landmarks or [], res.pose_landmarks or []):
                    pts = [[p.x, p.y, p.z] for p in world]
                    px = [[p.x, p.y] for p in norm2d]
                    conf = [getattr(p, "visibility", None) for p in norm2d]
                    # MediaPipe world landmarks: y down, hips at the origin
                    dets.append(P.from_landmarks(pts, P.MEDIAPIPE, px, conf, flip_y=True))
                if images is not None:
                    cv2.imwrite(str(images / f"{i:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
                yield i, i / fps, dets
            i += 1


def from_openpose(folder: str, scale: float, width: float, height: float):
    files = sorted(Path(folder).glob("*_keypoints.json")) or sorted(Path(folder).glob("*.json"))
    for i, fp in enumerate(files):
        d = json.loads(fp.read_text())
        dets = []
        for person in d.get("people") or []:
            k3 = person.get("pose_keypoints_3d") or []
            k2 = person.get("pose_keypoints_2d") or []
            if k3:
                pts = [k3[n:n + 3] if k3[n + 3] > 0 else None for n in range(0, len(k3), 4)]
                flip = True
            else:
                pts = [[k2[n] / width * scale, k2[n + 1] / height * scale, 0.0] if k2[n + 2] > 0 else None
                       for n in range(0, len(k2), 3)]
                flip = True                                      # image y grows downwards
            px = [[k2[n] / width, k2[n + 1] / height] if k2 and k2[n + 2] > 0 else None for n in range(0, len(k2), 3)]
            conf = [k2[n + 2] for n in range(0, len(k2), 3)] if k2 else None
            dets.append(P.from_landmarks(pts, P.BODY25, px or None, conf, flip_y=flip))
        yield i, None, dets


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--openpose")
    src.add_argument("--frames-jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", help="write the video's frames here (with --video)")
    ap.add_argument("--landmarker", default="pose_landmarker_full.task")
    ap.add_argument("--people", type=int, default=4)
    ap.add_argument("--every", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--fps", type=float, default=30.0, help="frame rate for --openpose")
    ap.add_argument("--scale", type=float, default=2.0, help="metres across the image width, for 2D OpenPose")
    ap.add_argument("--width", type=float, default=1920)
    ap.add_argument("--height", type=float, default=1080)
    ap.add_argument("--max-dist", type=float, default=0.6, help="metres a performer's hips may move between frames")
    a = ap.parse_args(argv)
    images = Path(a.images) if a.images else None
    if images:
        images.mkdir(parents=True, exist_ok=True)
    if a.video:
        src = from_mediapipe(a.video, a.landmarker, a.people, images, a.every)
    elif a.openpose:
        src = from_openpose(a.openpose, a.scale, a.width, a.height)
    else:
        src = ((f["frame"], f["t"], list(f["actors"].values())) for f in P.read_frames(a.frames_jsonl))
    tr = P.Tracker(max_dist=a.max_dist)
    n = 0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for i, t, dets in src:
            f.write(json.dumps(P.frame(i, t if t is not None else i / a.fps, tr.update(dets))) + "\n")
            n += 1
    print(f"{n} frames, {tr.next_id - 1} performers tracked -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
