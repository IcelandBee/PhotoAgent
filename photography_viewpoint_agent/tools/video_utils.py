from pathlib import Path
import math

import cv2
from PIL import Image

from photography_viewpoint_agent.schemas.video import FrameInfo, VideoMeta


def load_video_meta(video_path: str) -> VideoMeta:
    if not Path(video_path).is_file():
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(str(Path(video_path).resolve()))
    try:
        ok, first = cap.read()
        if not cap.isOpened() or not ok:
            raise ValueError(f"Cannot decode video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count < 1:
            raise ValueError("Video has invalid FPS or frame count metadata")
        return VideoMeta(fps=fps, frame_count=int(count), duration=int(count) / fps,
                         width=first.shape[1], height=first.shape[0])
    finally:
        cap.release()


def extract_video_frames(video_path: str, meta: VideoMeta, interval: int,
                         output_dir: Path) -> list[FrameInfo]:
    """Sequential decode avoids unreliable last-frame seeks and bounds RAM usage."""
    if interval <= 0:
        raise ValueError("Sampling interval must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []

    def save(index, pixels):
        frame_id = f"frame_{index:06d}"
        path = (output_dir / f"{frame_id}.jpg").resolve()
        Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)).save(path, quality=95)
        frames.append(FrameInfo(frame_id=frame_id, path=str(path), frame_index=index,
                                timestamp=index / meta.fps,
                                width=pixels.shape[1], height=pixels.shape[0]))

    cap = cv2.VideoCapture(str(Path(video_path).resolve()))
    index, last = 0, None
    try:
        while True:
            ok, pixels = cap.read()
            if not ok:
                break
            last = pixels
            if index % interval == 0:
                save(index, pixels)
            index += 1
    finally:
        cap.release()
    if last is None:
        raise ValueError("Video yielded no frames")
    # Do not silently label an earlier decoded frame as the actual final frame.
    if index < meta.frame_count:
        raise ValueError(f"Video decode ended early: decoded {index}/{meta.frame_count} frames")
    if frames[-1].frame_index != index - 1:
        save(index - 1, last)
    return frames
