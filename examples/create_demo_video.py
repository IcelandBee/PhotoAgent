"""Generate a small deterministic video without downloading assets."""
import argparse
from pathlib import Path
import cv2
import numpy as np


def create_video(path: Path, count: int = 65):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    if not writer.isOpened():
        raise RuntimeError("MP4 writer unavailable")
    try:
        for index in range(count):
            pixels = np.full((240, 320, 3), (80, 120, 170), dtype=np.uint8)
            cv2.rectangle(pixels, (0, 160), (319, 239), (70, 130, 60), -1)
            x = 90 + index
            cv2.circle(pixels, (x, 70), 16, (210, 210, 230), -1)
            cv2.rectangle(pixels, (x - 15, 87), (x + 15, 180), (70, 40, 190), -1)
            cv2.putText(pixels, f"Frame {index}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1)
            writer.write(pixels)
    finally:
        writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("workdir/demo.mp4"))
    create_video(parser.parse_args().output)
