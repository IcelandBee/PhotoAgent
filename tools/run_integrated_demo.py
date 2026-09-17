"""Offline real-video -> target generation -> Guidance demo; no model download."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from photography_viewpoint_agent.config.settings import AgentConfig
from photography_viewpoint_agent.graph.integrated_workflow import build_integrated_workflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("workdir/integrated-demo"))
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    video = root / "sample.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    if not writer.isOpened():
        raise RuntimeError("MJPG video encoder unavailable")
    try:
        y, x = np.indices((120, 160))
        pixels = np.stack((x, y, (x + y) // 2), axis=-1).astype(np.uint8)
        for _ in range(10):
            writer.write(pixels)
    finally:
        writer.release()
    result = build_integrated_workflow(AgentConfig(target_width=160, target_height=120)).invoke({
        "video_path": str(video), "session_directory": str(root / "session"),
        "session_id": "offline-demo", "step_id": 1,
        "manual_target_state": {"subject": {"mode": "follow_reference"},
                                "framing": {"reference_viewport": [0, 0, 1, 1]}},
    })
    output = root / "integrated_result.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"result": str(output), "guidance": result["guidance"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
