import argparse
import json
import sys
from pathlib import Path
from .core import GuideInput, GuideService
from .core.backends import LocalBackend, VLMBackend


def main(argv=None):
    parser = argparse.ArgumentParser(description="Follow a PhotoAgent Target Sketch")
    sub = parser.add_subparsers(dest="command", required=True)
    guide = sub.add_parser("guide")
    for name in ("video-path", "current-frame", "reference-frame", "target-sketch", "target-state", "render-meta", "session-directory"):
        guide.add_argument("--" + name, type=Path, required=True)
    guide.add_argument("--session-id")
    guide.add_argument("--step-id", type=int)
    guide.add_argument("--backend", choices=("local", "vlm"), default="local")
    guide.add_argument("--config", type=Path)
    guide.add_argument("--video-url")
    args = parser.parse_args(argv)
    try:
        backend = LocalBackend() if args.backend == "local" else VLMBackend(config_path=args.config)
        inputs = GuideInput(args.video_path, args.current_frame, args.reference_frame, args.target_sketch,
                            json.loads(args.target_state.read_text(encoding="utf-8-sig")),
                            json.loads(args.render_meta.read_text(encoding="utf-8-sig")),
                            args.session_id, args.step_id, args.video_url)
        result, directory = GuideService(backend).run(inputs, args.session_directory)
        print(json.dumps({"output_directory": str(directory), **result.to_dict()}, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(f"运行失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
