import argparse
import json
import logging
from pathlib import Path
from pydantic import BaseModel
from photography_viewpoint_agent.config.settings import AgentConfig
from photography_viewpoint_agent.graph.workflow import build_workflow
from photography_viewpoint_agent.schemas.target import TargetState


def json_default(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"Cannot serialize {type(value)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Photography Viewpoint Agent V0.3")
    parser.add_argument("--video", required=True)
    parser.add_argument("--target-state", required=True)
    parser.add_argument("--work-dir", default="./workdir")
    parser.add_argument("--frame-sample-interval", type=int, default=30)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--target-width", type=int, default=1280)
    parser.add_argument("--target-height", type=int, default=720)
    parser.add_argument("--yolo-model", default="yolo11n-seg.pt")
    parser.add_argument("--person-confidence", type=float, default=0.4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--debug", action="store_true", help="Save background and RGBA subject layers")
    parser.add_argument("--subject-noop-position-threshold", type=float, default=0.001)
    parser.add_argument("--subject-noop-scale-threshold", type=float, default=0.001)
    parser.add_argument("--fill-color", type=int, nargs=3, metavar=("R", "G", "B"),
                        default=(128, 128, 128), help="Expansion fill color, RGB 0..255")
    parser.add_argument("--subject-position-threshold", type=float, default=0.05)
    parser.add_argument("--subject-scale-threshold", type=float, default=0.05)
    parser.add_argument("--framing-threshold", type=float, default=0.03)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    output = Path(args.work_dir).resolve()
    try:
        config = AgentConfig(**{k: v for k, v in vars(args).items() if k not in ("video", "target_state")})
        output.mkdir(parents=True, exist_ok=True)
        # Refuse reuse to keep failed/new runs from exposing stale successful artifacts.
        if any(output.iterdir()):
            raise ValueError(f"Work directory must be empty; choose a new --work-dir: {output}")
        state = {"video_path": str(Path(args.video).resolve()), "error": None}
        try:
            state["manual_target_state"] = TargetState.model_validate_json(
                Path(args.target_state).read_text(encoding="utf-8-sig"))
            state = build_workflow(config).invoke(state)
        except Exception as exc:
            logging.exception("Workflow failed")
            state["error"] = f"{type(exc).__name__}: {exc}"
        (output / "result.json").write_text(
            json.dumps(state, default=json_default, ensure_ascii=False, indent=2), encoding="utf-8")
        if state.get("error"):
            logging.error("%s", state["error"])
            return 1
        print(f"Current Frame: {state['current_frame'].path}")
        print(f"Reference Frame: {state['reference_frame'].path}")
        print(f"Target State:\n{state['target_state'].model_dump_json(indent=2)}")
        print(f"Target Sketch: {state['target_sketch_path']}")
        print(f"Validation: {'PASS' if state['validation'].passed else 'FAIL'}")
        for message in state["validation"].messages:
            print(f"  {message}")
        print(f"Result: {output / 'result.json'}")
        return 0 if state["validation"].passed else 2
    except Exception:
        logging.exception("Cannot start workflow")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
