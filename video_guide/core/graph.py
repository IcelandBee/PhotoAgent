"""Guidance subgraph: one analysis per step, immutable session target."""
import hashlib
import json
import shutil
import uuid
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from .models import GuideInput, GuideResult
from .backends import LocalBackend
from .state import GuideGraphInput, GuideGraphOutput, GuideState
from .vision import read_image


def transient_model_error(error):
    if isinstance(error, HTTPError):
        return error.code in (429, 500, 502, 503, 504)
    return isinstance(error, (TimeoutError, ConnectionError, URLError))


def write_json(path, value):
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def ensure_target(root, inputs):
    """Publish once; reject replacement even if source paths are reused."""
    signature = {"video_path": str(Path(inputs.video_path).resolve()),
                 "target_state": inputs.target_state, "render_meta": inputs.render_meta}
    for name in ("reference_frame", "target_sketch"):
        signature[name] = hashlib.sha256(Path(getattr(inputs, name)).read_bytes()).hexdigest()
    signature = json.loads(json.dumps(signature, allow_nan=False))
    destination = root / "target"
    def check():
        saved = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if saved["signature"] != signature or saved["session_id"] != inputs.session_id:
            raise ValueError("Session target is locked; use a new session for a different target")
        for name in ("reference_frame", "target_sketch"):
            if hashlib.sha256(Path(saved["paths"][name]).read_bytes()).hexdigest() != signature[name]:
                raise ValueError("Archived target was modified")
        for name in ("target_state", "render_meta"):
            if json.loads((destination / (name + ".json")).read_text(encoding="utf-8")) != signature[name]:
                raise ValueError("Archived target metadata was modified")
        return saved["paths"]
    if destination.exists():
        return check()
    staging = root / (".target-" + uuid.uuid4().hex)
    staging.mkdir()
    paths = {"video_path": signature["video_path"]}
    for name in ("reference_frame", "target_sketch"):
        source = Path(getattr(inputs, name))
        filename = name + source.suffix.lower()
        shutil.copyfile(source, staging / filename)
        if hashlib.sha256((staging / filename).read_bytes()).hexdigest() != signature[name]:
            raise ValueError("Target source changed during publication")
        paths[name] = str(destination / filename)
    for name in ("target_state", "render_meta"):
        write_json(staging / (name + ".json"), signature[name])
    write_json(staging / "manifest.json", {"signature": signature, "paths": paths, "session_id": inputs.session_id})
    try:
        staging.rename(destination)
    except OSError:
        if not destination.exists():
            raise
        # Another step may publish the same target concurrently.
        result = check()
        for child in staging.iterdir():
            child.unlink()
        staging.rmdir()
        return result
    return paths


def build_guide_graph(backend=None, *, checkpointer=None):
    engine = backend if backend is not None else LocalBackend()

    def validate_input(state: GuideState):
        inputs = GuideInput(**{f.name: state[f.name] for f in fields(GuideInput) if f.name in state})
        inputs.validate()
        for name in ("current_frame", "reference_frame", "target_sketch"):
            read_image(getattr(inputs, name))
        root = Path(state["session_directory"]).resolve()
        root.mkdir(parents=True, exist_ok=True)
        paths = ensure_target(root, inputs)
        steps = root / "guidance"
        steps.mkdir(exist_ok=True)
        step = inputs.step_id or 1
        while True:
            destination = steps / f"step_{step:06d}"
            staging = steps / f".pending-step_{step:06d}"
            if destination.exists():
                if inputs.step_id is not None:
                    raise ValueError("Guidance step already exists; refusing overwrite")
                step += 1
                continue
            try:
                staging.mkdir()
                break
            except FileExistsError:
                if inputs.step_id is not None:
                    raise ValueError("Guidance step is pending; resume its checkpoint or choose a new step") from None
                step += 1
        execution_id = uuid.uuid4().hex
        write_json(staging / "execution.json", {"execution_id": execution_id})
        return {"execution_id": execution_id, "session_directory": str(root), "step_id": step,
                "staging_directory": str(staging), "saved_inputs": paths}

    def prepare_inputs(state: GuideState):
        staging = Path(state["staging_directory"])
        source = Path(state["current_frame"])
        target = staging / ("current_frame" + source.suffix.lower())
        temporary = target.with_name(target.name + ".part")
        shutil.copyfile(source, temporary)
        temporary.replace(target)
        return {"saved_inputs": {**state["saved_inputs"], "current_frame": str(target)}}

    def analyze_alignment(state: GuideState):
        paths = state["saved_inputs"]
        current, reference, target = [read_image(paths[n]) for n in ("current_frame", "reference_frame", "target_sketch")]
        result = engine.analyze(current, reference, target, Path(paths["video_path"]),
                                state["target_state"], state["render_meta"], state.get("video_url"))
        # Apply the same guard to custom/local backends as to VLM responses.
        result = GuideResult.from_dict(result.to_dict())
        if any(action.actor == 'subject' for action in result.actions):
            raise ValueError('Scene-fixed workflow forbids independent subject actions')
        return {"result": result.to_dict()}

    def persist_result(state: GuideState):
        staging = Path(state["staging_directory"])
        destination = Path(state["session_directory"]) / "guidance" / f"step_{state['step_id']:06d}"
        if destination.exists():
            manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("execution_id") != state["execution_id"]:
                raise ValueError("Step belongs to another execution; refusing overwrite")
            return {"output_directory": str(destination)}
        saved = dict(state["saved_inputs"])
        saved["current_frame"] = str(destination / Path(saved["current_frame"]).name)
        manifest = {"schema_version": 4, "execution_id": state["execution_id"],
                    "session_id": state.get("session_id"), "step_id": state["step_id"],
                    "created_at": datetime.now(timezone.utc).isoformat(), "backend": type(engine).__name__,
                    "orchestrator": "langgraph", "inputs": saved, "result": "result.json",
                    "video_transport": state["result"]["evidence"].get("video_transport", "url" if state.get("video_url") else "file"),
                    "model_settings": state["result"]["evidence"]}
        write_json(staging / "result.json", state["result"])
        write_json(staging / "manifest.json", manifest)
        staging.rename(destination)
        return {"output_directory": str(destination)}

    builder = StateGraph(GuideState, input_schema=GuideGraphInput, output_schema=GuideGraphOutput)
    for name, node in (("validate_input", validate_input), ("prepare_inputs", prepare_inputs), ("persist_result", persist_result)):
        builder.add_node(name, node)
    builder.add_node("analyze_alignment", analyze_alignment,
                     retry_policy=RetryPolicy(max_attempts=2, retry_on=transient_model_error))
    builder.add_edge(START, "validate_input")
    builder.add_edge("validate_input", "prepare_inputs")
    builder.add_edge("prepare_inputs", "analyze_alignment")
    builder.add_edge("analyze_alignment", "persist_result")
    builder.add_edge("persist_result", END)
    return builder.compile(checkpointer=checkpointer)
