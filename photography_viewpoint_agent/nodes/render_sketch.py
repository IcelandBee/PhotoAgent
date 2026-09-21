from pathlib import Path


def render_target_sketch(state, renderer, config):
    path, meta = renderer.render(state["reference_frame"], None, state["target_state"],
                                 Path(config.work_dir) / "target_sketch.png",
                                 reference_depth=state.get('reference_depth'))
    return {"target_sketch_path": path, "render_meta": meta}
