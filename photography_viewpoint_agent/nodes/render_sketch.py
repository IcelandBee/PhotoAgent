from pathlib import Path


def render_target_sketch(state, renderer, config):
    kwargs = ({'reference_depth':state.get('reference_depth')}
              if state['target_state'].viewpoint.mode=='depth_3d' else {})
    path, meta = renderer.render(state["reference_frame"], state.get("reference_subject"), state["target_state"],
                                 Path(config.work_dir) / "target_sketch.jpg",**kwargs)
    return {"target_sketch_path": path, "render_meta": meta}
