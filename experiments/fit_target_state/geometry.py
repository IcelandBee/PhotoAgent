import numpy as np
from photography_viewpoint_agent.schemas.target import TargetState

PARAMETER_NAMES = ("viewport_center_x", "viewport_center_y", "log_zoom_scale",
                   "subject_bottom_center_x", "subject_bottom_y", "subject_height")


def decode(parameters, reference_size, canvas_size, source_bbox, config):
    """Canonicalize six internal parameters and return standard TargetState."""
    p = np.asarray(parameters, dtype=float).copy()
    p[:2] = np.clip(p[:2], *config.center_bounds)
    p[2] = np.clip(p[2], *np.log(config.zoom_bounds))
    zoom = np.exp(p[2])
    rw, rh = reference_size
    cw, ch = canvas_size
    vw, vh = 1/zoom, (rw/rh)*(ch/cw)/zoom
    viewport = [p[0]-vw/2, p[1]-vh/2, p[0]+vw/2, p[1]+vh/2]
    bbox = None
    if config.subject_mode == "reposition":
        sw = (source_bbox[2]-source_bbox[0])*rw
        sh = (source_bbox[3]-source_bbox[1])*rh
        normalized_ratio = (sw/sh)*(ch/cw)
        h = np.clip(p[5], 0.005, min(1, 1/normalized_ratio))
        w = h*normalized_ratio
        cx, bottom = np.clip(p[3], w/2, 1-w/2), np.clip(p[4], h, 1)
        p[3:6] = cx, bottom, h
        bbox = [max(0,cx-w/2), max(0,bottom-h), min(1,cx+w/2), bottom]
    state = TargetState.model_validate({"subject":{"mode":config.subject_mode,"bbox":bbox},
                                       "framing":{"reference_viewport":viewport}})
    return p, state


def geometry_features(box):
    return np.array([(box[0]+box[2])/2, box[3], box[3]-box[1]])
