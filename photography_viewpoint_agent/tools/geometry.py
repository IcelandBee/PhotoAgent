"""Pure geometry in source pixels; normalized x and y have different physical units."""
import math


def fit_viewport(viewport, reference_size, canvas_size):
    width, height = reference_size
    cw, ch = canvas_size
    left, top, right, bottom = (viewport[0]*width, viewport[1]*height,
                               viewport[2]*width, viewport[3]*height)
    vw, vh = right-left, bottom-top
    if not all(math.isfinite(v) for v in (left, top, right, bottom, vw, vh)) or min(vw, vh) <= 0:
        raise ValueError("Viewport is not representable in source pixels")
    center_x, center_y = (left+right)/2, (top+bottom)/2
    expansion = viewport[0] < 0 or viewport[1] < 0 or viewport[2] > 1 or viewport[3] > 1
    scale = min(cw/vw, ch/vh) if expansion else max(cw/vw, ch/vh)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid viewport scale")
    vw, vh = cw/scale, ch/scale
    fitted = ((center_x-vw/2)/width, (center_y-vh/2)/height,
              (center_x+vw/2)/width, (center_y+vh/2)/height)
    return fitted, scale


def contain_subject(source_size, envelope, canvas_size):
    cw, ch = canvas_size
    sw, sh = source_size
    left, top, right, bottom = (envelope[0]*cw, envelope[1]*ch, envelope[2]*cw, envelope[3]*ch)
    scale = min((right-left)/sw, (bottom-top)/sh)
    return scale, (left+right-scale*sw)/2, bottom-scale*sh


def transform_bbox_by_viewport(source_bbox, effective_viewport):
    """Unclipped natural bbox in canvas-normalized coordinates, after aspect fit."""
    x0, y0, x1, y1 = effective_viewport
    return ((source_bbox[0]-x0)/(x1-x0), (source_bbox[1]-y0)/(y1-y0),
            (source_bbox[2]-x0)/(x1-x0), (source_bbox[3]-y0)/(y1-y0))


def clip_bbox_to_canvas(box):
    """Visible bbox, or None when fully outside; do not invent a zero-area bbox."""
    if box is None:
        return None
    left, top, right, bottom = max(0,box[0]), max(0,box[1]), min(1,box[2]), min(1,box[3])
    return (left,top,right,bottom) if left < right and top < bottom else None


def expected_subject_bbox(source_bbox, reference_size, envelope, canvas_size):
    rw, rh = reference_size
    cw, ch = canvas_size
    sw, sh = (source_bbox[2]-source_bbox[0])*rw, (source_bbox[3]-source_bbox[1])*rh
    scale, left, top = contain_subject((sw,sh), envelope, canvas_size)
    return (left/cw,top/ch,(left+sw*scale)/cw,(top+sh*scale)/ch)


def subject_is_noop(natural_bbox, expected_bbox, position_threshold, scale_threshold):
    """Only a fully visible subject can reuse its natural image without losing pixels."""
    if natural_bbox is None or any(v < 0 or v > 1 for v in natural_bbox):
        return False
    position = math.hypot((natural_bbox[0]+natural_bbox[2]-expected_bbox[0]-expected_bbox[2])/2,
                          natural_bbox[3]-expected_bbox[3])
    scale = max(abs((natural_bbox[2]-natural_bbox[0])-(expected_bbox[2]-expected_bbox[0])),
                abs((natural_bbox[3]-natural_bbox[1])-(expected_bbox[3]-expected_bbox[1])))
    return position <= position_threshold and scale <= scale_threshold
