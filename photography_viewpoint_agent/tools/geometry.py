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
