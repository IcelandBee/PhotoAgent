import cv2
import numpy as np
from PIL import Image
from photography_viewpoint_agent.schemas.subject import SubjectObservation
from photography_viewpoint_agent.tools.geometry import contain_subject


def load_subject_mask(source: Image.Image, observation: SubjectObservation):
    with Image.open(observation.mask_path) as stored:
        mask = stored.convert("L")
    if mask.size != source.size:
        raise ValueError("Subject mask dimensions do not match reference image")
    mask = mask.point(lambda value: 255 if value > 127 else 0)
    bounds = mask.getbbox()
    if bounds is None:
        raise ValueError("Subject mask is empty")
    width, height = source.size
    observed = (bounds[0]/width, bounds[1]/height, bounds[2]/width, bounds[3]/height)
    if any(abs(a-b) > 1e-6 for a, b in zip(observed, observation.bbox)):
        raise ValueError("Subject observation bbox must match tight mask bounds")
    return mask, bounds


def extract_subject(source: Image.Image, mask: Image.Image, bounds):
    layer = source.convert("RGBA")
    layer.putalpha(mask)
    return layer.crop(bounds)


def build_background(source: Image.Image, mask: Image.Image):
    pixels = np.array(source.convert("RGB"))
    binary = np.array(mask)
    # Three-pixel dilation removes dark mask fringes; feather stays inside the hole.
    hole = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    if np.all(hole):
        raise ValueError("Subject mask leaves no background pixels for inpainting")
    repaired = cv2.inpaint(pixels, hole, 3, cv2.INPAINT_TELEA)
    blurred = cv2.GaussianBlur(repaired, (5, 5), 0.8)
    inner_edge = hole - cv2.erode(hole, np.ones((3, 3), dtype=np.uint8))
    blend = cv2.GaussianBlur(inner_edge, (5, 5), 0.8).astype(np.float32)/255
    blend *= hole/255
    background = repaired*(1-blend[..., None]) + blurred*blend[..., None]
    return Image.fromarray(np.clip(np.rint(background), 0, 255).astype(np.uint8))


def transform_subject_by_bbox(layer: Image.Image, envelope, canvas_size):
    scale, left, top = contain_subject(layer.size, envelope, canvas_size)
    # The same inverse scale on both axes prevents even fractional anisotropic resize.
    placed = layer.transform(canvas_size, Image.Transform.AFFINE,
        (1/scale, 0, -left/scale, 0, 1/scale, -top/scale),
        Image.Resampling.BICUBIC, fillcolor=(0, 0, 0, 0))
    bounds = placed.getchannel("A").getbbox()
    if bounds is None:
        raise ValueError("Target subject is too small to render on the canvas")
    width, height = canvas_size
    actual = (bounds[0]/width, bounds[1]/height, bounds[2]/width, bounds[3]/height)
    return placed, actual


def composite(background: Image.Image, placed_subject: Image.Image):
    return Image.alpha_composite(background.convert("RGBA"), placed_subject).convert("RGB")
