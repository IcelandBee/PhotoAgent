import numpy as np
from PIL import Image
from photography_viewpoint_agent.renderer.viewport import ViewportRenderer


def test_crop_pixels_and_actual_metadata(frame, target, tmp_path):
    pixels = np.zeros((100, 100, 3), dtype=np.uint8)
    pixels[:, :50] = [255, 0, 0]
    pixels[:, 50:] = [0, 0, 255]
    Image.fromarray(pixels).save(frame.path)
    data = target.model_dump()
    data["framing"]["reference_viewport"] = [0.501, 0.001, 0.999, 0.999]
    target = type(target).model_validate(data)
    path, meta = ViewportRenderer(80, 60).render(frame, target, tmp_path / "sketch.jpg")
    with Image.open(path) as image:
        assert image.size == (80, 60)
        assert image.getpixel((40, 30))[2] > 240
    assert meta.rendered_viewport == (0.5, 0, 1, 1)
    assert meta.rendered_subject_bbox is None
