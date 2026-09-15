import numpy as np
import pytest
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


@pytest.mark.parametrize("viewport,padding", [
    ([0, 0, 1, 1], (0, 0, 0, 0)),
    ([0.1, 0.1, 0.9, 0.9], (0, 0, 0, 0)),
    ([-0.2, -0.2, 1.2, 1.2], (17, 17, 17, 17)),
    ([-0.25, -0.25, 1.25, 1.25], (20, 20, 20, 20)),
    ([-0.2, 0, 1, 1], (20, 0, 0, 0)),
    ([0, 0, 1.2, 1], (0, 0, 20, 0)),
    ([0, -0.2, 1, 1], (0, 20, 0, 0)),
    ([0, 0, 1, 1.2], (0, 0, 0, 20)),
    ([-0.2, 0.1, 0.9, 0.9], (22, 0, 0, 0)),
    ([0.2, 0.2, 1.2, 1.2], (0, 0, 24, 24)),
    ([1.1, 0, 1.5, 1], (0, 0, 120, 0)),
    ([-1, -1, -0.5, -0.5], (120, 120, 0, 0)),
])
def test_virtual_viewport_pixels(frame, target, tmp_path, viewport, padding):
    data = target.model_dump()
    data["framing"]["reference_viewport"] = viewport
    target = type(target).model_validate(data)
    fill = (32, 64, 96)
    path, meta = ViewportRenderer(120, 120, fill).render(frame, target, tmp_path / "result.png")
    pixels = np.array(Image.open(path))
    left, top, right, bottom = padding
    expected = np.full((120, 120, 3), fill, dtype=np.uint8)
    expected[top:120-bottom, left:120-right] = (0, 0, 128)  # navy source
    np.testing.assert_array_equal(pixels, expected)
    assert meta.padding == padding
    assert meta.rendered_viewport == pytest.approx(viewport)


@pytest.mark.parametrize("viewport,bounds", [([0, 0, 1, 1], (0, 0, 100, 100)),
    ([0.1, 0.1, 0.9, 0.9], (10, 10, 90, 90)),
    ([0.501, 0.001, 0.999, 0.999], (50, 0, 100, 100))])
def test_original_crop_pixel_compatibility(frame, target, tmp_path, viewport, bounds):
    source = Image.fromarray(np.random.default_rng(42).integers(0, 256, (100, 100, 3), dtype=np.uint8))
    source.save(frame.path)
    data = target.model_dump()
    data["framing"]["reference_viewport"] = viewport
    path, _ = ViewportRenderer(80, 60).render(frame, type(target).model_validate(data), tmp_path / "result.png")
    expected = source.crop(bounds).resize((80, 60), Image.Resampling.LANCZOS)
    np.testing.assert_array_equal(np.array(Image.open(path)), np.array(expected))


def test_mixed_expansion_crops_correct_source_region(frame, target, tmp_path):
    pixels = np.zeros((100, 100, 3), dtype=np.uint8)
    pixels[:10] = (255, 0, 0)
    pixels[10:90, :90] = (0, 255, 0)
    pixels[90:] = (0, 0, 255)
    pixels[:, 90:] = (255, 255, 0)
    Image.fromarray(pixels).save(frame.path)
    data = target.model_dump()
    data["framing"]["reference_viewport"] = [-0.2, 0.1, 0.9, 0.9]
    path, meta = ViewportRenderer(110, 80).render(frame, type(target).model_validate(data), tmp_path / "mixed.png")
    actual = np.array(Image.open(path))
    assert meta.padding == (20, 0, 0, 0)
    assert np.all(actual[:, :20] == 128)
    assert np.all(actual[:, 20:] == (0, 255, 0))
