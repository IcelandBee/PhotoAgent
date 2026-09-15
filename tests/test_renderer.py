import numpy as np
import pytest
from PIL import Image
from photography_viewpoint_agent.renderer.viewport import ViewportRenderer
from photography_viewpoint_agent.tools.geometry import fit_viewport


def plan(target, viewport):
    data = target.model_dump()
    data['framing']['reference_viewport'] = viewport
    return type(target).model_validate(data)


@pytest.mark.parametrize('viewport,padding', [
    ([0, 0, 1, 1], (0, 0, 0, 0)),
    ([0.1, 0.1, 0.9, 0.9], (0, 0, 0, 0)),
    ([-0.2, -0.2, 1.2, 1.2], (17, 17, 17, 17)),
    ([-0.25, -0.25, 1.25, 1.25], (20, 20, 20, 20)),
    ([-0.2, 0, 1, 1], (20, 10, 0, 10)),
    ([0, 0, 1.2, 1], (0, 10, 20, 10)),
    ([0, -0.2, 1, 1], (10, 20, 10, 0)),
    ([0, 0, 1, 1.2], (10, 0, 10, 20)),
    ([-0.2, 0.1, 0.9, 0.9], (22, 5, 0, 5)),
    ([0.2, 0.2, 1.2, 1.2], (0, 0, 24, 24)),
    ([1.6, 0, 2, 1], (0, 0, 120, 0)),
    ([-1, -1, -0.5, -0.5], (120, 120, 0, 0)),
])
def test_virtual_viewport_pixels(frame, target, tmp_path, viewport, padding):
    fill = (32, 64, 96)
    path, meta = ViewportRenderer(120, 120, fill).render(frame, plan(target, viewport), tmp_path/'result.png')
    pixels = np.array(Image.open(path))
    left, top, right, bottom = padding
    expected = np.full((120, 120, 3), fill, dtype=np.uint8)
    expected[top:120-bottom, left:120-right] = (0, 0, 128)
    np.testing.assert_array_equal(pixels, expected)
    assert meta.padding == padding
    x0, y0, x1, y1 = meta.rendered_viewport
    assert (x1-x0)/(y1-y0) == pytest.approx(1)
    assert (x0+x1)/2 == pytest.approx((viewport[0]+viewport[2])/2)
    assert (y0+y1)/2 == pytest.approx((viewport[1]+viewport[3])/2)


def test_identity_preserves_pixels(frame, target, tmp_path):
    source = np.random.default_rng(42).integers(0, 256, (100, 100, 3), dtype=np.uint8)
    Image.fromarray(source).save(frame.path)
    path, _ = ViewportRenderer(100, 100).render(frame, plan(target, [0,0,1,1]), tmp_path/'identity.png')
    np.testing.assert_array_equal(np.array(Image.open(path)), source)


@pytest.mark.parametrize('viewport,size,expected', [
    ([0.1,0.1,0.9,0.9], (200,100), (0.1,0.3,0.9,0.7)),
    ([-0.2,-0.2,1.2,1.2], (200,100), (-0.9,-0.2,1.9,1.2)),
    ([0.1,0.1,0.9,0.9], (100,200), (0.3,0.1,0.7,0.9)),
])
def test_aspect_fit(viewport, size, expected):
    fitted, _ = fit_viewport(viewport, (100,100), size)
    assert fitted == pytest.approx(expected)


def test_normalized_units_use_reference_dimensions():
    fitted, scale = fit_viewport((0,0,1,1), (1920,1080), (1280,720))
    assert fitted == pytest.approx((0,0,1,1))
    assert scale == pytest.approx(2/3)


@pytest.mark.parametrize('viewport', [[0.1,0.1,0.9,0.9], [-0.2,-0.2,1.2,1.2]])
def test_square_stays_square(frame, target, tmp_path, viewport):
    pixels = np.zeros((100,100,3),dtype=np.uint8)
    pixels[40:60,40:60] = 255
    Image.fromarray(pixels).save(frame.path)
    path, _ = ViewportRenderer(200,100, (0,0,0)).render(frame, plan(target,viewport), tmp_path/'square.png')
    mask = np.array(Image.open(path))[:,:,0] > 200
    ys, xs = np.nonzero(mask)
    assert abs((xs.max()-xs.min())-(ys.max()-ys.min())) <= 1


def test_mixed_expansion_crops_correct_source_region(frame, target, tmp_path):
    pixels = np.zeros((100,100,3), dtype=np.uint8)
    pixels[:10] = (255,0,0)
    pixels[10:90,:90] = (0,255,0)
    pixels[90:] = (0,0,255)
    pixels[:,90:] = (255,255,0)
    Image.fromarray(pixels).save(frame.path)
    # Physical viewport and canvas already match; no additional aspect correction.
    path, meta = ViewportRenderer(110,80).render(frame, plan(target,[-0.2,0.1,0.9,0.9]),tmp_path/'mixed.png')
    actual = np.array(Image.open(path))
    assert meta.padding == (20,0,0,0)
    assert np.all(actual[:,:20] == 128)
    assert np.all(actual[:,20:] == (0,255,0))
