import numpy as np
import pytest
from PIL import Image
from photography_viewpoint_agent.config.settings import AgentConfig
from photography_viewpoint_agent.schemas.subject import SubjectObservation
from photography_viewpoint_agent.schemas.video import FrameInfo
from photography_viewpoint_agent.schemas.target import TargetState
from photography_viewpoint_agent.renderer.target_sketch import TargetSketchRenderer
from photography_viewpoint_agent.tools.sketch_validator import validate_sketch
from photography_viewpoint_agent.tools.subject_utils import load_subject_mask, build_background


@pytest.fixture
def person(tmp_path):
    pixels = np.full((120, 200, 3), (40, 100, 180), dtype=np.uint8)
    pixels[30:90, 80:100] = (240, 20, 20)
    mask = np.zeros((120, 200), dtype=np.uint8)
    mask[30:90, 80:100] = 255
    path, mask_path = tmp_path/'person.png', tmp_path/'mask.png'
    Image.fromarray(pixels).save(path)
    Image.fromarray(mask).save(mask_path)
    frame = FrameInfo(frame_id='person', path=str(path), frame_index=0, timestamp=0, width=200, height=120)
    observation = SubjectObservation(bbox=(0.4,0.25,0.5,0.75), mask_path=str(mask_path), confidence=0.9)
    return frame, observation


def target_for(bbox, viewport=(0,0,1,1)):
    return TargetState.model_validate({'subject':{'bbox':bbox},'framing':{'reference_viewport':viewport}})


@pytest.mark.parametrize('envelope,actual', [
    ((0.4,0.25,0.5,0.75), (0.4,0.25,0.5,0.75)),  # unchanged
    ((0.1,0.25,0.2,0.75), (0.1,0.25,0.2,0.75)),  # move left
    ((0.7,0.5,0.75,0.75), (0.7,0.5,0.75,0.75)),  # half size
    ((0.05,0,0.25,1), (0.05,0,0.25,1)),           # double size, boundary clip
    ((0.1,0.2,0.5,0.8), (0.24,0.2,0.36,0.8)),   # wide envelope
    ((0.1,0.1,0.15,0.9), (0.1,0.65,0.15,0.9)),  # narrow envelope, height must not fail
])
def test_subject_geometry(person, tmp_path, envelope, actual):
    frame, observation = person
    target = target_for(envelope)
    path, meta = TargetSketchRenderer(200,120,debug=True).render(frame, observation,target,tmp_path/'result.png')
    assert meta.rendered_subject_bbox == pytest.approx(actual, abs=1/120)
    layer = Image.open(tmp_path/'placed_subject.png')
    box = layer.getchannel('A').getbbox()
    assert abs((box[2]-box[0])*3-(box[3]-box[1])) <= 3
    assert (box[0]+box[2])/400 == pytest.approx((envelope[0]+envelope[2])/2, abs=1/200)
    assert box[3]/120 == pytest.approx(envelope[3], abs=1/120)
    result = validate_sketch(target,meta,path,AgentConfig(target_width=200,target_height=120))
    assert result.passed, result.messages
    assert result.subject_scale_error < 0.01
    background = np.array(Image.open(tmp_path/'background_without_subject.jpg'))
    # Removal must actually alter the red source pixels, not merely paste a second person.
    assert background[60,90,0] < 80


def test_subject_and_background_independent(person,tmp_path):
    frame, observation = person
    target = target_for((0.7,0.5,0.75,0.75),(-0.25,-0.25,1.25,1.25))
    path,meta = TargetSketchRenderer(200,120,debug=True).render(frame,observation,target,tmp_path/'result.png')
    pixels = np.array(Image.open(path))
    assert np.all(pixels[0,0] == 128)
    assert pixels[75,145,0] > 200  # moved subject remains independent of background transform
    assert pixels[60,90,0] < 100  # no original red person
    assert meta.rendered_subject_bbox == pytest.approx((0.7,0.5,0.75,0.75))
    assert validate_sketch(target,meta,path,AgentConfig(target_width=200,target_height=120)).passed


@pytest.mark.parametrize('kind',['empty','wrong_size','wrong_bbox'])
def test_invalid_masks(person,tmp_path,kind):
    frame, observation = person
    if kind == 'empty':
        Image.new('L',(200,120)).save(observation.mask_path)
    elif kind == 'wrong_size':
        Image.new('L',(20,20),255).save(observation.mask_path)
    else:
        observation = observation.model_copy(update={'bbox':(0,0,1,1)})
    with pytest.raises(ValueError):
        TargetSketchRenderer(200,120).render(frame,observation,target_for((0,0,1,1)),tmp_path/'bad.png')


def test_no_background_pixels():
    with pytest.raises(ValueError,match='no background'):
        build_background(Image.new('RGB',(20,20)),Image.new('L',(20,20),255))


def test_validator_rejects_distorted_or_misplaced_subject(person,tmp_path):
    frame, observation = person
    target = target_for((0.1,0.2,0.5,0.8))
    path,meta = TargetSketchRenderer(200,120).render(frame,observation,target,tmp_path/'result.png')
    config = AgentConfig(target_width=200,target_height=120)
    for bbox in [(0.1,0.2,0.5,0.8), (0.7,0.2,0.82,0.8), (0.27,0.5,0.33,0.8)]:
        result = validate_sketch(target,meta.model_copy(update={'rendered_subject_bbox':bbox}),path,config)
        assert not result.passed
    wrong_background = meta.model_copy(update={'rendered_viewport':(0.2,0.2,0.8,0.8)})
    assert not validate_sketch(target,wrong_background,path,config).passed


@pytest.mark.parametrize('viewport',[(0.1,0.1,0.9,0.6),(-0.2,0,1.2,1)])
def test_aspect_correction_is_expected_not_validation_failure(person,tmp_path,viewport):
    frame, observation = person
    target = target_for((0.1,0.2,0.5,0.8),viewport)
    path,meta = TargetSketchRenderer(200,120).render(frame,observation,target,tmp_path/'result.png')
    result = validate_sketch(target,meta,path,AgentConfig(target_width=200,target_height=120))
    assert result.passed and result.framing_error == 0
    assert any('aspect-fitted' in message for message in result.messages)
    assert meta.rendered_viewport != target.framing.reference_viewport
