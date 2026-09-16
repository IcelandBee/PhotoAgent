from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from photography_viewpoint_agent.config.settings import AgentConfig
from photography_viewpoint_agent.schemas.subject import SubjectObservation
from photography_viewpoint_agent.schemas.target import TargetState
from photography_viewpoint_agent.renderer.target_sketch import TargetSketchRenderer
from photography_viewpoint_agent.renderer.viewport import ViewportRenderer
from photography_viewpoint_agent.tools.geometry import (
    fit_viewport, transform_bbox_by_viewport, clip_bbox_to_canvas, subject_is_noop,
)
from photography_viewpoint_agent.tools.sketch_validator import validate_sketch


def target_for(mode='follow_reference',bbox=None,viewport=(0,0,1,1)):
    return TargetState.model_validate({'subject':{'mode':mode,'bbox':bbox},
                                      'framing':{'reference_viewport':viewport}})


@pytest.fixture
def no_edit_tools(monkeypatch):
    import photography_viewpoint_agent.renderer.target_sketch as module
    def forbidden(*args,**kwargs):
        pytest.fail('Follow/no-op path must not load masks, extract, inpaint, transform, or composite')
    for name in ['load_subject_mask','extract_subject','build_background','transform_subject_by_bbox','composite']:
        monkeypatch.setattr(module,name,forbidden)


@pytest.mark.parametrize('viewport',[(0,0,1,1),(0.1,0.1,0.9,0.9),(-0.2,-0.2,1.2,1.2)])
@pytest.mark.parametrize('known_subject',[False,True])
def test_follow_exact_full_image_transform(frame,tmp_path,viewport,known_subject,no_edit_tools):
    pixels = np.random.default_rng(42).integers(0,256,(100,100,3),dtype=np.uint8)
    Image.fromarray(pixels).save(frame.path)
    observation = SubjectObservation(bbox=(0.3,0.3,0.5,0.7),mask_path='unused.png',confidence=0.9) if known_subject else None
    target = target_for(viewport=viewport)
    path,meta = TargetSketchRenderer(100,100,debug=True).render(frame,observation,target,tmp_path/'follow.png')
    expected,_ = ViewportRenderer(100,100).render(frame,target,tmp_path/'expected.png')
    np.testing.assert_array_equal(np.array(Image.open(path)),np.array(Image.open(expected)))
    if viewport == (0,0,1,1):
        np.testing.assert_array_equal(np.array(Image.open(path)),pixels)
    assert not meta.subject_transform_applied
    assert meta.subject_mode == 'follow_reference'
    assert not (tmp_path/'background_without_subject.jpg').exists()
    if not known_subject:
        assert meta.natural_subject_bbox is None and meta.rendered_subject_bbox is None
    result = validate_sketch(target,meta,path,AgentConfig(target_width=100,target_height=100))
    assert result.passed and result.subject_position_error is None and result.subject_scale_error is None


@pytest.mark.parametrize('viewport,expected',[
    ((0,0,1,1),(0.3,0.3,0.5,0.7)),
    ((0.1,0.1,0.9,0.9),(0.25,0.25,0.5,0.75)),
    ((-0.5,-0.5,1.5,1.5),(0.4,0.4,0.5,0.6)),
    ((-0.2,0.1,0.8,0.9),(0.5,0.25,0.7,0.75)),
    ((0.4,0.4,0.6,0.6),(-0.5,-0.5,0.5,1.5)),
])
def test_natural_bbox_mapping(viewport,expected):
    assert transform_bbox_by_viewport((0.3,0.3,0.5,0.7),viewport) == pytest.approx(expected)


def test_natural_bbox_uses_effective_viewport(frame,tmp_path,no_edit_tools):
    observation = SubjectObservation(bbox=(0.3,0.4,0.5,0.6),mask_path='unused.png',confidence=0.9)
    target = target_for(viewport=(0.1,0.1,0.9,0.9))
    path,meta = TargetSketchRenderer(200,100).render(frame,observation,target,tmp_path/'effective.png')
    assert meta.rendered_viewport == pytest.approx((0.1,0.3,0.9,0.7))
    assert meta.natural_subject_bbox == pytest.approx((0.25,0.25,0.5,0.75))
    assert validate_sketch(target,meta,path,AgentConfig(target_width=200,target_height=100)).passed


@pytest.mark.parametrize('source_bbox,visible',[
    ((0,0,0.1,0.1),None), ((0.3,0.4,0.5,0.7),(0,0,0.5,1)),
])
def test_follow_clipping_and_fully_outside(frame,tmp_path,source_bbox,visible,no_edit_tools):
    observation = SubjectObservation(bbox=source_bbox,mask_path='unused.png',confidence=0.9)
    target = target_for(viewport=(0.4,0.4,0.6,0.6))
    path,meta = TargetSketchRenderer(100,100).render(frame,observation,target,tmp_path/'clip.png')
    if visible is None:
        assert meta.rendered_subject_bbox is None
    else:
        assert meta.rendered_subject_bbox == pytest.approx(visible)
    assert validate_sketch(target,meta,path,AgentConfig(target_width=100,target_height=100)).passed


@pytest.mark.parametrize('bbox',[(0.3,0.3,0.5,0.7),(0.3005,0.3,0.5005,0.7),(0.2,0.3,0.6,0.7)])
def test_noop_exact_near_and_wide_envelope(frame,tmp_path,bbox,no_edit_tools):
    observation = SubjectObservation(bbox=(0.3,0.3,0.5,0.7),mask_path='unused.png',confidence=0.9)
    target = target_for('reposition',bbox)
    path,meta = TargetSketchRenderer(100,100,debug=True).render(frame,observation,target,tmp_path/'noop.png')
    assert meta.subject_mode == 'reposition' and not meta.subject_transform_applied
    assert meta.natural_subject_bbox == meta.rendered_subject_bbox == observation.bbox
    assert validate_sketch(target,meta,path,AgentConfig(target_width=100,target_height=100)).passed


def test_noop_under_expansion(frame,tmp_path,no_edit_tools):
    observation = SubjectObservation(bbox=(0.3,0.3,0.5,0.7),mask_path='unused.png',confidence=0.9)
    target = target_for('reposition',(0.4,0.4,0.5,0.6),(-0.5,-0.5,1.5,1.5))
    path,meta = TargetSketchRenderer(100,100).render(frame,observation,target,tmp_path/'noop.png')
    assert not meta.subject_transform_applied
    assert meta.rendered_subject_bbox == pytest.approx((0.4,0.4,0.5,0.6))
    assert validate_sketch(target,meta,path,AgentConfig(target_width=100,target_height=100)).passed


@pytest.mark.parametrize('bbox',[(0.4,0.3,0.6,0.7),(0.3,0.3,0.5,0.8)])
def test_significant_changes_do_not_noop(frame,tmp_path,bbox):
    observation = SubjectObservation(bbox=(0.3,0.3,0.5,0.7),mask_path=str(tmp_path/'missing.png'),confidence=0.9)
    # Missing mask is reached only when actual independent editing is required.
    with pytest.raises(FileNotFoundError):
        TargetSketchRenderer(100,100).render(frame,observation,target_for('reposition',bbox),tmp_path/'result.png')


def test_partial_crop_cannot_noop():
    assert not subject_is_noop((-0.1,0.2,0.4,0.8),(0,0.2,0.4,0.8),0.5,0.5)
    assert not subject_is_noop((0.3,0.3,0.5,0.7),(0.31,0.3,0.51,0.7),0.001,0.001)


def test_follow_validator_rejects_geometry_and_mode_lies(frame,tmp_path):
    observation = SubjectObservation(bbox=(0.3,0.3,0.5,0.7),mask_path='unused.png',confidence=0.9)
    target = target_for()
    path,meta = TargetSketchRenderer(100,100).render(frame,observation,target,tmp_path/'follow.png')
    config = AgentConfig(target_width=100,target_height=100)
    for updates in [{'subject_transform_applied':True},{'subject_mode':'reposition'},
                    {'natural_subject_bbox':(0.1,0.1,0.2,0.2)},
                    {'rendered_subject_bbox':(0.1,0.1,0.2,0.2)},
                    {'requested_viewport':(-1,-1,2,2)}]:
        assert not validate_sketch(target,meta.model_copy(update=updates),path,config).passed


def test_examples_are_valid():
    for path in Path('examples').glob('*.json'):
        TargetState.model_validate_json(path.read_text(encoding='utf-8-sig'))


def test_reposition_requires_observation(frame,tmp_path):
    with pytest.raises(ValueError,match='requires reference_subject'):
        TargetSketchRenderer(100,100).render(frame,None,target_for('reposition',(0.1,0.2,0.4,0.8)),tmp_path/'bad.png')


def test_configurable_noop_threshold(frame,tmp_path,no_edit_tools):
    observation = SubjectObservation(bbox=(0.3,0.3,0.5,0.7),mask_path='unused.png',confidence=0.9)
    target = target_for('reposition',(0.302,0.3,0.502,0.7))
    path,meta = TargetSketchRenderer(100,100,subject_noop_position_threshold=0.003).render(
        frame,observation,target,tmp_path/'noop.png')
    assert not meta.subject_transform_applied
    config = AgentConfig(target_width=100,target_height=100,subject_noop_position_threshold=0.003)
    assert validate_sketch(target,meta,path,config).passed
    assert not validate_sketch(target,meta,path,AgentConfig(target_width=100,target_height=100)).passed
