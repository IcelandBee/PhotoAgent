import numpy as np
import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError
from photography_viewpoint_agent.schemas.target import TargetState, ViewpointTarget
from photography_viewpoint_agent.schemas.subject import SubjectObservation
from photography_viewpoint_agent.renderer.camera_parameters import CameraWarpParameters
from photography_viewpoint_agent.renderer.viewpoint_warp import RotationViewpointWarper, rotation_homography
from photography_viewpoint_agent.renderer.target_sketch import TargetSketchRenderer


def marker():
    im=Image.new('RGB',(200,200))
    ImageDraw.Draw(im).rectangle((97,97,103,103),fill='white')
    return im


def centroid(im):
    y,x=np.where(np.asarray(im)[:,:,0]>200)
    return np.array([x.mean(),y.mean()])


def test_identity_exact():
    im=Image.fromarray(np.random.default_rng(0).integers(0,256,(80,120,3),dtype=np.uint8))
    result,meta=RotationViewpointWarper().warp(im)
    np.testing.assert_array_equal(result,im)
    np.testing.assert_allclose(meta['homography'],np.eye(3),atol=1e-12)
    assert meta['valid_fraction']==1


@pytest.mark.parametrize('axis,sign',[('yaw',-1),('yaw',1),('pitch',-1),('pitch',1)])
def test_camera_direction_pixels(axis,sign):
    result,_=RotationViewpointWarper().warp(marker(),**{axis+'_deg':sign*5},fill_color=(0,0,0))
    delta=centroid(result)-100
    if axis=='yaw':
        assert delta[0]*sign < -10 and abs(delta[1])<1
    else:
        assert delta[1]*sign > 10 and abs(delta[0])<1


@pytest.mark.parametrize('sign',[-1,1])
def test_roll_direction_pixels(sign):
    im=Image.new('RGB',(200,200))
    ImageDraw.Draw(im).rectangle((147,97,153,103),fill='white')
    result,_=RotationViewpointWarper().warp(im,roll_deg=sign*5,fill_color=(0,0,0))
    assert (centroid(result)[1]-100)*sign < -3


def test_intrinsics_composition_and_stability():
    previous=None
    for angle in np.linspace(-10,10,81):
        spec=CameraWarpParameters(mode='rotation',yaw_deg=angle,pitch_deg=angle/2,roll_deg=3)
        h,k,r=rotation_homography((590,786),spec)
        assert np.isfinite(h).all() and np.linalg.det(r)==pytest.approx(1)
        np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-12)
        np.testing.assert_allclose(h,k@r@np.linalg.inv(k))
        np.testing.assert_allclose(k[0,0],295/np.tan(np.pi/6))
        point=h@np.array([295,393,1]);point=point[:2]/point[2]
        if previous is not None: assert np.linalg.norm(point-previous)<5
        previous=point
    h,_,_=rotation_homography((590,786),CameraWarpParameters(mode='rotation',yaw_deg=5,pitch_deg=-3,roll_deg=2))
    hy,_,_=rotation_homography((590,786),CameraWarpParameters(mode='rotation',yaw_deg=5))
    hp,_,_=rotation_homography((590,786),CameraWarpParameters(mode='rotation',pitch_deg=-3))
    hr,_,_=rotation_homography((590,786),CameraWarpParameters(mode='rotation',roll_deg=2))
    np.testing.assert_allclose(h,hr@hp@hy,atol=1e-12)


def test_border_modes():
    im=Image.new('RGB',(200,200),(10,20,30))
    constant,meta=RotationViewpointWarper().warp(im,yaw_deg=10,fill_color=(77,88,99))
    replicate,_=RotationViewpointWarper().warp(im,yaw_deg=10,border_mode='replicate')
    assert 0<meta['valid_fraction']<1
    assert tuple(np.asarray(constant)[100,-1])==(77,88,99)
    assert tuple(np.asarray(replicate)[100,-1])==(10,20,30)


@pytest.mark.parametrize('fields',[{'yaw_deg':float('nan')},{'pitch_deg':16},
    {'horizontal_fov_deg':0},{'border_mode':'unknown'},{'mode':'none','yaw_deg':1}])
def test_invalid_parameters(fields):
    with pytest.raises(ValidationError):ViewpointTarget.model_validate({'mode':'rotation',**fields})


def test_old_none_zero_identical(frame,tmp_path):
    renderer=TargetSketchRenderer(100,100)
    base={'subject':{'mode':'follow_reference'},'framing':{'reference_viewport':[.1,.1,.9,.9]}}
    paths=[]
    for index,view in enumerate([None,{'mode':'none'},{'mode':'rotation','yaw_deg':0}]):
        data=dict(base)
        if view is not None:data['viewpoint']=view
        path,meta=renderer.render(frame,None,TargetState.model_validate(data),tmp_path/f'{index}.png')
        paths.append(path)
        assert meta.viewpoint_warp is None
    assert len({open(p,'rb').read() for p in paths})==1


def test_combined_and_follow_metadata(frame,tmp_path):
    mask=Image.new('L',(100,100));ImageDraw.Draw(mask).rectangle((40,20,59,79),fill=255)
    mask.save(tmp_path/'mask.png')
    obs=SubjectObservation(bbox=(.4,.2,.6,.8),mask_path=str(tmp_path/'mask.png'),confidence=1)
    spec={'subject':{'mode':'reposition','bbox':[.65,.25,.85,.85]},
        'framing':{'reference_viewport':[.1,.1,.9,.9]},
        'viewpoint':{'mode':'rotation','yaw_deg':5,'pitch_deg':-3}}
    renderer=TargetSketchRenderer(100,100)
    path,meta=renderer.render(frame,obs,TargetState.model_validate(spec),tmp_path/'combined.png')
    assert Image.open(path).size==(100,100)
    assert meta.subject_transform_applied and meta.viewpoint_warp
    assert meta.rendered_subject_bbox[3]==pytest.approx(.85,abs=.01)
    spec['subject']={'mode':'follow_reference'}
    _,meta=renderer.render(frame,obs,TargetState.model_validate(spec),tmp_path/'follow.png')
    assert not meta.subject_transform_applied
    assert meta.natural_subject_bbox[0]<(.4-.1)/.8

def test_reject_camera_plane_crossing():
    with pytest.raises(ValueError,match='camera plane'):
        RotationViewpointWarper().warp(Image.new('RGB',(100,1000)),pitch_deg=15,horizontal_fov_deg=150)


def test_cli_grid(tmp_path):
    import json
    from experiments.viewpoint_rotation.run import main
    source=tmp_path/'source.png';marker().save(source)
    output=tmp_path/'output/single.png'
    assert main(['--input',str(source),'--output',str(output),'--yaw','5','--pitch','-3','--grid'])==0
    assert output.is_file() and (output.parent/'viewpoint_grid.jpg').is_file()
    records=json.loads((output.parent/'grid_metadata.json').read_text())
    assert len(records)==25
    assert len(list(output.parent.glob('yaw_*.jpg')))==25
    assert records[0]['parameters']['yaw_deg']==-10
    assert records[-1]['parameters']['pitch_deg']==10
