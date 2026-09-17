import json
import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError
from photography_viewpoint_agent.schemas.target import ViewpointTarget,TargetState
from photography_viewpoint_agent.schemas.subject import SubjectObservation
from photography_viewpoint_agent.config.settings import AgentConfig
from photography_viewpoint_agent.depth_estimation.monocular import normalize_depth,PrecomputedDepthEstimator,save_observation
from photography_viewpoint_agent.renderer.point_cloud_warp import PointCloudViewpointWarper,project_points,repair_background_depth
from photography_viewpoint_agent.renderer.viewpoint_warp import rotation_homography
from photography_viewpoint_agent.graph.workflow import build_workflow
from photography_viewpoint_agent.app import main


def vp(**kwargs):
    return ViewpointTarget(mode='depth_3d',**kwargs)


def test_normalization_invalid_and_precomputed(tmp_path):
    d,stats=normalize_depth(np.array([[2,4],[0,np.nan]]),(2,2))
    assert np.median(d[d>0])==pytest.approx(1)
    assert d[1].sum()==0 and stats['valid_fraction']==.5
    with pytest.raises(ValueError):normalize_depth(np.zeros((2,2)),(2,2))
    with pytest.raises(ValueError):normalize_depth(np.ones((2,2)),(3,2))


@pytest.mark.parametrize('mode',['none','rotation'])
def test_translations_require_depth(mode):
    with pytest.raises(ValidationError):ViewpointTarget(mode=mode,translation_x=.03)
    assert ViewpointTarget(mode=mode).translation_x==0


@pytest.mark.parametrize('field,value',[('translation_z',float('nan')),('translation_x',.3),('pitch_deg',16)])
def test_invalid_depth_motion(field,value):
    with pytest.raises(ValidationError):vp(**{field:value})


def test_depth_identity_exact_and_deterministic():
    image=Image.fromarray(np.random.default_rng(4).integers(0,256,(60,100,3),dtype=np.uint8))
    depth=np.linspace(.3,2,6000).reshape(60,100)
    warper=PointCloudViewpointWarper()
    result,meta,valid=warper.warp(image,depth,vp())
    np.testing.assert_array_equal(result,image)
    assert valid.all() and meta['identity_fast_path']
    a,_,_=warper.warp(image,depth,vp(translation_x=.03))
    b,_,_=warper.warp(image,depth,vp(translation_x=.03))
    np.testing.assert_array_equal(a,b)


def test_parallax_and_motion_directions():
    depth=np.ones((60,100));depth[:,50:]=2
    ids,uv,_,k,_,_=project_points(depth,vp(translation_x=.03))
    delta=uv[0]-ids%100
    np.testing.assert_allclose(delta[ids%100<50],-k[0,0]*.03,atol=1e-6)
    np.testing.assert_allclose(delta[ids%100>=50],-k[0,0]*.03/2,atol=1e-6)
    ids,uv,_,_,_,_=project_points(depth,vp(translation_y=.03))
    assert np.all(uv[1]-ids//100>0)
    for t in [-.03,.03]:
        ids,uv,_,_,_,_=project_points(depth,vp(translation_z=t))
        radius=np.hypot(ids%100-50,ids//100-30)
        changed=np.hypot(uv[0]-50,uv[1]-30)
        assert np.all((changed-radius)[radius>0]*t>0)


def test_rotation_agrees_with_homography_before_rasterization():
    d=np.linspace(.2,2,6000).reshape(60,100)
    params=vp(yaw_deg=5,pitch_deg=-3,roll_deg=2)
    ids,uv,_,_,_,_=project_points(d,params)
    h,_,_=rotation_homography((100,60),params)
    expected=h@np.stack((ids%100,ids//100,np.ones_like(ids)))
    np.testing.assert_allclose(uv,expected[:2]/expected[2],atol=1e-5)


def test_z_buffer_near_surface_wins_and_splats():
    d=np.zeros((60,100),np.float32);d[30,40]=1;d[30,36]=2
    image=np.zeros((60,100,3),np.uint8);image[30,40]=(255,0,0);image[30,36]=(0,0,255)
    result,meta,valid=PointCloudViewpointWarper().warp(Image.fromarray(image),d,vp(translation_x=.1))
    assert tuple(np.asarray(result)[30,31])==(255,0,0)
    assert valid.sum()>2 and meta['kernel_size']==3
    replicated,m,v=PointCloudViewpointWarper().warp(Image.fromarray(image),d,vp(translation_x=.1,border_mode='replicate'))
    assert m['valid_fraction']==meta['valid_fraction']
    assert not np.any(np.all(np.asarray(replicated)==128,axis=2))
    with pytest.raises(ValueError,match='No points'):
        PointCloudViewpointWarper().warp(Image.fromarray(image),np.full((60,100),.05),vp(translation_z=.1))


def test_background_depth_repair_removes_subject_bump():
    d=np.ones((60,100),np.float32);d[20:40,40:60]=.2
    mask=Image.new('L',(100,60));ImageDraw.Draw(mask).rectangle((40,20,59,39),fill=255)
    repaired=repair_background_depth(d,mask)
    np.testing.assert_allclose(repaired,1,atol=.01)


def make_video(path):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,48))
    assert writer.isOpened()
    for i in range(3):writer.write(np.full((48,64,3),80+i*20,np.uint8))
    writer.release()


@pytest.mark.parametrize('mode',[None,'rotation','depth_3d'])
@pytest.mark.parametrize('subject_mode',['follow_reference','reposition'])
def test_complete_graph_conditional_depth(tmp_path,mode,subject_mode):
    path=tmp_path/'input.avi';make_video(path)
    class Depth:
        calls=0
        def estimate(self,frame,output_path):
            self.calls+=1
            return save_observation(np.ones((frame.height,frame.width)),frame,output_path,{'source':'test'})
    class Detector:
        calls=0
        def detect(self,frame,path):
            self.calls+=1
            mask=Image.new('L',(64,48));ImageDraw.Draw(mask).rectangle((20,10,29,29),fill=255);mask.save(path)
            return SubjectObservation(bbox=(20/64,10/48,30/64,30/48),mask_path=str(path),confidence=1)
    estimator,detector=Depth(),Detector()
    data={'subject':{'mode':subject_mode},'framing':{'reference_viewport':[.1,.1,.9,.9]}}
    if subject_mode=='reposition':data['subject']['bbox']=[.6,.2,.85,.9]
    if mode:data['viewpoint']={'mode':mode,'yaw_deg':3,**({'translation_x':.03} if mode=='depth_3d' else {})}
    out=tmp_path/'run'
    state=build_workflow(AgentConfig(work_dir=str(out),target_width=64,target_height=48,debug=True),
        detector=detector,depth_estimator=estimator).invoke({'video_path':str(path),'manual_target_state':TargetState.model_validate(data)})
    assert not state.get('error'),state.get('error')
    assert state['validation'].passed,state['validation'].messages
    assert estimator.calls==int(mode=='depth_3d')
    assert detector.calls==int(subject_mode=='reposition')
    assert ('reference_depth' in state)==(mode=='depth_3d')
    assert (out/'target_sketch.jpg').is_file()
    if mode=='depth_3d':
        assert (out/'reference_depth_visualization.png').is_file()
        assert (out/'viewpoint_warped_background.jpg').is_file()
        assert (out/'viewpoint_valid_mask.png').is_file()
        assert state['render_meta'].viewpoint_warp['background_depth_repaired']==(subject_mode=='reposition')


def test_precomputed_cli_and_failure(tmp_path):
    video=tmp_path/'video.avi';make_video(video)
    depth=tmp_path/'depth.npy';np.save(depth,np.ones((48,64)))
    args=['--video',str(video),'--target-state','examples/51_depth3d__translate_right_small.json',
          '--depth-path',str(depth),'--target-width','64','--target-height','48']
    assert main(args+['--work-dir',str(tmp_path/'good')])==0
    np.save(depth,np.ones((3,3)))
    assert main(args+['--work-dir',str(tmp_path/'bad')])==1
    state=json.loads((tmp_path/'bad/result.json').read_text())
    assert 'estimate_reference_depth' in state['error']
    assert 'target_sketch_path' not in state
