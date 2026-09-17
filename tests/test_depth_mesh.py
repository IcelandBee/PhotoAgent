import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError
from photography_viewpoint_agent.schemas.depth import DepthObservation
from photography_viewpoint_agent.schemas.target import ViewpointTarget,TargetState
from photography_viewpoint_agent.renderer.mesh_warp import build_mesh,camera_intrinsics,transform_mesh,screen_to_ndc,MeshViewpointWarper
from photography_viewpoint_agent.renderer.point_cloud_warp import project_points
from photography_viewpoint_agent.depth_estimation.storage import save_observation,load_depth,PrecomputedDepthEstimator
from photography_viewpoint_agent.depth_estimation.factory import resolve_backend
from photography_viewpoint_agent.config.settings import AgentConfig
from photography_viewpoint_agent.renderer.target_sketch import TargetSketchRenderer
from photography_viewpoint_agent.graph.workflow import build_workflow
from photography_viewpoint_agent.schemas.subject import SubjectObservation
from experiments.fit_target_state.images import frame_for
from .test_depth3d import make_video
from .helpers.mesh_oracle import CpuTriangleOracle


def observation(width=64,height=48,metric=False,metadata=None):
    return DepthObservation(depth_path='unused.npy',width=width,height=height,metric=metric,
        unit='meter' if metric else 'relative',normalization='none' if metric else 'median_one',metadata=metadata or {})


def viewpoint(**kwargs):return ViewpointTarget(mode='depth_mesh',**kwargs)


def sample_image(w=64,h=48):
    y,x=np.indices((h,w));return Image.fromarray(np.stack((x*3,y*4,x+y),axis=-1).astype(np.uint8))


def test_metric_storage_roundtrip_and_sidecar(tmp_path):
    image=sample_image();path=tmp_path/'ref.png';image.save(path);frame=frame_for(path,image.size)
    raw=np.linspace(2,20,64*48).reshape(48,64).astype(np.float32)
    obs=save_observation(raw,frame,tmp_path/'metric.npy',{'focal_length_px':90},metric=True)
    np.testing.assert_array_equal(load_depth(obs,image.size),raw)
    copied=PrecomputedDepthEstimator(obs.depth_path).estimate(frame,tmp_path/'copied.npy')
    assert copied.metric and copied.metadata['focal_length_px']==90
    np.testing.assert_array_equal(load_depth(copied,image.size),raw)
    relative=save_observation(raw,frame,tmp_path/'relative.npy',{})
    assert np.median(load_depth(relative,image.size))==pytest.approx(1)
    with pytest.raises(ValidationError):DepthObservation(depth_path='x',width=2,height=2,metric=True)


def test_config_defaults_and_intrinsics():
    config=AgentConfig()
    assert resolve_backend(config,'depth_mesh')=='depth_pro'
    assert resolve_backend(config,'depth_3d')=='depth_anything'
    assert resolve_backend(AgentConfig(depth_backend='depth_anything'),'depth_mesh')=='depth_anything'
    k,source=camera_intrinsics((64,48),viewpoint(),observation(metadata={'focal_length_px':77}))
    assert k[0,0]==77 and source=='depth_pro'
    k,source=camera_intrinsics((64,48),viewpoint(),observation())
    assert k[0,0]==pytest.approx(32/np.tan(np.pi/6)) and source=='manual_fov'
    with pytest.raises(ValidationError):viewpoint(border_mode='replicate')


def test_flat_identity_and_translation_cpu_oracle():
    image=sample_image();d=np.ones((48,64),np.float32)
    warper=MeshViewpointWarper(stride=2,device='cpu_test_oracle',rasterizer=CpuTriangleOracle())
    output,meta,valid=warper.warp(image,d,viewpoint(),observation())
    assert valid.mean()>.99
    assert np.abs(np.asarray(output).astype(float)-np.asarray(image)).mean()<.5
    output,meta,valid=warper.warp(image,d,viewpoint(translation_x=.04),observation())
    assert valid[2:-2,2:-5].all() and not valid[:,-1].any()


@pytest.mark.parametrize('stride',[1,2,4])
def test_depth_discontinuity_cut(stride):
    im=sample_image();d=np.ones((48,64),np.float32);d[:,32:]=5
    k,_=camera_intrinsics(im.size,viewpoint(),observation())
    mesh=build_mesh(im,d,k,stride=stride,threshold=.12)
    z=mesh.vertices[mesh.faces,2]
    assert np.all(np.ptp(z,axis=1)==0)
    assert mesh.stats['num_removed_depth_edge_faces']>0
    total=mesh.stats['num_kept_faces']+sum(mesh.stats[n] for n in ['num_removed_invalid_faces','num_removed_depth_edge_faces','num_removed_mask_boundary_faces'])
    assert total==mesh.stats['num_candidate_faces']


def test_thin_depth_edge_and_subject_mask_not_bridged():
    im=sample_image();d=np.ones((48,64),np.float32);d[:,31]=5
    k,_=camera_intrinsics(im.size,viewpoint(),observation())
    mesh=build_mesh(im,d,k,stride=4,threshold=.12)
    x=mesh.source_uv[mesh.faces,0]
    assert not ((x.min(axis=1)<31)&(x.max(axis=1)>31)).any()
    mask=np.zeros((48,64),np.uint8);mask[:,31]=255
    mesh=build_mesh(im,np.ones_like(d),k,stride=4,subject_mask=mask)
    x=mesh.source_uv[mesh.faces,0]
    assert not ((x.min(axis=1)<31)&(x.max(axis=1)>31)).any()
    assert mesh.stats['num_removed_mask_boundary_faces']>0


def test_disocclusion_remains_invalid_cpu_oracle():
    im=sample_image();d=np.ones((48,64),np.float32);d[:,32:]=5
    warper=MeshViewpointWarper(device='cpu_test_oracle',rasterizer=CpuTriangleOracle())
    out,meta,valid=warper.warp(im,d,viewpoint(translation_x=.08),observation())
    assert not valid[20:25,29:31].any()
    assert np.all(np.asarray(out)[~valid]==128)


@pytest.mark.parametrize('axis',['translation_x','translation_y','yaw_deg','pitch_deg'])
@pytest.mark.parametrize('sign',[-1,1])
def test_direction_parity(axis,sign):
    d=np.ones((48,64),np.float32);im=sample_image();amount=.03 if axis.startswith('translation') else 5
    v=viewpoint(**{axis:sign*amount});k,_=camera_intrinsics(im.size,v,observation())
    mesh=build_mesh(im,d,k,stride=1)
    xyz,uv,_,_,_=transform_mesh(mesh.vertices,k,v,1)
    ids,point_uv,*_=project_points(d,v.model_copy(update={'mode':'depth_3d'}))
    np.testing.assert_allclose(uv,point_uv.T,atol=1e-5)
    ndc=screen_to_ndc(uv,xyz[:,2],im.size)
    np.testing.assert_allclose((63-ndc[:,0]*48)/2,uv[:,0],atol=1e-5)


def test_metric_motion_scale_invariance():
    im=sample_image();d=np.full((48,64),10,np.float32);v=viewpoint(translation_x=.05)
    k,_=camera_intrinsics(im.size,v,observation())
    mesh=build_mesh(im,d,k);_,uv,_,_,movement=transform_mesh(mesh.vertices,k,v,10)
    small=build_mesh(im,d/10,k);_,uv2,*_=transform_mesh(small.vertices,k,v,1)
    np.testing.assert_allclose(uv,uv2,atol=1e-6);assert movement[0]==pytest.approx(.5)


@pytest.mark.parametrize('case',['right','forward','combined'])
def test_mesh_complete_workflow_with_cpu_oracle(tmp_path,case):
    path=tmp_path/'video.avi';make_video(path)
    class Depth:
        calls=0
        def estimate(self,frame,output_path):
            self.calls+=1
            return save_observation(np.full((48,64),10),frame,output_path,
                {'source':'synthetic_test','focal_length_px':60,'depth_inference_seconds':0},metric=True)
    class Detector:
        def detect(self,frame,path):
            mask=np.zeros((48,64),np.uint8);mask[15:30,20:30]=255;Image.fromarray(mask).save(path)
            return SubjectObservation(bbox=(20/64,15/48,30/64,30/48),mask_path=str(path),confidence=1)
    v=viewpoint(**({'translation_z':.03} if case=='forward' else {'translation_x':.03}))
    target=TargetState.model_validate({'viewpoint':v.model_dump(),'subject':{'mode':'reposition','bbox':[.65,.2,.9,.9]} if case=='combined' else {'mode':'follow_reference'},'framing':{'reference_viewport':[.1,.1,.9,.9] if case=='combined' else [0,0,1,1]}})
    output=tmp_path/'out';depth=Depth()
    renderer=TargetSketchRenderer(64,48,debug=True,mesh_warper=MeshViewpointWarper(device='cpu_test_oracle',rasterizer=CpuTriangleOracle()))
    result=build_workflow(AgentConfig(work_dir=str(output),target_width=64,target_height=48,debug=True),
        renderer=renderer,depth_estimator=depth,detector=Detector()).invoke({'video_path':str(path),'manual_target_state':target})
    assert not result.get('error'),result.get('error')
    assert result['validation'].passed,result['validation'].messages
    assert depth.calls==1 and result['reference_depth'].metric
    assert result['render_meta'].viewpoint_warp['rasterizer']=='CpuTriangleOracle'
    for file in ['depth_metric.npy','reference_depth_visualization.png','depth_edge_mask.png','mesh_valid_vertex_mask.png','mesh_face_mask.png','viewpoint_mesh_render.jpg','viewpoint_valid_mask.png','background_canvas.jpg','target_sketch.jpg','mesh_metadata.json']:
        assert (output/file).is_file()


def test_missing_gpu_fails_before_depth_model(tmp_path,monkeypatch):
    from photography_viewpoint_agent.renderer.mesh_warp import PyTorch3DRasterizer
    def unavailable(self):raise RuntimeError('CUDA unavailable: see docs/depth_mesh.md')
    monkeypatch.setattr(PyTorch3DRasterizer,'check_available',unavailable)
    path=tmp_path/'video.avi';make_video(path)
    class NeverDepth:
        def estimate(self,*args):pytest.fail('Preflight must fail before loading a large model')
    t=TargetState.model_validate({'viewpoint':{'mode':'depth_mesh'},'subject':{'mode':'follow_reference'},'framing':{'reference_viewport':[0,0,1,1]}})
    result=build_workflow(AgentConfig(work_dir=str(tmp_path/'out')),depth_estimator=NeverDepth()).invoke({'video_path':str(path),'manual_target_state':t})
    assert 'CUDA unavailable' in result['error'] and 'reference_depth' not in result

def test_depth_pro_postprocessing_preserves_metric_scale(tmp_path):
    import torch
    pytest.importorskip('transformers.models.depth_pro')
    from types import SimpleNamespace
    from transformers import DepthProImageProcessor
    from photography_viewpoint_agent.depth_estimation.depth_pro import DepthProEstimator
    class Inputs(dict):
        def to(self,device):return self
    real=DepthProImageProcessor()
    class Processor:
        def __call__(self,**kwargs):return Inputs()
        def post_process_depth_estimation(self,*args,**kwargs):
            return real.post_process_depth_estimation(*args,**kwargs)
    estimator=DepthProEstimator()
    estimator.processor=Processor()
    estimator.model=lambda **kwargs:SimpleNamespace(predicted_depth=torch.full((1,4,4),2.),field_of_view=torch.tensor([60.]))
    path=tmp_path/'input.png';sample_image().save(path)
    obs=estimator.estimate(frame_for(path,(64,48)),tmp_path/'depth.npy')
    assert obs.metric and obs.metadata['horizontal_fov_deg']==60
    f=32/np.tan(np.pi/6)
    assert obs.metadata['focal_length_px']==pytest.approx(f)
    np.testing.assert_allclose(load_depth(obs,(64,48)),f/(2*64),rtol=1e-5)
    assert obs.metadata['depth_inference_seconds']>=0
