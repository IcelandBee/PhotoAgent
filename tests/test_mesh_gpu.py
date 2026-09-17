"""Run on the NVIDIA server; skipped honestly on CPU/no-PyTorch3D hosts."""
import importlib.util
import numpy as np
import pytest
import torch
from PIL import Image
from photography_viewpoint_agent.schemas.depth import DepthObservation
from photography_viewpoint_agent.schemas.target import ViewpointTarget
from photography_viewpoint_agent.renderer.mesh_warp import MeshViewpointWarper

pytestmark=pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec('pytorch3d') is None,
    reason='Requires an actual CUDA-enabled PyTorch3D build')


def make_case():
    y,x=np.indices((48,64));im=Image.fromarray(np.stack((x*3,y*4,x+y),axis=-1).astype(np.uint8))
    obs=DepthObservation(depth_path='unused',width=64,height=48)
    return im,np.ones((48,64),np.float32),obs


@pytest.mark.parametrize('stride',[1,2])
def test_gpu_flat_identity(stride):
    im,d,obs=make_case()
    out,meta,valid=MeshViewpointWarper(stride=stride).warp(im,d,ViewpointTarget(mode='depth_mesh'),obs)
    assert valid.mean()>.98
    assert np.abs(np.asarray(out).astype(float)[valid]-np.asarray(im)[valid]).mean()<1.5
    assert meta['rasterizer']=='PyTorch3DRasterizer'


def test_gpu_flat_translation_no_pinholes():
    im,d,obs=make_case()
    out,meta,valid=MeshViewpointWarper().warp(im,d,ViewpointTarget(mode='depth_mesh',translation_x=.04),obs)
    assert valid[3:-3,3:-6].all()
    assert not valid[:,-1].any()


def test_gpu_depth_discontinuity_leaves_hole():
    im,d,obs=make_case();d[:,32:]=5
    out,meta,valid=MeshViewpointWarper().warp(im,d,ViewpointTarget(mode='depth_mesh',translation_x=.08),obs)
    assert meta['num_removed_depth_edge_faces']>0
    assert not valid[20:25,29:31].any()
    assert np.all(np.asarray(out)[~valid]==128)


def test_gpu_zbuffer_and_perspective_colors():
    from photography_viewpoint_agent.renderer.mesh_warp import PyTorch3DRasterizer,screen_to_ndc
    uv=np.array([[8,8],[52,8],[8,40]],float)
    ndc=np.vstack((screen_to_ndc(uv,np.array([1,2,4]),(64,48)),screen_to_ndc(uv,np.full(3,10),(64,48))))
    faces=np.array([[3,4,5],[0,1,2]])
    colors=np.vstack((np.eye(3),np.zeros((3,3))))
    image,valid=PyTorch3DRasterizer()(ndc,faces,colors,(64,48),(128,128,128))
    u,v=19,16
    screen=np.array([.5,.25,.25]);weights=screen/np.array([1,2,4]);weights/=weights.sum()
    assert valid[v,u]
    np.testing.assert_allclose(image[v,u],weights*255,atol=2)

@pytest.mark.parametrize('axis,amount,direction',[('yaw_deg',5,-1),('yaw_deg',-5,1),('pitch_deg',5,1),('pitch_deg',-5,-1),('translation_x',.03,-1),('translation_x',-.03,1)])
def test_gpu_camera_directions(axis,amount,direction):
    im,d,obs=make_case()
    a=np.zeros((48,64,3),np.uint8);a[20:29,28:37]=255
    out,meta,valid=MeshViewpointWarper(stride=1).warp(Image.fromarray(a),d,ViewpointTarget(mode='depth_mesh',**{axis:amount}),obs)
    y,x=np.where(np.asarray(out)[:,:,0]>220)
    displacement=y.mean()-24 if axis=='pitch_deg' else x.mean()-32
    assert displacement*direction>1
