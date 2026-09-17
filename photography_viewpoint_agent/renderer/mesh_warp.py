"""Edge-aware regular-grid 2.5D mesh; CPU geometry and lazy GPU rasterization."""
from dataclasses import dataclass
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
from .viewpoint_warp import rotation_matrix


@dataclass
class MeshGeometry:
    vertices: np.ndarray
    colors: np.ndarray
    faces: np.ndarray
    source_uv: np.ndarray
    depth_edge_mask: np.ndarray
    valid_vertex_mask: np.ndarray
    face_mask: np.ndarray
    stats: dict


def camera_intrinsics(size,viewpoint,observation):
    width,height=size
    focal=observation.metadata.get('focal_length_px')
    if focal is not None:
        if not np.isfinite(focal) or focal<=0:
            raise ValueError('Invalid estimated focal length; refusing silent fallback')
        if observation.metadata.get('focal_reference_width',width)!=width:
            raise ValueError('Estimated focal belongs to a different image width')
        source='depth_pro'
    else:
        focal=(width/2)/np.tan(np.deg2rad(viewpoint.horizontal_fov_deg)/2)
        source='manual_fov'
    return np.array([[focal,0,width/2],[0,focal,height/2],[0,0,1]],dtype=np.float64),source


def build_mesh(image,depth,k,stride=2,threshold=.12,subject_mask=None):
    """Vectorized face generation; all masks/RGB/Z use one sampling grid.

    A full-resolution edge guard prevents stride from bridging a thin hidden edge.
    The dilated subject region is excluded entirely: conservative boundary cutting.
    """
    if not isinstance(stride,int) or stride<1 or not np.isfinite(threshold) or threshold<=0:
        raise ValueError('Invalid mesh stride or depth edge threshold')
    width,height=image.size
    depth=np.asarray(depth,dtype=np.float32)
    if depth.shape!=(height,width) or min(width,height)<2:
        raise ValueError('Mesh needs matching RGB/depth dimensions of at least 2x2')
    good=np.isfinite(depth)&(depth>0)
    if not good.any():raise ValueError('Mesh has no valid depth')
    logz=np.zeros_like(depth);logz[good]=np.log(depth[good])
    edge=np.zeros_like(good)
    ex=(np.abs(logz[:,1:]-logz[:,:-1])>threshold)&good[:,1:]&good[:,:-1]
    ey=(np.abs(logz[1:]-logz[:-1])>threshold)&good[1:]&good[:-1]
    edge[:,1:]|=ex;edge[:,:-1]|=ex;edge[1:]|=ey;edge[:-1]|=ey
    xs=np.unique(np.r_[np.arange(0,width,stride),width-1]).astype(int)
    ys=np.unique(np.r_[np.arange(0,height,stride),height-1]).astype(int)
    x,y=np.meshgrid(xs,ys);z=depth[y,x];valid=good[y,x]
    safe_z=np.where(valid,z,1)
    vertices=np.stack(((x-k[0,2])*safe_z/k[0,0],(y-k[1,2])*safe_z/k[1,1],safe_z),axis=-1).reshape(-1,3)
    colors=np.asarray(image.convert('RGB'))[y,x].reshape(-1,3).astype(np.float32)/255
    indices=np.arange(x.size).reshape(x.shape)
    a,b,c,d=indices[:-1,:-1].ravel(),indices[:-1,1:].ravel(),indices[1:,:-1].ravel(),indices[1:,1:].ravel()
    candidates=np.stack((np.stack((a,b,c),axis=1),np.stack((b,d,c),axis=1)),axis=1).reshape(-1,3)
    kernel=np.ones((2*stride+1,2*stride+1),np.uint8)
    invalid_guard=cv2.dilate((~good).astype(np.uint8),kernel)[y,x].ravel()>0
    invalid=(~valid.ravel()[candidates]).any(axis=1)|invalid_guard[candidates].any(axis=1)
    logs=np.log(np.maximum(safe_z.ravel()[candidates],1e-30))
    edge_guard=cv2.dilate(edge.astype(np.uint8),kernel)[y,x].ravel()>0
    depth_cut=(np.ptp(logs,axis=1)>threshold)|edge_guard[candidates].any(axis=1)
    mask_cut=np.zeros(len(candidates),bool)
    if subject_mask is not None:
        mask=np.asarray(subject_mask)>0
        if mask.shape!=depth.shape:raise ValueError('Subject mask dimensions mismatch')
        # Three-pixel subject dilation, then stride guard catches boundaries between samples.
        expanded=cv2.dilate(mask.astype(np.uint8),np.ones((7,7),np.uint8))
        sampled=cv2.dilate(expanded,kernel)[y,x].ravel()>0
        mask_cut=sampled[candidates].any(axis=1)
    kept=~(invalid|depth_cut|mask_cut)
    stats={'num_vertices':int(x.size),'num_valid_vertices':int(valid.sum()),
        'num_candidate_faces':len(candidates),'num_kept_faces':int(kept.sum()),
        'num_removed_invalid_faces':int(invalid.sum()),
        'num_removed_depth_edge_faces':int((depth_cut&~invalid).sum()),
        'num_removed_mask_boundary_faces':int((mask_cut&~invalid&~depth_cut).sum()),
        'removal_count_order':'invalid, depth edge, subject region (exclusive counts)',
        'subject_cut_policy':'exclude dilated masked region including interior; no repaired texture stretched across it'}
    return MeshGeometry(vertices,colors,candidates[kept],np.stack((x,y),axis=-1).reshape(-1,2),
        edge,valid,kept.reshape(len(ys)-1,2*(len(xs)-1)),stats)


def transform_mesh(vertices,k,viewpoint,scene_reference_depth):
    normalized=np.array([viewpoint.translation_x,viewpoint.translation_y,viewpoint.translation_z])
    movement=normalized*scene_reference_depth
    center=movement*np.array([1,-1,1])
    r=rotation_matrix(viewpoint)
    target=(r@(vertices-center).T).T
    projected=(k@target.T).T
    uv=projected[:,:2]/np.where(np.abs(projected[:,2:])>1e-8,projected[:,2:],1e-8)
    return target,uv,r,center,movement


def screen_to_ndc(uv,z,size):
    """PyTorch3D +X left/+Y up; align integer source centers to raster pixel centers.

    Non-square NDC scales both axes by the SHORT side. Keep camera Z for correct
    perspective barycentrics and depth sorting; do not use inverse Z here.
    """
    width,height=size
    return np.column_stack(((width-1-2*uv[:,0])/min(size),
                            (height-1-2*uv[:,1])/min(size),z)).astype(np.float32)


class PyTorch3DRasterizer:
    def __init__(self,device='cuda'):
        self.device=device
        self._checked=False

    def check_available(self):
        if self._checked:return
        try:
            import torch
            from pytorch3d import _C
            from pytorch3d.structures import Meshes
        except (ImportError,OSError) as exc:
            raise RuntimeError('depth_mesh requires a CUDA-enabled PyTorch3D build matching torch/CUDA. See docs/depth_mesh.md optional installation; no point-cloud fallback.') from exc
        if torch.device(self.device).type!='cuda' or not torch.cuda.is_available():
            raise RuntimeError('depth_mesh requires an NVIDIA CUDA device; this host has no usable CUDA. See docs/depth_mesh.md. CPU topology tests remain available.')
        try:
            from pytorch3d.renderer.mesh import rasterize_meshes
            vertices=torch.tensor([[-.5,-.5,1],[.5,-.5,1],[0,.5,1]],device=self.device)
            faces=torch.tensor([[0,1,2]],device=self.device)
            rasterize_meshes(Meshes(verts=[vertices],faces=[faces]),image_size=4,faces_per_pixel=1)
            torch.cuda.synchronize(self.device)
        except (RuntimeError,ValueError) as exc:
            raise RuntimeError(f'PyTorch3D CUDA preflight failed: {exc}. Rebuild against this torch/CUDA pair; see docs/depth_mesh.md') from exc
        self._checked=True

    def __call__(self,ndc,faces,colors,size,fill_color):
        self.check_available()
        import torch
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer.mesh import rasterize_meshes
        from pytorch3d.ops import interpolate_face_attributes
        with torch.inference_mode():
            verts=torch.as_tensor(ndc,dtype=torch.float32,device=self.device)
            tri=torch.as_tensor(faces,dtype=torch.int64,device=self.device)
            rgb=torch.as_tensor(colors,dtype=torch.float32,device=self.device)
            meshes=Meshes(verts=[verts],faces=[tri])
            width,height=size
            pix,_,bary,_=rasterize_meshes(meshes,image_size=(height,width),
                # PyTorch3D uses strict triangle interiors at blur=0. A microscopic
                # NDC^2 tolerance includes shared edges, not disocclusion-sized holes.
                faces_per_pixel=1,blur_radius=1e-12,perspective_correct=True,
                cull_backfaces=False,cull_to_frustum=False,clip_barycentric_coords=True)
            valid=pix[0,:,:,0]>=0
            interp=interpolate_face_attributes(pix,bary,rgb[tri])[0,:,:,0]
            fill=torch.tensor(fill_color,dtype=torch.float32,device=self.device)/255
            pixels=torch.where(valid[:,:,None],interp,fill)
            torch.cuda.synchronize(self.device)
            return (pixels.clamp(0,1)*255).round().byte().cpu().numpy(),valid.cpu().numpy()


class MeshViewpointWarper:
    def __init__(self,stride=2,threshold=.12,device='cuda',rasterizer=None):
        self.stride,self.threshold,self.device=stride,threshold,device
        # Injection is for tests; no automatic CPU fallback exists in production.
        self.rasterizer=rasterizer if rasterizer is not None else PyTorch3DRasterizer(device)

    def check_available(self):
        checker=getattr(self.rasterizer,'check_available',None)
        if checker:checker()

    def warp(self,image,depth,viewpoint,observation,fill_color=(128,128,128),subject_mask=None,debug_dir=None,scene_reference_depth=None):
        if viewpoint.mode!='depth_mesh' or viewpoint.border_mode!='constant':
            raise ValueError('Mesh requires depth_mesh and constant fill')
        self.check_available()
        started=time.perf_counter()
        values=np.asarray(depth)[np.isfinite(depth)&(np.asarray(depth)>0)]
        if not len(values):raise ValueError('No valid depth for mesh')
        scene=float(np.median(values)) if scene_reference_depth is None else scene_reference_depth
        k,intrinsics_source=camera_intrinsics(image.size,viewpoint,observation)
        mesh=build_mesh(image,depth,k,self.stride,self.threshold,subject_mask)
        target,uv,r,center,movement=transform_mesh(mesh.vertices,k,viewpoint,scene)
        front=np.isfinite(target).all(axis=1)&(target[:,2]>1e-5)
        keep=front[mesh.faces].all(axis=1)
        faces=mesh.faces[keep]
        ndc=screen_to_ndc(uv,target[:,2],image.size)
        # Faces crossing the near plane are conservatively removed, never stretched.
        ndc[~front]=[0,0,1]
        construction=time.perf_counter()-started
        if debug_dir is not None:
            debug_dir=Path(debug_dir);debug_dir.mkdir(parents=True,exist_ok=True)
            for name,mask in [('depth_edge_mask',mesh.depth_edge_mask),('mesh_valid_vertex_mask',mesh.valid_vertex_mask),('mesh_face_mask',mesh.face_mask)]:
                Image.fromarray(mask.astype(np.uint8)*255).save(debug_dir/f'{name}.png')
        raster_started=time.perf_counter()
        if len(faces):
            rgb,valid=self.rasterizer(ndc,faces,mesh.colors,image.size,fill_color)
        else:
            rgb=np.full((image.height,image.width,3),fill_color,np.uint8)
            valid=np.zeros((image.height,image.width),bool)
        raster_seconds=time.perf_counter()-raster_started
        meta={'backend':'depth_mesh','parameters':viewpoint.model_dump(),
            'depth_backend':observation.metadata.get('depth_backend',observation.metadata.get('source','unknown')),
            'metric_depth':observation.metric,'focal_length_px':float(k[0,0]),'intrinsics_source':intrinsics_source,
            'intrinsics':k.tolist(),'rotation':r.tolist(),'camera_center_source':center.tolist(),
            'scene_reference_depth':scene,'translation_normalized':[viewpoint.translation_x,viewpoint.translation_y,viewpoint.translation_z],
            'translation_metric_equivalent':movement.tolist() if observation.metric else None,
            'translation_in_depth_units':movement.tolist(),'mesh_stride':self.stride,'depth_edge_threshold':self.threshold,
            **mesh.stats,'num_removed_camera_plane_faces':int((~keep).sum()),'num_rasterized_faces':len(faces),
            'valid_fraction':float(np.mean(valid)),'projected_subject_bbox':None,'device':self.device,
            'rasterizer':type(self.rasterizer).__name__,'mesh_construction_seconds':construction,
            'raster_blur_radius_ndc_squared':1e-12,
            'rasterization_seconds':raster_seconds,'total_depth_mesh_seconds':time.perf_counter()-started,
            'depth_inference_seconds':observation.metadata.get('depth_inference_seconds'),
            'depth_node_estimation_seconds':observation.metadata.get('depth_node_estimation_seconds'),
            'mesh_face_mask_layout':'each source grid cell has two horizontal pixels, one per triangle; white=topology kept'}
        result=Image.fromarray(rgb)
        if debug_dir is not None:
            result.save(debug_dir/'viewpoint_mesh_render.jpg',quality=95)
            (debug_dir/'mesh_metadata.json').write_text(json.dumps(meta,indent=2,allow_nan=False),encoding='utf-8')
        return result,meta,valid
