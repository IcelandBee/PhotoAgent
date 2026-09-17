"""Deterministic CPU 2.5D forward splatting with a nearest-surface Z buffer."""
import cv2
import numpy as np
from PIL import Image
from photography_viewpoint_agent.renderer.viewpoint_warp import rotation_homography


def project_points(depth, viewpoint):
    height,width = depth.shape
    _,k,r = rotation_homography((width,height),viewpoint)
    y,x = np.indices(depth.shape,dtype=np.float32)
    valid = np.isfinite(depth) & (depth>0)
    ids = np.flatnonzero(valid)
    z = depth.ravel()[ids]
    xyz = np.stack(((x.ravel()[ids]-k[0,2])*z/k[0,0],
                    (y.ravel()[ids]-k[1,2])*z/k[1,1],z))
    # Camera displacement is in SOURCE axes: +x right, +y up, +z forward.
    center = np.array([viewpoint.translation_x,-viewpoint.translation_y,viewpoint.translation_z])
    transformed = r @ (xyz-center[:,None])
    front = transformed[2]>1e-6
    ids,transformed = ids[front],transformed[:,front]
    uv = k @ transformed
    uv = uv[:2]/uv[2]
    return ids,uv,transformed[2],k,r,center


def repair_background_depth(depth, subject_mask):
    hole = cv2.dilate(np.asarray(subject_mask),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    hole = ((hole>0) | ~np.isfinite(depth) | (depth<=0)).astype(np.uint8)*255
    if np.all(hole):
        raise ValueError('No valid background depth remains outside subject')
    # NS supports float depth; preserve all unmasked source values and original scale.
    repaired = cv2.inpaint(np.where(hole,0,depth).astype(np.float32),hole,3,cv2.INPAINT_NS)
    valid_values = depth[hole==0]
    repaired[hole>0] = np.clip(repaired[hole>0],valid_values.min(),valid_values.max())
    return repaired


class PointCloudViewpointWarper:
    def __init__(self, splat_radius=1):
        if not isinstance(splat_radius,int) or not 1<=splat_radius<=3:
            raise ValueError('splat_radius must be an integer in [1,3]')
        self.radius = splat_radius

    def warp(self, image, depth, viewpoint, fill_color=(128,128,128), subject_bbox=None):
        if viewpoint.mode != 'depth_3d':
            raise ValueError('Point cloud backend requires depth_3d')
        width,height = image.size
        depth = np.asarray(depth,dtype=np.float32)
        if depth.shape != (height,width):
            raise ValueError('Depth and RGB dimensions differ')
        valid_source = np.isfinite(depth)&(depth>0)
        if not valid_source.any():
            raise ValueError('No valid positive depth')
        ids,uv,z,k,r,center = project_points(depth,viewpoint)
        pixels = np.asarray(image.convert('RGB')).reshape(-1,3)
        result = np.full((height*width,3),fill_color,np.uint8)
        projected_box = None
        if subject_bbox is not None:
            sx,sy = ids%width,ids//width
            left,top,right,bottom = subject_bbox
            inside=(sx>=left*width)&(sx<right*width)&(sy>=top*height)&(sy<bottom*height)
            if inside.any():
                p=uv[:,inside]
                projected_box=[float(p[0].min()/width),float(p[1].min()/height),
                    float((p[0].max()+1)/width),float((p[1].max()+1)/height)]
        if not viewpoint.active:
            result[ids]=pixels[ids]
            covered=valid_source.ravel().copy()
        else:
            # Reject out-of-frame projections before integer conversion (near Z=0 can be huge).
            possible=np.isfinite(uv).all(axis=0)&(uv[0]>=-self.radius-1)&(uv[0]<width+self.radius+1)&(uv[1]>=-self.radius-1)&(uv[1]<height+self.radius+1)
            ids,uv,z=ids[possible],uv[:,possible],z[possible]
            base=np.rint(uv).astype(np.int64)
            zbuffer=np.full(height*width,np.inf)
            score=np.full(height*width,np.inf)

            def splats():
                for dy in range(-self.radius,self.radius+1):
                    for dx in range(-self.radius,self.radius+1):
                        x,y=base[0]+dx,base[1]+dy
                        inside=(x>=0)&(x<width)&(y>=0)&(y<height)
                        point=np.flatnonzero(inside)
                        dest=y[inside]*width+x[inside]
                        distance=(x[inside]-uv[0,inside])**2+(y[inside]-uv[1,inside])**2
                        # Tiny stable tie breaker; source indices remain reproducible.
                        cost=distance+ids[inside]/(height*width)*1e-8
                        yield point,dest,cost

            for point,dest,cost in splats():
                np.minimum.at(zbuffer,dest,z[point])
            for point,dest,cost in splats():
                nearest=z[point]==zbuffer[dest]
                np.minimum.at(score,dest[nearest],cost[nearest])
            for point,dest,cost in splats():
                winner=(z[point]==zbuffer[dest])&(cost==score[dest])
                result[dest[winner]]=pixels[ids[point[winner]]]
            covered=np.isfinite(zbuffer)
        valid_mask=covered.reshape(height,width)
        if not covered.any():
            raise ValueError('No points project into target canvas; reduce camera motion')
        result=result.reshape(height,width,3)
        if viewpoint.border_mode=='replicate' and not covered.all():
            _,labels=cv2.distanceTransformWithLabels((~valid_mask).astype(np.uint8),
                cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
            nearest_colors=result[valid_mask]
            result[~valid_mask]=nearest_colors[labels[~valid_mask]-1]
        values=depth[valid_source]
        meta={'backend':'depth_3d','parameters':viewpoint.model_dump(),
            'intrinsics':k.tolist(),'rotation':r.tolist(),'camera_center_source':center.tolist(),
            'extrinsic_translation':(-r@center).tolist(),'image_size':[width,height],
            'splat_radius':self.radius,'kernel_size':2*self.radius+1,
            'valid_fraction':float(covered.mean()),'source_valid_fraction':float(valid_source.mean()),
            'depth_stats':{'min':float(values.min()),'median':float(np.median(values)),'max':float(values.max())},
            'projected_subject_bbox':projected_box,'identity_fast_path':not viewpoint.active}
        return Image.fromarray(result),meta,valid_mask
