"""Tiny CPU oracle used ONLY in synthetic tests. Not a production fallback."""
import numpy as np


class CpuTriangleOracle:
    def __call__(self,ndc,faces,colors,size,fill_color):
        width,height=size
        uv=np.column_stack(((width-1-ndc[:,0]*min(size))/2,(height-1-ndc[:,1]*min(size))/2))
        rgb=np.full((height,width,3),fill_color,np.uint8)
        zbuffer=np.full((height,width),np.inf)
        for face in faces:  # Small synthetic meshes only; production construction is vectorized.
            p=uv[face];z=ndc[face,2]
            x0,y0=np.maximum(np.floor(p.min(axis=0)).astype(int),0)
            x1,y1=np.minimum(np.ceil(p.max(axis=0)).astype(int),[width-1,height-1])
            if x0>x1 or y0>y1:continue
            matrix=np.vstack((p.T,np.ones(3)))
            if abs(np.linalg.det(matrix))<1e-8:continue
            yy,xx=np.mgrid[y0:y1+1,x0:x1+1]
            bary=np.linalg.solve(matrix,np.stack((xx.ravel(),yy.ravel(),np.ones(xx.size))))
            inside=(bary>=-1e-5).all(axis=0)
            perspective=bary/z[:,None];perspective/=perspective.sum(axis=0)
            zz=(perspective*z[:,None]).sum(axis=0)
            win=inside&(zz<zbuffer[yy.ravel(),xx.ravel()])
            color=np.clip(perspective.T@colors[face]*255,0,255).round().astype(np.uint8)
            rgb[yy.ravel()[win],xx.ravel()[win]]=color[win]
            zbuffer[yy.ravel()[win],xx.ravel()[win]]=zz[win]
        return rgb,np.isfinite(zbuffer)
