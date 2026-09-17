from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from photography_viewpoint_agent.schemas.depth import DepthObservation


from .storage import normalize_depth, save_observation, load_depth, save_visualization, PrecomputedDepthEstimator


class MonocularDepthEstimator:
    """Depth Anything relative inverse-depth -> heuristic positive Z, median=1."""
    def __init__(self, model='depth-anything/Depth-Anything-V2-Small-hf', device='cpu'):
        self.model_name,self.device = model,device
        self.model = self.processor = None

    def estimate(self, frame, output_path):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        if self.model is None:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name,use_fast=False)
            self.model = AutoModelForDepthEstimation.from_pretrained(self.model_name,
                use_safetensors=True).to(self.device).eval()
            if self.model.config.model_type != 'depth_anything' or getattr(self.model.config,'depth_estimation_type','relative') != 'relative':
                raise ValueError('This estimator expects relative inverse-depth Depth Anything weights')
        with Image.open(frame.path) as image:
            inputs = self.processor(images=image.convert('RGB'),return_tensors='pt').to(self.device)
        with torch.inference_mode():
            raw = self.model(**inputs).predicted_depth
            raw = torch.nn.functional.interpolate(raw[:,None],size=(frame.height,frame.width),
                mode='bicubic',align_corners=False)[0,0].cpu().numpy()
        if not np.isfinite(raw).all():
            raise ValueError('Model returned nonfinite inverse depth')
        lo,hi = np.percentile(raw,[2,98])
        # Affine ambiguity is unresolved: choose positive inverse depth in [0.1,1.0].
        inverse = .1+.9*np.clip((raw-lo)/max(float(hi-lo),1e-6),0,1)
        return save_observation(1/inverse,frame,output_path,{'source':'monocular',
            'model':self.model_name,'device':self.device,'raw_representation':'relative_inverse_depth',
            'raw_percentiles_2_98':[float(lo),float(hi)],
            'conversion':'inverse=0.1+0.9*clip((raw-p2)/(p98-p2),0,1); Z=1/inverse; median(Z)=1',
            'metric':False})
