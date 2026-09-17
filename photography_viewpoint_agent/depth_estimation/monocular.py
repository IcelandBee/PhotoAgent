from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from photography_viewpoint_agent.schemas.depth import DepthObservation


def normalize_depth(depth, size):
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape != (size[1], size[0]):
        raise ValueError('Depth dimensions must exactly match reference frame')
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError('Depth contains no finite positive values')
    median = float(np.median(depth[valid]))
    normalized = np.zeros(depth.shape, np.float32)
    normalized[valid] = depth[valid]/median
    if not np.isfinite(normalized).all():
        raise ValueError('Depth normalization overflow')
    return normalized, {'input_median':median,'valid_fraction':float(valid.mean()),
        'min':float(normalized[valid].min()),'median':float(np.median(normalized[valid])),
        'max':float(normalized[valid].max())}


def save_observation(depth, frame, path, metadata):
    normalized, stats = normalize_depth(depth,(frame.width,frame.height))
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.save(path,normalized,allow_pickle=False)
    return DepthObservation(depth_path=str(path.resolve()),width=frame.width,height=frame.height,
        metadata={**metadata,'normalization_stats':stats})


def load_depth(observation, size):
    if (observation.width,observation.height) != size:
        raise ValueError('Depth observation dimensions mismatch')
    depth = np.load(observation.depth_path,allow_pickle=False)
    # Validate the stored normalized Z map; do not reinterpret it as disparity.
    normalized, _ = normalize_depth(depth,size)
    return normalized


def save_visualization(depth, path):
    valid = np.isfinite(depth) & (depth>0)
    inverse = np.zeros_like(depth)
    inverse[valid] = 1/depth[valid]
    lo,hi = np.percentile(inverse[valid],[2,98])
    gray = np.uint8(np.clip((inverse-lo)/max(float(hi-lo),1e-6),0,1)*255)
    rgb = cv2.cvtColor(cv2.applyColorMap(gray,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
    rgb[~valid] = 0
    Image.fromarray(rgb).save(path)


class PrecomputedDepthEstimator:
    """Input .npy is positive camera Z, NOT raw inverse-depth model output."""
    def __init__(self, path):
        self.path = Path(path)

    def estimate(self, frame, output_path):
        depth = np.load(self.path,allow_pickle=False)
        return save_observation(depth,frame,output_path,{'source':'precomputed',
            'input_path':str(self.path.resolve()),'invalid_pixels':'excluded'})


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
