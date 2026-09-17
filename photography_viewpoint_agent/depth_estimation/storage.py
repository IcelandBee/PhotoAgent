from pathlib import Path
import json
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


def save_observation(depth, frame, path, metadata, *, metric=False):
    if metric:
        normalized = validate_metric_depth(depth,(frame.width,frame.height))
        values=normalized[normalized>0]
        stats={'min':float(values.min()),'median':float(np.median(values)),'max':float(values.max()),
               'valid_fraction':float(np.mean(normalized>0))}
    else:
        normalized, stats = normalize_depth(depth,(frame.width,frame.height))
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.save(path,normalized,allow_pickle=False)
    observation = DepthObservation(depth_path=str(path.resolve()),width=frame.width,height=frame.height,
        metric=metric,unit='meter' if metric else 'relative',normalization='none' if metric else 'median_one',
        metadata={**metadata,'metric':metric,'unit':'meter' if metric else 'relative','normalization_stats':stats})
    path.with_suffix('.json').write_text(observation.model_dump_json(indent=2),encoding='utf-8')
    return observation


def validate_metric_depth(depth,size):
    depth=np.asarray(depth,dtype=np.float32)
    if depth.shape != (size[1],size[0]):
        raise ValueError('Depth dimensions must exactly match reference frame')
    valid=np.isfinite(depth)&(depth>0)
    if not valid.any():
        raise ValueError('Depth contains no finite positive values')
    return np.where(valid,depth,0).astype(np.float32)


def load_depth(observation, size):
    if (observation.width,observation.height) != size:
        raise ValueError('Depth observation dimensions mismatch')
    depth = np.load(observation.depth_path,allow_pickle=False)
    if observation.metric:
        return validate_metric_depth(depth,size)
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
    def __init__(self, path, metadata_path=None):
        self.path = Path(path)
        self.metadata_path = Path(metadata_path) if metadata_path else self.path.with_suffix('.json')
        self.explicit_metadata = metadata_path is not None

    def estimate(self, frame, output_path):
        depth = np.load(self.path,allow_pickle=False)
        if self.metadata_path.exists() or self.explicit_metadata:
            stored=DepthObservation.model_validate_json(self.metadata_path.read_text(encoding='utf-8-sig'))
            if (stored.width,stored.height)!=(frame.width,frame.height):
                raise ValueError('Depth sidecar dimensions mismatch reference frame')
            return save_observation(depth,frame,output_path,{**stored.metadata,
                'input_source':stored.metadata.get('source'),'source':'precomputed',
                'input_path':str(self.path.resolve())},metric=stored.metric)
        return save_observation(depth,frame,output_path,{'source':'precomputed',
            'input_path':str(self.path.resolve()),'invalid_pixels':'excluded'})
