"""Config-driven defaults; the graph never selects a concrete model class."""
from .monocular import MonocularDepthEstimator
from .storage import PrecomputedDepthEstimator
from .depth_pro import DepthProEstimator


def resolve_backend(config,mode):
    if config.depth_path:
        return 'precomputed'
    if config.depth_backend!='auto':
        return config.depth_backend
    return 'depth_anything'


def create_depth_estimator(config,mode):
    backend=resolve_backend(config,mode)
    if backend=='precomputed':
        if not config.depth_path:raise ValueError('precomputed depth backend requires --depth-path')
        return PrecomputedDepthEstimator(config.depth_path,config.depth_metadata_path)
    if backend=='depth_pro':
        return DepthProEstimator(config.depth_model or 'apple/DepthPro-hf',config.depth_device)
    return MonocularDepthEstimator(config.depth_model or 'depth-anything/Depth-Anything-V2-Small-hf',config.depth_device)
