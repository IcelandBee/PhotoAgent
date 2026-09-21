from pathlib import Path
from photography_viewpoint_agent.depth_estimation.monocular import load_depth, save_visualization
from photography_viewpoint_agent.depth_estimation.factory import create_depth_estimator
import numpy as np
import time


def estimate_reference_depth(state, estimator, config):
    frame = state['reference_frame']
    output = Path(config.work_dir)
    estimator=estimator if estimator is not None else create_depth_estimator(config,config.viewpoint_backend)
    started=time.perf_counter()
    observation = estimator.estimate(frame,output/'reference_depth.npy')
    observation=observation.model_copy(update={'metadata':{
        **observation.metadata,'depth_node_estimation_seconds':time.perf_counter()-started}})
    (output/'reference_depth.json').write_text(observation.model_dump_json(indent=2),encoding='utf-8')
    depth = load_depth(observation,(frame.width,frame.height))
    if config.debug or config.depth_debug:
        save_visualization(depth,output/'reference_depth_visualization.png')
        if observation.metric:np.save(output/'depth_metric.npy',depth,allow_pickle=False)
    return {'reference_depth':observation}
