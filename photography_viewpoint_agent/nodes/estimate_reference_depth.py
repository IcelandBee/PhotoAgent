from pathlib import Path
from photography_viewpoint_agent.depth_estimation.monocular import load_depth, save_visualization


def estimate_reference_depth(state, estimator, config):
    frame = state['reference_frame']
    output = Path(config.work_dir)
    observation = estimator.estimate(frame,output/'reference_depth.npy')
    depth = load_depth(observation,(frame.width,frame.height))
    if config.debug or config.depth_debug:
        save_visualization(depth,output/'reference_depth_visualization.png')
    return {'reference_depth':observation}
