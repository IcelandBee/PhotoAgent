"""Research only: direct R+t warp, never a TargetState/Workflow entrypoint."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image
from photography_viewpoint_agent.renderer.camera_parameters import CameraWarpParameters
from photography_viewpoint_agent.renderer.point_cloud_warp import PointCloudViewpointWarper
from photography_viewpoint_agent.renderer.mesh_warp import MeshViewpointWarper, camera_intrinsics
from photography_viewpoint_agent.depth_estimation.storage import load_depth
from photography_viewpoint_agent.schemas.depth import DepthObservation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--depth', required=True, type=Path)
    parser.add_argument('--parameters', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--mesh-device', default='cuda')
    args = parser.parse_args(argv)
    data = json.loads(args.parameters.read_text(encoding='utf-8-sig'))
    # Archived examples contain old framing/subject fields. This utility renders
    # ONLY their low-level camera parameters, never a production composition.
    parameters = CameraWarpParameters.model_validate(data.get('viewpoint', data))
    if parameters.mode not in ('depth_3d', 'depth_mesh'):
        raise ValueError('Research translation requires a point or mesh backend')
    with Image.open(args.reference) as image:
        source = image.convert('RGB')
    sidecar = args.depth.with_suffix('.json')
    if sidecar.exists():
        observation = DepthObservation.model_validate_json(sidecar.read_text(encoding='utf-8-sig'))
        observation = observation.model_copy(update={'depth_path': str(args.depth.resolve())})
    else:
        observation = DepthObservation(depth_path=str(args.depth.resolve()), width=source.width, height=source.height)
    depth = load_depth(observation, source.size)
    scene = float(np.median(depth[depth > 0]))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if parameters.mode == 'depth_mesh':
        result, meta, valid = MeshViewpointWarper(device=args.mesh_device).warp(
            source, depth, parameters, observation, debug_dir=args.output_dir)
    else:
        k, origin = camera_intrinsics(source.size, parameters, observation)
        result, meta, valid = PointCloudViewpointWarper().warp(
            source, depth / scene if observation.metric else depth, parameters,
            focal_length_px=float(k[0, 0]) if origin == 'depth_pro' else None)
    result.save(args.output_dir / 'research_warp.png')
    Image.fromarray(valid.astype(np.uint8) * 255).save(args.output_dir / 'valid_mask.png')
    (args.output_dir / 'research_metadata.json').write_text(
        json.dumps({'research_only': True, 'warp': meta}, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
