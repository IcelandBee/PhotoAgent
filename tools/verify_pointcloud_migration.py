"""Compare production rendering against a completed batch using its saved depth."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from photography_viewpoint_agent.config.settings import AgentConfig
from photography_viewpoint_agent.schemas.depth import DepthObservation
from photography_viewpoint_agent.schemas.target import TargetState
from photography_viewpoint_agent.schemas.video import FrameInfo
from photography_viewpoint_agent.renderer.target_sketch import TargetSketchRenderer
from photography_viewpoint_agent.tools.sketch_validator import validate_sketch


def verify(batch_dir, output_dir):
    batch, output = Path(batch_dir).resolve(), Path(output_dir).resolve()
    summary = json.loads((batch / 'experiment_summary.json').read_text(encoding='utf-8'))
    if summary['status'] != 'completed':
        raise ValueError('Expected a completed batch')
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for info in summary['images']:
        directory = batch / info['directory']
        source = directory / 'source.png'
        with Image.open(source) as image:
            width, height = image.size
        frame = FrameInfo(frame_id=directory.name, path=str(source), frame_index=0,
                          timestamp=0, width=width, height=height)
        observation = DepthObservation.model_validate_json((directory / 'depth/depth.json').read_text(encoding='utf-8'))
        observation = observation.model_copy(update={'depth_path': str(directory / 'depth/depth.npy')})
        config = AgentConfig(target_width=width, target_height=height,
            fill_color=summary['config']['fill_color'], splat_radius=summary['config']['splat_radius'])
        renderer = TargetSketchRenderer(width, height, fill_color=config.fill_color, splat_radius=config.splat_radius)
        for case in (c for c in summary['cases'] if c['source'] == info['source']):
            preset = summary['presets'][case['preset']]
            target = TargetState.model_validate({'viewpoint': {'mode': 'camera', **preset['rotation'],
                **{'translation_' + k: v for k, v in preset['translation'].items()}},
                'framing': {'focal_scale': preset['focal_scale']}})
            path, meta = renderer.render(frame, None, target,
                output / directory.name / case['preset'] / 'target_sketch.png', reference_depth=observation)
            check = validate_sketch(target, meta, path, config)
            with Image.open(path) as image, Image.open(batch / case['result_path']) as expected:
                delta = np.abs(np.asarray(image).astype(float) - np.asarray(expected))
            with Image.open(meta.valid_mask_path) as mask, Image.open((batch / case['result_path']).with_name('valid_mask.png')) as expected:
                mask_equal = np.array_equal(np.asarray(mask), np.asarray(expected))
            record = {'source': info['source'], 'preset': case['preset'],
                'validation_passed': check.passed, 'rgb_mae': float(delta.mean()),
                'rgb_max_difference': float(delta.max()), 'mask_equal': mask_equal,
                'valid_pixel_ratio': meta.valid_pixel_ratio, 'hole_pixel_ratio': meta.hole_pixel_ratio,
                'result': path}
            records.append(record)
            print(f"{case['preset']}: RGB max diff={delta.max():g}, mask equal={mask_equal}, validation={check.passed}", flush=True)
    passed = all(r['validation_passed'] and r['rgb_max_difference'] == 0 and r['mask_equal'] for r in records)
    report = {'passed': passed, 'cases': records, 'source_batch': str(batch),
              'depth_policy': 'saved batch depth reused; no inference in this comparison'}
    (output / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    raise SystemExit(0 if verify(args.batch_dir, args.output_dir) else 1)
