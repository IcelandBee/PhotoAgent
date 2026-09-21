"""Read-only audit of a completed batch, including lossless comparison panels."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def verify(output):
    output = Path(output)
    summary = json.loads((output / 'experiment_summary.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'completed', summary['status']
    assert summary['depth_estimate_calls'] == summary['input_image_count']
    assert len(summary['cases']) == summary['input_image_count'] * summary['preset_count']
    assert summary['failed_case_count'] == 0
    stats = {}
    implementations = {}
    for info in summary['images']:
        directory = output / info['directory']
        assert info['depth_estimate_calls'] == 1
        with Image.open(directory / 'source.png') as image:
            source = np.array(image)
        height, width = source.shape[:2]
        for filename in ('all_results_grid.jpg', 'all_comparisons_grid.jpg'):
            with Image.open(directory / 'summary' / filename) as image:
                image.verify()
        records = [r for r in summary['cases'] if r['source'] == info['source']]
        assert len(records) == summary['preset_count']
        for record in records:
            path = output / record['result_path']
            meta = json.loads(path.with_name('metadata.json').read_text(encoding='utf-8'))
            with Image.open(path) as image:
                result = np.array(image)
            with Image.open(path.with_name('valid_mask.png')) as image:
                mask_bytes = np.array(image)
            assert set(np.unique(mask_bytes)).issubset({0, 255})
            valid = mask_bytes > 0
            assert result.shape == source.shape and valid.shape == source.shape[:2]
            assert abs(float(valid.mean()) - record['valid_pixel_ratio']) < 1e-12
            assert abs(1 - float(valid.mean()) - record['hole_pixel_ratio']) < 1e-12
            assert np.all(result[~valid] == meta['blank_color'])
            assert meta['depth_sha256'] == info['depth_sha256']
            assert meta['renderer'] == 'point_cloud' and any(meta['translation'].values())
            ks, kt = np.array(meta['source_intrinsics']), np.array(meta['target_intrinsics'])
            assert np.isclose(kt[0, 0] / ks[0, 0], meta['focal_scale'])
            assert np.isclose(kt[1, 1] / ks[1, 1], meta['focal_scale'])
            with Image.open(output / record['compare_path']) as image:
                panels = np.array(image)
            assert panels.shape[:2] == (height + 160, 2 * width)
            np.testing.assert_array_equal(panels[40:40+height, :width], source)
            np.testing.assert_array_equal(panels[40:40+height, width:], result)
            stats.setdefault(record['preset'], []).append(float(valid.mean()))
            impl = meta['renderer_debug']['implementation']
            implementations[impl] = implementations.get(impl, 0) + 1
    return {
        'verified_images': len(summary['images']), 'verified_cases': len(summary['cases']),
        'depth_estimate_calls': summary['depth_estimate_calls'], 'implementations': implementations,
        'all_panels_pixel_exact': True, 'all_invalid_pixels_constant_fill': True,
        'per_preset': {name: {'success': len(values), 'valid_mean': float(np.mean(values)),
                              'valid_min': min(values), 'valid_max': max(values),
                              'hole_mean': 1-float(np.mean(values))} for name, values in stats.items()},
        'identity_projection_max_error_px': max(i.get('identity', {}).get('projection_max_error_px', 0) for i in summary['images']),
        'identity_rgb_mae_range': [min(i['identity']['rgb_mae_on_valid'] for i in summary['images']),
                                   max(i['identity']['rgb_mae_on_valid'] for i in summary['images'])]
                                  if all('identity' in i for i in summary['images']) else None,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir), indent=2))
