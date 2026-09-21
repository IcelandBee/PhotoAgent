"""Research-only single-RGB batch. Always depth + point splats; no dispatch/subjects."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from photography_viewpoint_agent.camera_rendering.camera_state import CameraState
from photography_viewpoint_agent.camera_rendering.intrinsics import CameraIntrinsics
from photography_viewpoint_agent.camera_rendering.depth_renderer import (
    DepthRenderer, project_explicit_intrinsics, splat_explicit_intrinsics, prepare_point_cloud, project_cloud,
)
from photography_viewpoint_agent.config.settings import AgentConfig
from photography_viewpoint_agent.depth_estimation.factory import create_depth_estimator, resolve_backend
from photography_viewpoint_agent.depth_estimation.storage import load_depth, save_visualization
from photography_viewpoint_agent.schemas.video import FrameInfo

DEFAULT_PRESETS = Path(__file__).with_name('presets.json')
GROUPS = ('crop', 'expand', 'viewpoint')
CONVENTION = {
    'translation': '+x right, +y up, +z forward; source-camera axes',
    'rotation': '+yaw right, +pitch up, +roll clockwise looking along +z',
    'units': 'translation / scene median depth, not meters; rotation in degrees',
    'transform': 'X_target = R @ (X_source - C); C=(tx,-ty,tz)',
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def camera_state(k, preset):
    target = {**k.model_dump(), 'fx': k.fx * preset['focal_scale'], 'fy': k.fy * preset['focal_scale']}
    return CameraState(source_intrinsics=k, target_intrinsics=target,
                       translation=preset['translation'], rotation=preset['rotation'])


def load_presets(path):
    presets = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(presets, dict) or not presets:
        raise ValueError('Presets must be a nonempty mapping')
    k = CameraIntrinsics.from_horizontal_fov(64, 48)
    for name, preset in presets.items():
        if not re.fullmatch(r'[a-z][a-z0-9_]*', name):
            raise ValueError(f'Unsafe preset name: {name}')
        if set(preset) != {'group', 'semantic_direction', 'translation', 'rotation', 'focal_scale'}:
            raise ValueError(f'{name}: unexpected or missing preset fields')
        if preset['group'] not in GROUPS or not isinstance(preset['semantic_direction'], str):
            raise ValueError(f'{name}: invalid group/direction')
        if not math.isfinite(preset['focal_scale']) or preset['focal_scale'] <= 0:
            raise ValueError(f'{name}: focal_scale must be finite and positive')
        if camera_state(k, preset).translation.norm == 0:
            raise ValueError(f'{name}: experiment translation must be nonzero')
    return presets


def comparison(source, result, name, preset):
    width, height = source.size
    if result.size != source.size:
        raise ValueError('Result dimensions differ from source')
    sheet = Image.new('RGB', (width * 2, height + 160), 'white')
    sheet.paste(source, (0, 40))
    sheet.paste(result, (width, 40))
    draw = ImageDraw.Draw(sheet)
    for x, title in ((10, 'SOURCE'), (width + 10, 'RESULT')):
        draw.text((x, 8), title, fill='black', font_size=22)
    t, r = preset['translation'], preset['rotation']
    lines = [name, f't(x,y,z)=({t.get("x",0):g}, {t.get("y",0):g}, {t.get("z",0):g})',
             f'yaw={r.get("yaw_deg",0):g}  pitch={r.get("pitch_deg",0):g}  roll={r.get("roll_deg",0):g}  focal_scale={preset["focal_scale"]:g}']
    for i, line in enumerate(lines):
        draw.text((10, height + 50 + i * 30), line, fill='black', font_size=21)
    return sheet


def grids(source, directory, presets, records):
    """Display-only thumbnails; native results/compares are never resized."""
    for paired, filename in ((False, 'all_results_grid.jpg'), (True, 'all_comparisons_grid.jpg')):
        cell_w, cell_h = (620, 490) if paired else (380, 560)
        rows = 1 + sum(math.ceil(sum(p['group'] == g for p in presets.values()) / 4) for g in GROUPS)
        sheet = Image.new('RGB', (4 * cell_w, rows * (cell_h + 40)), '#eeeeee')
        draw = ImageDraw.Draw(sheet)
        thumb = ImageOps.contain(source, (cell_w - 16, cell_h - 48))
        sheet.paste(thumb, (8, 40))
        draw.text((8, 8), 'SOURCE', fill='black', font_size=24)
        row = 1
        for group in GROUPS:
            names = [name for name, p in presets.items() if p['group'] == group]
            for index, name in enumerate(names):
                x, y = index % 4 * cell_w, (row + index // 4) * (cell_h + 40)
                if index % 4 == 0:
                    draw.text((8, y), group.upper(), fill='black', font_size=24)
                record = records[name]
                if record['status'] == 'success':
                    path = record['compare_path' if paired else 'result_path']
                    with Image.open(directory.parent / path) as image:
                        thumb = ImageOps.contain(image, (cell_w - 16, cell_h - 48))
                        sheet.paste(thumb, (x + 8, y + 40))
                else:
                    draw.text((x + 8, y + 80), 'FAILED (see metadata)', fill='red', font_size=20)
                draw.text((x + 8, y + cell_h), name, fill='black', font_size=18)
            row += math.ceil(len(names) / 4)
        sheet.save(directory / 'summary' / filename, quality=94, subsampling=0)


def identity_check(source, depth, k, directory, blank, geometry=None):
    state = CameraState(source_intrinsics=k, target_intrinsics=k)
    geometry = geometry if geometry is not None else prepare_point_cloud(source, depth, k)
    image, valid, _ = splat_explicit_intrinsics(source, depth, state, 1, blank, geometry=geometry)
    ids, uv, *_ = project_cloud(geometry, state)
    expected = np.stack((ids % source.width, ids // source.width))
    error = float(np.max(np.abs(uv - expected)))
    if error > 1e-4:
        raise ValueError(f'Identity projection error: {error}')
    mae = float(np.abs(np.asarray(image).astype(float)[valid] - np.asarray(source)[valid]).mean())
    directory.mkdir()
    image.save(directory / 'identity_render.png')
    Image.fromarray(valid.astype(np.uint8) * 255).save(directory / 'valid_mask.png')
    comparison(source, image, 'debug_identity', {'translation': {}, 'rotation': {}, 'focal_scale': 1}).save(directory / 'compare.png')
    record = {'projection_max_error_px': error, 'rgb_mae_on_valid': mae,
              'valid_pixel_ratio': float(valid.mean()), 'identity_copy_fast_path': False,
              'note': 'Nearest-Z 3x3 splats can choose a nearby shallower pixel even at identity.'}
    write_json(directory / 'metadata.json', record)
    return record


def run_batch(input_dir, output_dir, *, presets_path=DEFAULT_PRESETS, max_images=None,
              config=None, save_depth=True, save_debug=False, depth_estimator=None):
    config = config or AgentConfig(depth_backend='depth_anything')
    presets = load_presets(presets_path)
    input_dir, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if not input_dir.is_dir():
        raise ValueError(f'Input directory not found: {input_dir}')
    if max_images is not None and max_images < 1:
        raise ValueError('max_images must be positive')
    files = sorted((p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in
                    ('.jpg', '.jpeg', '.png', '.webp')), key=lambda p: p.name.casefold())
    if not files:
        raise ValueError('No supported images found')
    selected = files[:max_images] if max_images else files
    output.mkdir(parents=True, exist_ok=False)
    estimator = depth_estimator or create_depth_estimator(config, 'depth_3d')
    renderer = DepthRenderer(config.splat_radius)
    summary = {'research_only': True, 'input_dir': str(input_dir), 'output_dir': str(output),
               'discovered_image_count': len(files), 'input_image_count': len(selected),
               'preset_count': len(presets), 'presets': presets, 'coordinate_convention': CONVENTION,
               'config': config.model_dump(), 'cases': [], 'images': [],
               'depth_estimate_calls': 0, 'status': 'running',
               'geometry_reuse': 'One immutable source point cloud shared by all presets.',
               'intrinsics_policy': 'Explicit source horizontal FOV=60 deg; estimator focal metadata does not override K.',
               'no_subject_edit': True, 'no_post_render_framing': True}
    started = time.perf_counter()

    def checkpoint():
        summary['success_case_count'] = sum(r['status'] == 'success' for r in summary['cases'])
        summary['failed_case_count'] = sum(r['status'] == 'failed' for r in summary['cases'])
        summary['elapsed_seconds'] = time.perf_counter() - started
        write_json(output / 'experiment_summary.json', summary)

    checkpoint()
    for index, path in enumerate(selected, 1):
        slug = re.sub(r'[^\w.-]', '_', path.stem)[:80]
        directory = output / f'image_{index:03d}_{slug}'
        directory.mkdir()
        (directory / 'summary').mkdir()
        info = {'source': str(path), 'directory': directory.name, 'depth_estimate_calls': 0}
        summary['images'].append(info)
        records, source, depth = {}, None, None
        print(f'[{index}/{len(selected)}] {path.name}: depth', flush=True)
        try:
            with Image.open(path) as image:
                source = ImageOps.exif_transpose(image).convert('RGB')
            source.save(directory / 'source.png')
            frame = FrameInfo(frame_id=directory.name, frame_index=0, timestamp=0,
                              path=str(directory / 'source.png'), width=source.width, height=source.height)
            summary['depth_estimate_calls'] += 1
            info['depth_estimate_calls'] += 1
            observation = estimator.estimate(frame, directory / 'depth' / 'depth.npy')
            depth = load_depth(observation, source.size)
            save_visualization(depth, directory / 'depth' / 'depth_vis.png')
            scene = float(np.median(depth[depth > 0]))
            depth = depth / scene if observation.metric else depth
            depth.setflags(write=False)
            digest = hashlib.sha256(depth.tobytes()).hexdigest()
            k = CameraIntrinsics.from_horizontal_fov(*source.size)
            geometry = prepare_point_cloud(source, depth, k)
            info['source_geometry_builds'] = 1
            info.update(depth_sha256=digest, point_count=int(np.count_nonzero(depth > 0)),
                        image_size=list(source.size), depth_model=observation.metadata.get('model'),
                        scene_reference_depth=scene, depth_retained=save_depth)
        except Exception as exc:
            info['error'] = f'{type(exc).__name__}: {exc}'
            depth = None
        if depth is not None and save_debug:
            try:
                info['identity'] = identity_check(source, depth, k, directory / 'debug', config.fill_color, geometry)
            except Exception as exc:
                info['debug_error'] = f'{type(exc).__name__}: {exc}'
        for name, preset in presets.items():
            case_dir = directory / preset['group'] / name
            case_dir.mkdir(parents=True)
            record = {'source': str(path), 'preset': name, 'group': preset['group'],
                      'result_path': None, 'compare_path': None, 'valid_pixel_ratio': None,
                      'hole_pixel_ratio': None, 'status': 'failed', 'error': None}
            try:
                if depth is None:
                    raise RuntimeError(info['error'])
                state = camera_state(k, preset)
                if state.translation.norm == 0:
                    raise ValueError('Nonzero camera translation required')
                result = renderer.render(source, state, depth, config.fill_color, geometry=geometry)
                if result.renderer_type != 'depth_3d' or result.image.size != source.size:
                    raise ValueError('Wrong renderer or output dimensions')
                if result.valid_mask.shape != depth.shape or not result.valid_mask.any():
                    raise ValueError('Invalid/empty coverage')
                if hashlib.sha256(depth.tobytes()).hexdigest() != digest:
                    raise ValueError('Shared depth mutated')
                result.image.save(case_dir / 'result.png')
                Image.fromarray(result.valid_mask.astype(np.uint8) * 255).save(case_dir / 'valid_mask.png')
                comparison(source, result.image, name, preset).save(case_dir / 'compare.png')
                if save_debug:
                    for key, array in result.debug_info.items():
                        np.save(case_dir / f'{key}.npy', array, allow_pickle=False)
                ratio = float(result.valid_mask.mean())
                record.update(status='success', result_path=(case_dir / 'result.png').relative_to(output).as_posix(),
                              compare_path=(case_dir / 'compare.png').relative_to(output).as_posix(),
                              valid_pixel_ratio=ratio, hole_pixel_ratio=1-ratio)
                metadata = {**record, 'semantic_direction': preset['semantic_direction'],
                            'translation': state.translation.model_dump(),
                            'rotation_deg': dict(zip(('yaw', 'pitch', 'roll'),
                                                     (state.rotation.yaw_deg, state.rotation.pitch_deg, state.rotation.roll_deg))),
                            'focal_scale': preset['focal_scale'], 'renderer': 'point_cloud',
                            'depth_backend': resolve_backend(config, 'depth_3d'),
                            'depth_model': observation.metadata.get('model'), 'depth_device': config.depth_device,
                            'depth_sha256': digest, 'scene_reference_depth': scene,
                            'image_size': list(source.size), 'source_intrinsics': k.matrix().tolist(),
                            'target_intrinsics': state.target_intrinsics.matrix().tolist(),
                            'rotation_matrix': state.rotation_matrix().tolist(),
                            'camera_center_source': state.camera_center().tolist(),
                            'blank_color': list(config.fill_color), 'renderer_debug': result.metadata,
                            'coordinate_convention': CONVENTION}
                write_json(case_dir / 'metadata.json', metadata)
                print(f'  {name}: valid={ratio:.4f}', flush=True)
            except Exception as exc:
                record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                write_json(case_dir / 'metadata.json', record)
                print(f'  {name}: FAILED {record["error"]}', flush=True)
            records[name] = record
            summary['cases'].append(record)
            checkpoint()
        try:
            if source is not None:
                grids(source, directory, presets, records)
        except Exception as exc:
            info['grid_error'] = f'{type(exc).__name__}: {exc}'
        if not save_depth:
            for filename in ('depth.npy', 'depth.json', 'depth_vis.png'):
                (directory / 'depth' / filename).unlink(missing_ok=True)
        info['status'] = 'success' if all(r['status'] == 'success' for r in records.values()) and not any(
            key in info for key in ('error', 'grid_error', 'debug_error')) else 'failed'
        checkpoint()
    summary['status'] = 'completed' if all(i['status'] == 'success' for i in summary['images']) else 'completed_with_errors'
    checkpoint()
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-images', type=int)
    parser.add_argument('--presets', type=Path, default=DEFAULT_PRESETS)
    parser.add_argument('--device', default='cpu', help='Depth device; point splatting uses CPU')
    parser.add_argument('--depth-backend', choices=['depth_anything', 'depth_pro', 'auto'], default='depth_anything')
    parser.add_argument('--depth-model')
    parser.add_argument('--blank-color', type=int, nargs=3, default=(128, 128, 128), metavar=('R','G','B'))
    parser.add_argument('--save-depth', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--save-debug', action='store_true')
    args = parser.parse_args(argv)
    config = AgentConfig(depth_backend=args.depth_backend, depth_model=args.depth_model,
                         depth_device=args.device, fill_color=tuple(args.blank_color))
    result = run_batch(args.input_dir, args.output_dir, presets_path=args.presets, max_images=args.max_images,
                       config=config, save_depth=args.save_depth, save_debug=args.save_debug)
    print(f'{result["status"]}: {result["success_case_count"]} succeeded, {result["failed_case_count"]} failed', flush=True)
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
