"""Explicit-intrinsics point-cloud rendering with depth-aware bilinear splats."""
from dataclasses import dataclass
import numpy as np
from PIL import Image
from .geometry import transform_points
from .result import CameraRenderResult


@dataclass(frozen=True)
class PreparedPointCloud:
    ids: np.ndarray
    vertices: np.ndarray
    colors: np.ndarray
    source_intrinsics: object


def prepare_point_cloud(image, depth, intrinsics):
    ks = intrinsics
    depth = np.asarray(depth)
    if image.size != ks.size or depth.shape != (image.height, image.width):
        raise ValueError('Source RGB, depth and intrinsics dimensions must match')
    ids = np.flatnonzero(np.isfinite(depth) & (depth > 0))
    if not len(ids):
        raise ValueError('No finite positive depth')
    z = depth.ravel()[ids]
    x, y = ids % ks.image_width, ids // ks.image_width
    vertices = np.column_stack(((x - ks.cx) * z / ks.fx, (y - ks.cy) * z / ks.fy, z))
    colors = np.asarray(image.convert('RGB')).reshape(-1, 3).copy()
    for array in (ids, vertices, colors):
        array.setflags(write=False)
    return PreparedPointCloud(ids, vertices, colors, ks)


def project_cloud(cloud, state):
    if cloud.source_intrinsics != state.source_intrinsics:
        raise ValueError('Prepared cloud belongs to different source intrinsics')
    target, uv, r, center, _ = transform_points(
        cloud.vertices, state.target_intrinsics.matrix(), state.warp_parameters())
    good = np.isfinite(target).all(axis=1) & (target[:, 2] > 1e-6)
    return cloud.ids[good], uv[good].T, target[good, 2], r, center


def project_explicit_intrinsics(depth, state):
    image = Image.new('RGB', state.source_intrinsics.size)
    return project_cloud(prepare_point_cloud(image, depth, state.source_intrinsics), state)


def bilinear_footprint(u, v):
    """Four target pixels and interpolation weights for each projected point."""
    x0, y0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    dx, dy = u - x0, v - y0
    return (
        (x0, y0, (1.0 - dx) * (1.0 - dy)),
        (x0 + 1, y0, dx * (1.0 - dy)),
        (x0, y0 + 1, (1.0 - dx) * dy),
        (x0 + 1, y0 + 1, dx * dy),
    )


def rasterize_bilinear(cloud, ids, uv, z, size, fill_color, depth_rel_tol=0.01):
    """Nearest-depth gate followed by weighted color accumulation.

    This is the bilinear research rasterizer promoted unchanged into production.
    Projection and point-cloud construction remain shared by both renderers.
    """
    width, height = size
    finite = np.isfinite(uv).all(axis=0)
    ids, uv, z = ids[finite], uv[:, finite], z[finite]
    footprint = bilinear_footprint(uv[0], uv[1])
    pixel_count = width * height
    zbuffer = np.full(pixel_count, np.inf)
    prepared = []
    candidate_count = 0
    for x, y, weight in footprint:
        inside = ((x >= 0) & (x < width) & (y >= 0) & (y < height)
                  & np.isfinite(weight) & (weight > 0))
        point = np.flatnonzero(inside)
        dest = y[inside] * width + x[inside]
        local_weight = weight[inside]
        candidate_count += len(dest)
        np.minimum.at(zbuffer, dest, z[point])
        prepared.append((point, dest, local_weight))

    weight_sum = np.zeros(pixel_count, np.float64)
    rgb_sum = np.zeros((3, pixel_count), np.float64)
    accepted_count = 0
    for point, dest, weight in prepared:
        tolerance = np.maximum(1e-6, depth_rel_tol * zbuffer[dest])
        accept = np.abs(z[point] - zbuffer[dest]) <= tolerance
        if not np.any(accept):
            continue
        point, dest, weight = point[accept], dest[accept], weight[accept]
        accepted_count += len(dest)
        weight_sum += np.bincount(dest, weights=weight, minlength=pixel_count)
        colors = cloud.colors[ids[point]].astype(np.float64, copy=False)
        for channel in range(3):
            rgb_sum[channel] += np.bincount(
                dest, weights=weight * colors[:, channel], minlength=pixel_count)

    valid_flat = weight_sum > 0
    result = np.full((pixel_count, 3), fill_color, np.uint8)
    blended = np.rint(rgb_sum[:, valid_flat] / weight_sum[valid_flat]).T
    result[valid_flat] = np.clip(blended, 0, 255).astype(np.uint8)
    valid = valid_flat.reshape(height, width)
    projected = np.where(valid, zbuffer.reshape(height, width), 0).astype(np.float32)
    valid_count = int(valid_flat.sum())
    return Image.fromarray(result.reshape(height, width, 3)), valid, projected, {
        'candidate_count': candidate_count,
        'accepted_candidate_count': accepted_count,
        'average_candidates_per_valid_pixel': candidate_count / max(valid_count, 1),
        'average_accepted_candidates_per_valid_pixel': accepted_count / max(valid_count, 1),
    }


def splat_explicit_intrinsics(image, depth, state, radius, fill_color, *, geometry=None):
    cloud = geometry if geometry is not None else prepare_point_cloud(image, depth, state.source_intrinsics)
    ids, uv, z, _, _ = project_cloud(cloud, state)
    width, height = state.target_intrinsics.size
    possible = (np.isfinite(uv).all(axis=0) & (uv[0] >= -radius - 1)
                & (uv[0] < width + radius + 1) & (uv[1] >= -radius - 1)
                & (uv[1] < height + radius + 1))
    ids, uv, z = ids[possible], uv[:, possible], z[possible]
    base = np.rint(uv).astype(np.int64)
    zbuffer = np.full(width * height, np.inf)
    score = np.full(width * height, np.inf)
    result = np.full((width * height, 3), fill_color, np.uint8)
    colors = cloud.colors

    def splats():
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = base[0] + dx, base[1] + dy
                inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
                point = np.flatnonzero(inside)
                dest = y[inside] * width + x[inside]
                cost = ((x[inside] - uv[0, inside]) ** 2 + (y[inside] - uv[1, inside]) ** 2
                        + ids[inside] / (image.width * image.height) * 1e-8)
                yield point, dest, cost

    # Compute the candidate footprint once; retain the baseline's three passes.
    candidates = list(splats())
    for point, dest, cost in candidates:
        np.minimum.at(zbuffer, dest, z[point])
    for point, dest, cost in candidates:
        nearest = z[point] == zbuffer[dest]
        np.minimum.at(score, dest[nearest], cost[nearest])
    for point, dest, cost in candidates:
        winner = (z[point] == zbuffer[dest]) & (cost == score[dest])
        result[dest[winner]] = colors[ids[point[winner]]]
    valid = np.isfinite(zbuffer).reshape(height, width)
    projected_depth = np.where(valid, zbuffer.reshape(height, width), 0).astype(np.float32)
    return Image.fromarray(result.reshape(height, width, 3)), valid, projected_depth


class DepthRenderer:
    def __init__(self, splat_radius=1, *, renderer='bilinear'):
        if type(splat_radius) is not int or not 0 <= splat_radius <= 3:
            raise ValueError('splat_radius must be an integer in [0, 3]')
        if renderer not in ('bilinear', 'nearest_z'):
            raise ValueError(f'Unsupported renderer: {renderer}. Expected one of: bilinear, nearest_z.')
        self.radius = splat_radius
        self.renderer = renderer

    def render(self, image, state, depth, fill_color=(128, 128, 128), *, geometry=None):
        # All poses, including identity and pure focal/rotation changes, project points.
        if self.renderer == 'bilinear':
            cloud = geometry if geometry is not None else prepare_point_cloud(
                image, depth, state.source_intrinsics)
            ids, uv, z, _, _ = project_cloud(cloud, state)
            result, valid, projected, stats = rasterize_bilinear(
                cloud, ids, uv, z, state.target_intrinsics.size, fill_color)
            implementation = 'depth_aware_bilinear_point_splat'
        else:
            result, valid, projected = splat_explicit_intrinsics(
                image, depth, state, self.radius, fill_color, geometry=geometry)
            stats = {}
            implementation = 'explicit_intrinsics_point_splat_adapter'
        return CameraRenderResult(result, valid, 'depth_3d', state,
            {'depth_used': True, 'implementation': implementation,
             'point_renderer': self.renderer, 'depth_rel_tol': 0.01 if self.renderer == 'bilinear' else None,
             'splat_radius': self.radius if self.renderer == 'nearest_z' else None,
             'geometry_reused': geometry is not None,
             'identity_copy_fast_path': False, **stats}, {'projected_depth': projected})
