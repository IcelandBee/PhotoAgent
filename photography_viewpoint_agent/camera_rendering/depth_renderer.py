"""Single explicit-K point renderer, using the experimentally validated Z splats."""
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

    # Same three-pass nearest-surface policy as the preserved baseline.
    for point, dest, cost in splats():
        np.minimum.at(zbuffer, dest, z[point])
    for point, dest, cost in splats():
        nearest = z[point] == zbuffer[dest]
        np.minimum.at(score, dest[nearest], cost[nearest])
    for point, dest, cost in splats():
        winner = (z[point] == zbuffer[dest]) & (cost == score[dest])
        result[dest[winner]] = colors[ids[point[winner]]]
    valid = np.isfinite(zbuffer).reshape(height, width)
    projected_depth = np.where(valid, zbuffer.reshape(height, width), 0).astype(np.float32)
    return Image.fromarray(result.reshape(height, width, 3)), valid, projected_depth


class DepthRenderer:
    def __init__(self, splat_radius=1):
        if type(splat_radius) is not int or not 0 <= splat_radius <= 3:
            raise ValueError('splat_radius must be an integer in [0, 3]')
        self.radius = splat_radius

    def render(self, image, state, depth, fill_color=(128, 128, 128), *, geometry=None):
        # All poses, including identity and pure focal/rotation changes, project points.
        result, valid, projected = splat_explicit_intrinsics(
            image, depth, state, self.radius, fill_color, geometry=geometry)
        return CameraRenderResult(result, valid, 'depth_3d', state,
            {'depth_used': True, 'implementation': 'explicit_intrinsics_point_splat_adapter',
             'splat_radius': self.radius, 'geometry_reused': geometry is not None,
             'identity_copy_fast_path': False}, {'projected_depth': projected})
