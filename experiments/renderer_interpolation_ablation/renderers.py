"""Research comparisons around the production point-cloud renderer.

Gaussian remains research-only. Bilinear uses the promoted production
rasterizer so experiments and the workflow cannot silently diverge.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from photography_viewpoint_agent.camera_rendering.depth_renderer import (
    bilinear_footprint,
    prepare_point_cloud,
    project_cloud,
    rasterize_bilinear,
)
from photography_viewpoint_agent.camera_rendering.result import CameraRenderResult


@dataclass(frozen=True)
class RendererConfig:
    depth_rel_tol: float = 0.01
    gaussian_radius: int = 1
    gaussian_sigma: float = 0.75

    def __post_init__(self):
        if not 0 <= self.depth_rel_tol <= 1:
            raise ValueError("depth_rel_tol must be in [0, 1]")
        if type(self.gaussian_radius) is not int or self.gaussian_radius < 0:
            raise ValueError("gaussian_radius must be a nonnegative integer")
        if not np.isfinite(self.gaussian_sigma) or self.gaussian_sigma <= 0:
            raise ValueError("gaussian_sigma must be finite and positive")


def gaussian_footprint(u, v, radius=1, sigma=0.75):
    """Return a fixed, positive Gaussian footprint around rounded centers."""
    base_x, base_y = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
    footprint = []
    scale = 2.0 * sigma * sigma
    for oy in range(-radius, radius + 1):
        for ox in range(-radius, radius + 1):
            x, y = base_x + ox, base_y + oy
            weight = np.exp(-((x - u) ** 2 + (y - v) ** 2) / scale)
            footprint.append((x, y, weight))
    return tuple(footprint)


class ResearchPointRenderer:
    names = ("nearest_z", "bilinear", "gaussian")

    def __init__(self, name, config=None):
        if name not in self.names:
            raise ValueError(f"Unknown renderer: {name}")
        self.name = name
        self.config = config or RendererConfig()

    def render(self, image, state, depth, fill_color=(128, 128, 128), *, geometry=None):
        cloud = geometry if geometry is not None else prepare_point_cloud(
            image, depth, state.source_intrinsics)
        project_started = time.perf_counter()
        ids, uv, z, _, _ = project_cloud(cloud, state)
        project_seconds = time.perf_counter() - project_started
        raster_started = time.perf_counter()
        if self.name == "nearest_z":
            output, valid, projected, stats = self._nearest(
                image, cloud, ids, uv, z, state.target_intrinsics.size, fill_color)
        else:
            output, valid, projected, stats = self._weighted(
                cloud, ids, uv, z, state.target_intrinsics.size, fill_color)
        raster_seconds = time.perf_counter() - raster_started
        metadata = {
            "implementation": f"research_{self.name}_point_splat",
            "renderer": self.name,
            "depth_used": True,
            "geometry_reused": geometry is not None,
            "project_seconds": project_seconds,
            "rasterization_seconds": raster_seconds,
            "depth_rel_tol": self.config.depth_rel_tol if self.name != "nearest_z" else None,
            "splat_radius": 1 if self.name == "nearest_z" else (
                self.config.gaussian_radius if self.name == "gaussian" else None),
            "gaussian_sigma": self.config.gaussian_sigma if self.name == "gaussian" else None,
            **stats,
        }
        return CameraRenderResult(
            output, valid, "depth_3d", state, metadata, {"projected_depth": projected})

    @staticmethod
    def _nearest(image, cloud, ids, uv, z, size, fill_color):
        """Exact radius-one production policy, kept local for split timing."""
        width, height = size
        radius = 1
        possible = (
            np.isfinite(uv).all(axis=0)
            & (uv[0] >= -radius - 1)
            & (uv[0] < width + radius + 1)
            & (uv[1] >= -radius - 1)
            & (uv[1] < height + radius + 1)
        )
        ids, uv, z = ids[possible], uv[:, possible], z[possible]
        base = np.rint(uv).astype(np.int64)
        zbuffer = np.full(width * height, np.inf)
        score = np.full(width * height, np.inf)
        result = np.full((width * height, 3), fill_color, np.uint8)
        candidate_count = 0

        def splats():
            nonlocal candidate_count
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    x, y = base[0] + dx, base[1] + dy
                    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
                    point = np.flatnonzero(inside)
                    dest = y[inside] * width + x[inside]
                    cost = (
                        (x[inside] - uv[0, inside]) ** 2
                        + (y[inside] - uv[1, inside]) ** 2
                        + ids[inside] / (image.width * image.height) * 1e-8
                    )
                    yield point, dest, cost

        first = list(splats())
        candidate_count = sum(len(dest) for _, dest, _ in first)
        for point, dest, _ in first:
            np.minimum.at(zbuffer, dest, z[point])
        for point, dest, cost in first:
            nearest = z[point] == zbuffer[dest]
            np.minimum.at(score, dest[nearest], cost[nearest])
        for point, dest, cost in first:
            winner = (z[point] == zbuffer[dest]) & (cost == score[dest])
            result[dest[winner]] = cloud.colors[ids[point[winner]]]
        valid = np.isfinite(zbuffer).reshape(height, width)
        projected = np.where(valid, zbuffer.reshape(height, width), 0).astype(np.float32)
        valid_count = int(valid.sum())
        return Image.fromarray(result.reshape(height, width, 3)), valid, projected, {
            "candidate_count": candidate_count,
            "accepted_candidate_count": valid_count,
            "average_candidates_per_valid_pixel": candidate_count / max(valid_count, 1),
        }

    def _weighted(self, cloud, ids, uv, z, size, fill_color):
        if self.name == 'bilinear':
            return rasterize_bilinear(
                cloud, ids, uv, z, size, fill_color, self.config.depth_rel_tol)
        width, height = size
        finite = np.isfinite(uv).all(axis=0)
        ids, uv, z = ids[finite], uv[:, finite], z[finite]
        footprint = gaussian_footprint(
            uv[0], uv[1], self.config.gaussian_radius, self.config.gaussian_sigma)
        pixel_count = width * height
        zbuffer = np.full(pixel_count, np.inf)
        prepared = []
        candidate_count = 0
        # Pass one: exact nearest target-surface depth for every footprint pixel.
        for x, y, weight in footprint:
            inside = (
                (x >= 0) & (x < width) & (y >= 0) & (y < height)
                & np.isfinite(weight) & (weight > 0)
            )
            point = np.flatnonzero(inside)
            dest = y[inside] * width + x[inside]
            local_weight = weight[inside]
            candidate_count += len(dest)
            np.minimum.at(zbuffer, dest, z[point])
            prepared.append((point, dest, local_weight))

        weight_sum = np.zeros(pixel_count, np.float64)
        rgb_sum = np.zeros((3, pixel_count), np.float64)
        accepted_count = 0
        # Pass two: accumulate only samples belonging to the nearest depth surface.
        for point, dest, weight in prepared:
            tolerance = np.maximum(1e-6, self.config.depth_rel_tol * zbuffer[dest])
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
        # Production converts assigned numeric colors to uint8.  Round weighted
        # estimates first so interpolation is unbiased around half values.
        blended = np.rint(rgb_sum[:, valid_flat] / weight_sum[valid_flat]).T
        result[valid_flat] = np.clip(blended, 0, 255).astype(np.uint8)
        valid = valid_flat.reshape(height, width)
        projected = np.where(valid, zbuffer.reshape(height, width), 0).astype(np.float32)
        valid_count = int(valid_flat.sum())
        return Image.fromarray(result.reshape(height, width, 3)), valid, projected, {
            "candidate_count": candidate_count,
            "accepted_candidate_count": accepted_count,
            "average_candidates_per_valid_pixel": candidate_count / max(valid_count, 1),
            "average_accepted_candidates_per_valid_pixel": accepted_count / max(valid_count, 1),
        }


def create_renderers(config=None):
    config = config or RendererConfig()
    return {name: ResearchPointRenderer(name, config) for name in ResearchPointRenderer.names}
