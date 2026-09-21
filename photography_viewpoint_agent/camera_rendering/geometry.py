"""Pose math extracted from tested rotation/mesh helpers, without rasterizers."""
import numpy as np


def rotation_matrix(viewpoint):
    yaw, pitch, roll = -np.deg2rad([viewpoint.yaw_deg, viewpoint.pitch_deg, viewpoint.roll_deg])
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return rz @ rx @ ry


def transform_points(vertices, k, viewpoint, scene_reference_depth=1.0):
    movement = np.array([viewpoint.translation_x, viewpoint.translation_y,
                         viewpoint.translation_z]) * scene_reference_depth
    center = movement * np.array([1, -1, 1])
    rotation = rotation_matrix(viewpoint)
    target = (rotation @ (vertices - center).T).T
    projected = (k @ target.T).T
    uv = projected[:, :2] / np.where(np.abs(projected[:, 2:]) > 1e-8, projected[:, 2:], 1e-8)
    return target, uv, rotation, center, movement
