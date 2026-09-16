"""Pure rotation: x right, y down, z forward; source rays -> target rays.

Positive camera yaw right / pitch up move the scene left / down.
Positive camera roll clockwise (looking along +z) moves the scene CCW.
R = Rz(-roll) @ Rx(-pitch) @ Ry(-yaw), applied right-to-left.
This defines fixed sequential ray rotations, not interchangeable Euler orders.
"""
import cv2
import numpy as np
from PIL import Image
from photography_viewpoint_agent.schemas.target import ViewpointTarget


def rotation_homography(size, viewpoint):
    width, height = size
    if min(size) <= 0:
        raise ValueError('Image dimensions must be positive')
    yaw, pitch, roll = -np.deg2rad([viewpoint.yaw_deg, viewpoint.pitch_deg, viewpoint.roll_deg])
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    rx = np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]])
    rz = np.array([[cr,-sr,0],[sr,cr,0],[0,0,1]])
    rotation = rz @ rx @ ry
    focal = (width/2)/np.tan(np.deg2rad(viewpoint.horizontal_fov_deg)/2)
    intrinsic = np.array([[focal,0,width/2],[0,focal,height/2],[0,0,1]])
    homography = intrinsic @ rotation @ np.linalg.inv(intrinsic)
    corners = np.array([[0,0,1],[width,0,1],[width,height,1],[0,height,1]]).T
    # Refuse rotations whose forward/inverse projections cross the camera plane.
    for matrix in (homography, np.linalg.inv(homography)):
        if np.min((matrix @ corners)[2]) <= 1e-6:
            raise ValueError('Rotation/FoV crosses the camera plane; reduce angles or FoV')
    return homography, intrinsic, rotation


def warp_bbox(box, size, homography):
    """Axis-aligned bounds of the projected four corners, intentionally unclipped."""
    width, height = size
    left, top, right, bottom = box
    points = np.array([[left*width,top*height,1],[right*width,top*height,1],
                       [right*width,bottom*height,1],[left*width,bottom*height,1]]).T
    projected = homography @ points
    xy = projected[:2]/projected[2]
    return (float(xy[0].min()/width),float(xy[1].min()/height),
            float(xy[0].max()/width),float(xy[1].max()/height))


class RotationViewpointWarper:
    def warp(self, image, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
             horizontal_fov_deg=60.0, border_mode='constant', fill_color=(128,128,128)):
        viewpoint = ViewpointTarget(mode='rotation',yaw_deg=yaw_deg,pitch_deg=pitch_deg,
            roll_deg=roll_deg,horizontal_fov_deg=horizontal_fov_deg,border_mode=border_mode)
        h, k, r = rotation_homography(image.size, viewpoint)
        source = image.convert('RGB')
        if not viewpoint.active:
            result = source.copy()  # Exact identity: no interpolation.
        else:
            result = Image.fromarray(cv2.warpPerspective(np.asarray(source),h,image.size,
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT if border_mode=='constant'
                else cv2.BORDER_REPLICATE,borderValue=tuple(fill_color)))
        valid = cv2.warpPerspective(np.full((image.height,image.width),255,np.uint8),h,image.size,
            flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        return result, {'homography':h.tolist(),'intrinsics':k.tolist(),'rotation':r.tolist(),
            'parameters':viewpoint.model_dump(),'valid_fraction':float(np.mean(valid>0)),
            'image_size':list(image.size)}
