"""图像匹配证据；不生成目标图。"""
import cv2
import numpy as np
from PIL import Image, ImageOps


def read_image(path):
    with Image.open(path) as image:
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")


def gray(image):
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)


def alignment(reference, scene):
    """返回参考图到场景图的单应变换及内点比例。"""
    detector = cv2.ORB_create(nfeatures=2500)
    k1, d1 = detector.detectAndCompute(gray(reference), None)
    k2, d2 = detector.detectAndCompute(gray(scene), None)
    if d1 is None or d2 is None or len(d2) < 2:
        return None, 0.0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    matches = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.7 * p[1].distance]
    if len(matches) < 12:
        return None, 0.0
    src = np.float32([k1[m.queryIdx].pt for m in matches])
    dst = np.float32([k2[m.trainIdx].pt for m in matches])
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    ratio = float(mask.mean()) if mask is not None else 0.0
    if matrix is None or not np.isfinite(matrix).all() or ratio < 0.6 or int(mask.sum()) < 10:
        return None, ratio
    return matrix, ratio


def corners(image, matrix):
    w, h = image.size
    return cv2.perspectiveTransform(np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]), matrix)[0]


def compare(current, target):
    a = np.asarray(current.resize((256, 256))).astype(float)
    b = np.asarray(target.resize((256, 256))).astype(float)
    error = float(np.abs(a - b).mean() / 255)
    if current.size == target.size and error < 0.015:
        return True, 0.98, {"mean_pixel_error": error, "method": "pixel_similarity"}
    matrix, ratio = alignment(target, current)
    if matrix is None:
        return False, 0.25, {"mean_pixel_error": error, "inlier_ratio": ratio, "method": "insufficient_matches"}
    points = corners(target, matrix)
    w, h = current.size
    expected = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    displacement = float(np.linalg.norm((points - expected) / [w, h], axis=1).mean())
    return displacement < 0.07, min(0.95, ratio), {"inlier_ratio": ratio, "corner_displacement": displacement, "method": "homography"}
