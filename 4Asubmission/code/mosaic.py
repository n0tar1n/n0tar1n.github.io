"""From-scratch homography, inverse-warping, and mosaicing utilities.

The project specification forbids high-level homography and warping helpers.
This module intentionally does not use cv2.findHomography, cv2.warpPerspective,
cv2.getPerspectiveTransform, skimage transforms, or scipy interpolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as an RGB uint8 array."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Save an RGB image, clipping floating point values to display range."""
    output = np.clip(image, 0, 255).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(output, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not save image: {path}")


def computeH(source_points: np.ndarray, destination_points: np.ndarray) -> np.ndarray:
    """Return H that maps source (x, y) points to destination points.

    Uses an overdetermined least-squares solve when more than four matches are
    supplied. H[2, 2] is fixed to one, leaving eight unknowns.
    """
    src = np.asarray(source_points, dtype=np.float64)
    dst = np.asarray(destination_points, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 2 or dst.shape != src.shape:
        raise ValueError("source_points and destination_points must both have shape (N, 2).")
    if len(src) < 4:
        raise ValueError("At least four point correspondences are required.")

    rows: list[list[float]] = []
    values: list[float] = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.extend([u, v])

    h, _, rank, _ = np.linalg.lstsq(np.asarray(rows), np.asarray(values), rcond=None)
    if rank < 8:
        raise ValueError("Degenerate correspondences; choose points that are not collinear.")
    return np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])


def transform_points(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to (N, 2) points."""
    pts = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack((pts, np.ones(len(pts))))
    transformed = (np.asarray(H, dtype=np.float64) @ homogeneous.T).T
    return transformed[:, :2] / transformed[:, 2:3]


def image_corners(image: np.ndarray) -> np.ndarray:
    """Return image corners in (x, y) order."""
    height, width = image.shape[:2]
    return np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=float)


def _sample_nearest(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    xi = np.rint(x).astype(int)
    yi = np.rint(y).astype(int)
    valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    values = np.zeros((len(x), image.shape[2]), dtype=np.float64)
    values[valid] = image[yi[valid], xi[valid]]
    return values, valid


def _sample_bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    valid = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    values = np.zeros((len(x), image.shape[2]), dtype=np.float64)
    if not np.any(valid):
        return values, valid

    xv, yv = x[valid], y[valid]
    x0, y0 = np.floor(xv).astype(int), np.floor(yv).astype(int)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    dx, dy = xv - x0, yv - y0

    top = (1 - dx)[:, None] * image[y0, x0] + dx[:, None] * image[y0, x1]
    bottom = (1 - dx)[:, None] * image[y1, x0] + dx[:, None] * image[y1, x1]
    values[valid] = (1 - dy)[:, None] * top + dy[:, None] * bottom
    return values, valid


def warp_image(
    image: np.ndarray,
    H: np.ndarray,
    output_shape: tuple[int, int],
    method: str = "bilinear",
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-warp image with H (source -> output), returning RGB and mask."""
    if method not in {"nearest", "bilinear"}:
        raise ValueError("method must be 'nearest' or 'bilinear'.")
    out_height, out_width = output_shape
    yy, xx = np.indices((out_height, out_width), dtype=np.float64)
    output_points = np.column_stack((xx.ravel(), yy.ravel(), np.ones(out_height * out_width)))
    source_points = (np.linalg.inv(H) @ output_points.T).T
    source_points /= source_points[:, 2:3]

    rgb = image.astype(np.float64)
    if method == "nearest":
        values, valid = _sample_nearest(rgb, source_points[:, 0], source_points[:, 1])
    else:
        values, valid = _sample_bilinear(rgb, source_points[:, 0], source_points[:, 1])
    return values.reshape(out_height, out_width, 3), valid.reshape(out_height, out_width)


def warpImageNearestNeighbor(image: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Assignment-compatible nearest-neighbor warp into a tight output canvas."""
    corners = transform_points(H, image_corners(image))
    min_xy = np.floor(corners.min(axis=0)).astype(int)
    max_xy = np.ceil(corners.max(axis=0)).astype(int)
    translate = np.array([[1, 0, -min_xy[0]], [0, 1, -min_xy[1]], [0, 0, 1]], dtype=float)
    warped, _ = warp_image(image, translate @ H, (max_xy[1] - min_xy[1] + 1, max_xy[0] - min_xy[0] + 1), "nearest")
    return warped


def warpImageBilinear(image: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Assignment-compatible bilinear warp into a tight output canvas."""
    corners = transform_points(H, image_corners(image))
    min_xy = np.floor(corners.min(axis=0)).astype(int)
    max_xy = np.ceil(corners.max(axis=0)).astype(int)
    translate = np.array([[1, 0, -min_xy[0]], [0, 1, -min_xy[1]], [0, 0, 1]], dtype=float)
    warped, _ = warp_image(image, translate @ H, (max_xy[1] - min_xy[1] + 1, max_xy[0] - min_xy[0] + 1), "bilinear")
    return warped


def make_feather_weights(image: np.ndarray) -> np.ndarray:
    """Create a source-image alpha mask that increases toward the center."""
    height, width = image.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float64)
    distance_to_edge = np.minimum.reduce((xx + 1, yy + 1, width - xx, height - yy))
    return distance_to_edge / distance_to_edge.max()


def make_mosaic(images: Sequence[np.ndarray], homographies_to_reference: Sequence[np.ndarray], method: str = "bilinear") -> np.ndarray:
    """Warp and feather-blend images into one reference-coordinate mosaic.

    homographies_to_reference[i] maps coordinates in images[i] to coordinates
    in images[0]. The first homography should therefore be identity.
    """
    if len(images) != len(homographies_to_reference) or not images:
        raise ValueError("Provide one homography for every image.")

    all_corners = np.vstack([transform_points(H, image_corners(image)) for image, H in zip(images, homographies_to_reference)])
    min_xy = np.floor(all_corners.min(axis=0)).astype(int)
    max_xy = np.ceil(all_corners.max(axis=0)).astype(int)
    output_width, output_height = max_xy - min_xy + 1
    translation = np.array([[1, 0, -min_xy[0]], [0, 1, -min_xy[1]], [0, 0, 1]], dtype=float)

    color_sum = np.zeros((output_height, output_width, 3), dtype=np.float64)
    weight_sum = np.zeros((output_height, output_width), dtype=np.float64)
    for image, H in zip(images, homographies_to_reference):
        canvas_H = translation @ H
        warped, valid = warp_image(image, canvas_H, (output_height, output_width), method)
        source_weights = np.repeat(make_feather_weights(image)[..., None], 3, axis=2)
        warped_weights, weight_valid = warp_image(source_weights, canvas_H, (output_height, output_width), "bilinear")
        weights = warped_weights[..., 0] * valid * weight_valid
        color_sum += warped * weights[..., None]
        weight_sum += weights
    return color_sum / np.maximum(weight_sum[..., None], 1e-8)


def select_correspondences(source_image: np.ndarray, destination_image: np.ndarray, count: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Interactively collect matching points, clicked in the same order."""
    if count < 4:
        raise ValueError("Select at least four correspondences.")
    plt.figure(figsize=(10, 7))
    plt.imshow(source_image)
    plt.title(f"Click {count} points in SOURCE image, in a memorable order; press Enter when done")
    source = np.asarray(plt.ginput(count, timeout=0), dtype=float)
    plt.close()
    if len(source) != count:
        raise ValueError(f"Expected {count} source points, received {len(source)}.")

    plt.figure(figsize=(10, 7))
    plt.imshow(destination_image)
    plt.title("Click the same points in DESTINATION image, in exactly the same order")
    destination = np.asarray(plt.ginput(count, timeout=0), dtype=float)
    plt.close()
    if len(destination) != count:
        raise ValueError(f"Expected {count} destination points, received {len(destination)}.")
    return source, destination


def show_correspondences(source_image: np.ndarray, destination_image: np.ndarray, source: np.ndarray, destination: np.ndarray, output: str | Path) -> None:
    """Save a labeled side-by-side correspondence diagnostic."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 7))
    for axis, image, points, title in zip(axes, (source_image, destination_image), (source, destination), ("Source", "Destination")):
        axis.imshow(image)
        axis.scatter(points[:, 0], points[:, 1], c="lime", s=35, edgecolors="black")
        for index, (x, y) in enumerate(points, start=1):
            axis.text(x + 4, y + 4, str(index), color="yellow", fontsize=10, weight="bold")
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
