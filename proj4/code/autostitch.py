"""Automatic panorama stitching with Harris features, ANMS, and RANSAC."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from skimage import color, io
from skimage.feature import corner_harris, peak_local_max
from skimage.transform import resize


@dataclass
class Features:
    points_yx: np.ndarray
    descriptors: np.ndarray
    response: np.ndarray


@dataclass
class MatchSet:
    indices_a: np.ndarray
    indices_b: np.ndarray
    ratios: np.ndarray


def read_image(path: str | Path, max_dimension: int | None = None) -> np.ndarray:
    with Image.open(path) as source:
        image = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    image = image.astype(np.float64)
    image = image / 255.0 if image.max() > 1 else image
    if max_dimension and max(image.shape[:2]) > max_dimension:
        scale = max_dimension / max(image.shape[:2])
        image = resize(
            image,
            (round(image.shape[0] * scale), round(image.shape[1] * scale)),
            anti_aliasing=True,
            preserve_range=True,
        )
    return image


def grayscale(image: np.ndarray) -> np.ndarray:
    return color.rgb2gray(image) if image.ndim == 3 else image


def detect_harris(image: np.ndarray, edge_discard: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Return a Harris response image and candidate points in (y, x) order."""
    gray = grayscale(image)
    response = corner_harris(gray, method="eps", sigma=1)
    candidates = peak_local_max(response, min_distance=1)
    valid = (
        (candidates[:, 0] >= edge_discard)
        & (candidates[:, 0] < gray.shape[0] - edge_discard)
        & (candidates[:, 1] >= edge_discard)
        & (candidates[:, 1] < gray.shape[1] - edge_discard)
    )
    return response, candidates[valid]


def adaptive_nonmaximal_suppression(
    points_yx: np.ndarray,
    response: np.ndarray,
    count: int = 500,
    robust_factor: float = 0.9,
) -> np.ndarray:
    """Choose spatially distributed strong corners using Brown et al.'s ANMS."""
    if len(points_yx) <= count:
        return points_yx.copy()

    strengths = response[points_yx[:, 0], points_yx[:, 1]]
    order = np.argsort(strengths)[::-1]
    sorted_points = points_yx[order].astype(np.float64)
    sorted_strengths = strengths[order]
    radii = np.full(len(sorted_points), np.inf)

    for i in range(1, len(sorted_points)):
        stronger = sorted_strengths[:i] > robust_factor * sorted_strengths[i]
        if np.any(stronger):
            squared_distances = np.sum((sorted_points[:i][stronger] - sorted_points[i]) ** 2, axis=1)
            radii[i] = squared_distances.min()

    selected = np.argsort(radii)[::-1][:count]
    return sorted_points[selected].astype(int)


def strongest_candidates(points_yx: np.ndarray, response: np.ndarray, limit: int = 3000) -> np.ndarray:
    if len(points_yx) <= limit:
        return points_yx
    strengths = response[points_yx[:, 0], points_yx[:, 1]]
    return points_yx[np.argpartition(strengths, -limit)[-limit:]]


def extract_descriptors(
    image: np.ndarray,
    points_yx: np.ndarray,
    window_size: int = 40,
    descriptor_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract bias/gain-normalized 8x8 descriptors from blurred 40x40 windows."""
    gray = gaussian_filter(grayscale(image), sigma=1)
    half = window_size // 2
    valid_points: list[np.ndarray] = []
    descriptors: list[np.ndarray] = []

    for y, x in points_yx:
        patch = gray[y - half : y + half, x - half : x + half]
        if patch.shape != (window_size, window_size):
            continue
        small_patch = resize(
            patch,
            (descriptor_size, descriptor_size),
            anti_aliasing=True,
            preserve_range=True,
        )
        vector = small_patch.ravel()
        vector = vector - vector.mean()
        vector /= vector.std() + 1e-8
        valid_points.append(np.array([y, x]))
        descriptors.append(vector)

    if not descriptors:
        return np.empty((0, 2), dtype=int), np.empty((0, descriptor_size**2))
    return np.asarray(valid_points, dtype=int), np.vstack(descriptors)


def compute_features(image: np.ndarray, max_corners: int = 500) -> Features:
    response, candidates = detect_harris(image)
    candidates = strongest_candidates(candidates, response)
    selected = adaptive_nonmaximal_suppression(candidates, response, max_corners)
    points, descriptors = extract_descriptors(image, selected)
    return Features(points, descriptors, response)


def match_features(
    descriptors_a: np.ndarray,
    descriptors_b: np.ndarray,
    ratio_threshold: float = 0.7,
) -> MatchSet:
    """Match descriptors by Lowe's first-to-second-neighbor distance ratio."""
    if len(descriptors_a) == 0 or len(descriptors_b) < 2:
        return MatchSet(np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0))
    distances = cdist(descriptors_a, descriptors_b, metric="sqeuclidean")
    nearest = np.argpartition(distances, kth=1, axis=1)[:, :2]
    first_is_larger = distances[np.arange(len(distances)), nearest[:, 0]] > distances[
        np.arange(len(distances)), nearest[:, 1]
    ]
    nearest[first_is_larger] = nearest[first_is_larger, ::-1]
    first = distances[np.arange(len(distances)), nearest[:, 0]]
    second = distances[np.arange(len(distances)), nearest[:, 1]]
    ratios = np.sqrt(first / (second + 1e-12))
    accepted = ratios < ratio_threshold
    return MatchSet(np.flatnonzero(accepted), nearest[accepted, 0], ratios[accepted])


def _normalize_points(points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points_xy.mean(axis=0)
    mean_distance = np.mean(np.linalg.norm(points_xy - center, axis=1))
    scale = np.sqrt(2) / max(mean_distance, 1e-12)
    transform = np.array([[scale, 0, -scale * center[0]], [0, scale, -scale * center[1]], [0, 0, 1]])
    homogeneous = np.column_stack((points_xy, np.ones(len(points_xy))))
    return (transform @ homogeneous.T).T[:, :2], transform


def compute_homography(source_xy: np.ndarray, destination_xy: np.ndarray) -> np.ndarray:
    """Estimate the projective transform that maps source_xy to destination_xy."""
    if len(source_xy) < 4 or len(source_xy) != len(destination_xy):
        raise ValueError("Homography estimation requires at least four point pairs.")
    source_normalized, source_transform = _normalize_points(source_xy)
    destination_normalized, destination_transform = _normalize_points(destination_xy)
    design = []
    for (x, y), (u, v) in zip(source_normalized, destination_normalized):
        design.extend(([-x, -y, -1, 0, 0, 0, u * x, u * y, u], [0, 0, 0, -x, -y, -1, v * x, v * y, v]))
    _, _, right_vectors = np.linalg.svd(np.asarray(design))
    normalized_homography = right_vectors[-1].reshape(3, 3)
    homography = np.linalg.inv(destination_transform) @ normalized_homography @ source_transform
    return homography / homography[2, 2]


def project_points(homography: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points_xy, np.ones(len(points_xy))))
    projected = (homography @ homogeneous.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return projected[:, :2] / projected[:, 2:3]


def inverse_warp(image: np.ndarray, source_from_canvas: np.ndarray, output_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly sample an image into a canvas using an inverse homography."""
    output_y, output_x = np.indices(output_shape, dtype=float)
    canvas_points = np.stack((output_x.ravel(), output_y.ravel(), np.ones(output_x.size)))
    source_points = source_from_canvas @ canvas_points
    source_x = (source_points[0] / source_points[2]).reshape(output_shape)
    source_y = (source_points[1] / source_points[2]).reshape(output_shape)
    height, width = image.shape[:2]
    valid = (source_x >= 0) & (source_x < width - 1) & (source_y >= 0) & (source_y < height - 1)
    x0 = np.floor(np.clip(source_x, 0, width - 2)).astype(int)
    y0 = np.floor(np.clip(source_y, 0, height - 2)).astype(int)
    dx = source_x - x0
    dy = source_y - y0
    top = image[y0, x0] * (1 - dx[..., None]) + image[y0, x0 + 1] * dx[..., None]
    bottom = image[y0 + 1, x0] * (1 - dx[..., None]) + image[y0 + 1, x0 + 1] * dx[..., None]
    warped = top * (1 - dy[..., None]) + bottom * dy[..., None]
    warped[~valid] = 0
    return warped, valid


def ransac_homography(
    source_xy: np.ndarray,
    destination_xy: np.ndarray,
    iterations: int = 3000,
    threshold: float = 3.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Use four-point RANSAC and refit the best consensus set."""
    if len(source_xy) < 4:
        raise ValueError("At least four matches are needed for RANSAC.")
    rng = np.random.default_rng() if rng is None else rng
    best_inliers = np.zeros(len(source_xy), dtype=bool)
    best_error = np.inf

    for _ in range(iterations):
        sample = rng.choice(len(source_xy), 4, replace=False)
        try:
            homography = compute_homography(source_xy[sample], destination_xy[sample])
            error = np.linalg.norm(project_points(homography, source_xy) - destination_xy, axis=1)
        except np.linalg.LinAlgError:
            continue
        inliers = error <= threshold
        mean_error = error[inliers].mean() if np.any(inliers) else np.inf
        if inliers.sum() > best_inliers.sum() or (
            inliers.sum() == best_inliers.sum() and mean_error < best_error
        ):
            best_inliers, best_error = inliers, mean_error

    if best_inliers.sum() < 4:
        raise RuntimeError("RANSAC could not find a valid homography.")
    return compute_homography(source_xy[best_inliers], destination_xy[best_inliers]), best_inliers


def matched_coordinates(features_a: Features, features_b: Features, matches: MatchSet) -> tuple[np.ndarray, np.ndarray]:
    source_xy = features_a.points_yx[matches.indices_a][:, ::-1].astype(float)
    destination_xy = features_b.points_yx[matches.indices_b][:, ::-1].astype(float)
    return source_xy, destination_xy


def estimate_pairwise_homography(
    image_a: np.ndarray,
    image_b: np.ndarray,
    max_corners: int = 500,
    ratio_threshold: float = 0.7,
) -> tuple[np.ndarray, Features, Features, MatchSet, np.ndarray]:
    features_a = compute_features(image_a, max_corners)
    features_b = compute_features(image_b, max_corners)
    matches = match_features(features_a.descriptors, features_b.descriptors, ratio_threshold)
    source_xy, destination_xy = matched_coordinates(features_a, features_b, matches)
    homography, inliers = ransac_homography(source_xy, destination_xy)
    return homography, features_a, features_b, matches, inliers


def _corners(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    return np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=float)


def stitch_images(
    images: Sequence[np.ndarray], homographies: Sequence[np.ndarray], return_mask: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Blend images into a common canvas; H_i maps image i to image i+1."""
    if len(images) != len(homographies) + 1:
        raise ValueError("Expected one fewer homography than images.")
    transforms = [np.eye(3)]
    for homography in homographies:
        transforms.append(transforms[-1] @ np.linalg.inv(homography))

    all_corners = np.vstack([project_points(transform, _corners(image)) for image, transform in zip(images, transforms)])
    lower = np.floor(all_corners.min(axis=0)).astype(int)
    upper = np.ceil(all_corners.max(axis=0)).astype(int)
    translation = np.array([[1, 0, -lower[0]], [0, 1, -lower[1]], [0, 0, 1]])
    output_shape = (upper[1] - lower[1] + 1, upper[0] - lower[0] + 1)
    total = np.zeros((*output_shape, 3), dtype=float)
    weights = np.zeros(output_shape, dtype=float)

    for image, transform in zip(images, transforms):
        canvas_transform = translation @ transform
        inverse_map = np.linalg.inv(canvas_transform)
        warped, mask = inverse_warp(image, inverse_map, output_shape)
        total += warped
        weights += mask
    panorama = total / np.maximum(weights[..., None], 1e-8)
    return (panorama, weights > 0) if return_mask else panorama


def plot_diagnostics(
    image_a: np.ndarray,
    image_b: np.ndarray,
    features_a: Features,
    features_b: Features,
    matches: MatchSet,
    inliers: np.ndarray,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for image, features, label in ((image_a, features_a, "a"), (image_b, features_b, "b")):
        _, candidates = detect_harris(image)
        candidates = strongest_candidates(candidates, features.response)
        figure, axis = plt.subplots(figsize=(10, 7))
        axis.imshow(image)
        axis.scatter(candidates[:, 1], candidates[:, 0], s=3, c="yellow")
        axis.set_axis_off()
        figure.tight_layout()
        figure.savefig(output_dir / f"harris_corners_{label}.png", dpi=180)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(10, 7))
        axis.imshow(image)
        axis.scatter(features.points_yx[:, 1], features.points_yx[:, 0], s=5, c="lime")
        axis.set_axis_off()
        figure.tight_layout()
        figure.savefig(output_dir / f"anms_{label}.png", dpi=180)
        plt.close(figure)

        shown = min(16, len(features.descriptors))
        if shown:
            figure, axes = plt.subplots(4, 4, figsize=(6, 6))
            for axis, descriptor, point in zip(axes.flat, features.descriptors[:shown], features.points_yx[:shown]):
                axis.imshow(descriptor.reshape(8, 8), cmap="gray")
                axis.set_title(f"({point[1]}, {point[0]})", fontsize=7)
                axis.set_axis_off()
            for axis in axes.flat[shown:]:
                axis.set_axis_off()
            figure.tight_layout()
            figure.savefig(output_dir / f"descriptors_{label}.png", dpi=180)
            plt.close(figure)

    height = max(image_a.shape[0], image_b.shape[0])
    canvas = np.ones((height, image_a.shape[1] + image_b.shape[1], 3))
    canvas[: image_a.shape[0], : image_a.shape[1]] = image_a
    canvas[: image_b.shape[0], image_a.shape[1] :] = image_b
    figure, axis = plt.subplots(figsize=(14, 7))
    axis.imshow(canvas)
    for index in np.flatnonzero(inliers):
        y_a, x_a = features_a.points_yx[matches.indices_a[index]]
        y_b, x_b = features_b.points_yx[matches.indices_b[index]]
        axis.plot((x_a, x_b + image_a.shape[1]), (y_a, y_b), color="lime", linewidth=0.6)
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_dir / "ransac_inlier_matches.png", dpi=180)
    plt.close(figure)


def main(arguments: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="Ordered, overlapping input images")
    parser.add_argument("--output-dir", default="results", help="Directory for mosaic and diagnostic figures")
    parser.add_argument("--max-corners", type=int, default=500)
    parser.add_argument("--ratio-threshold", type=float, default=0.7)
    parser.add_argument("--max-dimension", type=int, default=1600, help="Longest working image dimension; use 0 for original size")
    args = parser.parse_args(arguments)
    if len(args.images) < 2:
        parser.error("Provide at least two overlapping images.")

    images = [read_image(path, args.max_dimension or None) for path in args.images]
    output_dir = Path(args.output_dir)
    homographies = []
    for index, (image_a, image_b) in enumerate(zip(images[:-1], images[1:])):
        homography, features_a, features_b, matches, inliers = estimate_pairwise_homography(
            image_a, image_b, args.max_corners, args.ratio_threshold
        )
        homographies.append(homography)
        plot_diagnostics(image_a, image_b, features_a, features_b, matches, inliers, output_dir / f"pair_{index + 1}")
        print(f"Pair {index + 1}: {inliers.sum()} RANSAC inliers out of {len(inliers)} Lowe-ratio matches")

    output_dir.mkdir(parents=True, exist_ok=True)
    panorama, mask = stitch_images(images, homographies, return_mask=True)
    rgba = np.dstack((np.clip(panorama, 0, 1), mask.astype(float)))
    io.imsave(output_dir / "automatic_mosaic.png", (rgba * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
