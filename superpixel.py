#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A simple SLIC superpixel implementation from scratch with visualization.

Features:
- Read an image
- Run SLIC in Lab + XY space
- Save boundary visualization
- Optional numeric labels on each superpixel region
- CLI support

Usage:
python superpixel.py \
    --input /nas/qirui/sam3/scenefun3d_ex/cut_masks/421254/35/INT__door_handle__001__area9664.png \
    --output /nas/qirui/sam3/scenefun3d_ex/superpixel_results/output.png \
    --num-superpixels 1800 \
    --compactness 10 \
    --max-iter 10 
    # --annotate

Optional:
    --save-label-map label_map.png
"""

# TODO: Test this switch img
# /nas/qirui/sam3/scenefun3d_ex/cut_masks/421254/515/INT__switch__000__area10213.png

import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


@dataclass
class ClusterCenter:
    l: float
    a: float
    b: float
    x: float
    y: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SLIC superpixel from scratch with visualization.")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, required=True, help="Path to save visualization image")
    parser.add_argument("--num-superpixels", type=int, default=200, help="Target number of superpixels")
    parser.add_argument("--compactness", type=float, default=10.0, help="Compactness parameter m")
    parser.add_argument("--max-iter", type=int, default=10, help="Maximum SLIC iterations")
    parser.add_argument("--annotate", action="store_true", help="Whether to draw region IDs on the visualization")
    parser.add_argument("--min-region-size-factor", type=float, default=0.25,
                        help="Minimum connected region size factor relative to expected region size")
    parser.add_argument("--boundary-color", type=str, default="255,0,0",
                        help="Boundary color as R,G,B, e.g. 255,0,0")
    parser.add_argument("--save-label-map", type=str, default="",
                        help="Optional path to save pseudo-color label map")
    return parser.parse_args()


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def save_image(path: str, img: np.ndarray) -> None:
    Image.fromarray(img.astype(np.uint8)).save(path)


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb / 255.0
    mask = rgb <= 0.04045
    linear = np.empty_like(rgb, dtype=np.float64)
    linear[mask] = rgb[mask] / 12.92
    linear[~mask] = ((rgb[~mask] + 0.055) / 1.055) ** 2.4
    return linear


def rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    # rgb shape: (H, W, 3), values in [0,255]
    linear = srgb_to_linear(rgb)

    # sRGB D65
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=np.float64)

    xyz = linear @ M.T
    return xyz


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    # D65 white reference
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883

    x = xyz[..., 0] / Xn
    y = xyz[..., 1] / Yn
    z = xyz[..., 2] / Zn

    delta = 6 / 29

    def f(t: np.ndarray) -> np.ndarray:
        mask = t > delta ** 3
        out = np.empty_like(t, dtype=np.float64)
        out[mask] = np.cbrt(t[mask])
        out[~mask] = t[~mask] / (3 * delta ** 2) + 4 / 29
        return out

    fx = f(x)
    fy = f(y)
    fz = f(z)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    lab = np.stack([L, a, b], axis=-1)
    return lab


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return xyz_to_lab(rgb_to_xyz(rgb))


def compute_gradient(lab: np.ndarray) -> np.ndarray:
    # Simple gradient for center perturbation
    H, W, _ = lab.shape
    grad = np.zeros((H, W), dtype=np.float64)

    # avoid borders for simplicity
    dx = np.sum((lab[:, 2:, :] - lab[:, :-2, :]) ** 2, axis=2)
    dy = np.sum((lab[2:, :, :] - lab[:-2, :, :]) ** 2, axis=2)

    grad[:, 1:-1] += dx
    grad[1:-1, :] += dy
    return grad


def initialize_centers(lab: np.ndarray, step: float) -> List[ClusterCenter]:
    H, W, _ = lab.shape
    grad = compute_gradient(lab)
    centers: List[ClusterCenter] = []

    half_step = int(step / 2)
    ys = list(range(half_step, H, int(step)))
    xs = list(range(half_step, W, int(step)))

    for y in ys:
        for x in xs:
            best_x, best_y = x, y
            best_grad = grad[y, x]

            # move to lowest gradient position in 3x3 neighborhood
            for ny in range(max(0, y - 1), min(H, y + 2)):
                for nx in range(max(0, x - 1), min(W, x + 2)):
                    if grad[ny, nx] < best_grad:
                        best_grad = grad[ny, nx]
                        best_x, best_y = nx, ny

            l, a, b = lab[best_y, best_x]
            centers.append(ClusterCenter(float(l), float(a), float(b), float(best_x), float(best_y)))

    return centers


def slic(lab: np.ndarray, num_superpixels: int, compactness: float, max_iter: int) -> np.ndarray:
    H, W, _ = lab.shape
    N = H * W
    step = math.sqrt(N / max(1, num_superpixels))
    centers = initialize_centers(lab, step)

    labels = -np.ones((H, W), dtype=np.int32)
    distances = np.full((H, W), np.inf, dtype=np.float64)

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    for _ in range(max_iter):
        distances.fill(np.inf)

        for idx, c in enumerate(centers):
            x0 = int(max(c.x - 2 * step, 0))
            x1 = int(min(c.x + 2 * step + 1, W))
            y0 = int(max(c.y - 2 * step, 0))
            y1 = int(min(c.y + 2 * step + 1, H))

            region_lab = lab[y0:y1, x0:x1]
            region_x = xx[y0:y1, x0:x1]
            region_y = yy[y0:y1, x0:x1]

            dc = np.sqrt(
                (region_lab[..., 0] - c.l) ** 2 +
                (region_lab[..., 1] - c.a) ** 2 +
                (region_lab[..., 2] - c.b) ** 2
            )

            ds = np.sqrt(
                (region_x - c.x) ** 2 +
                (region_y - c.y) ** 2
            )

            D = np.sqrt(dc ** 2 + ((compactness / step) * ds) ** 2)

            current = distances[y0:y1, x0:x1]
            mask = D < current
            current[mask] = D[mask]
            labels_region = labels[y0:y1, x0:x1]
            labels_region[mask] = idx

        # update centers
        new_centers: List[ClusterCenter] = []
        for idx, c in enumerate(centers):
            mask = labels == idx
            if not np.any(mask):
                new_centers.append(c)
                continue

            ys, xs = np.where(mask)
            region_lab = lab[mask]
            l_mean = float(np.mean(region_lab[:, 0]))
            a_mean = float(np.mean(region_lab[:, 1]))
            b_mean = float(np.mean(region_lab[:, 2]))
            x_mean = float(np.mean(xs))
            y_mean = float(np.mean(ys))
            new_centers.append(ClusterCenter(l_mean, a_mean, b_mean, x_mean, y_mean))

        centers = new_centers

    return labels


def get_neighbors(y: int, x: int, H: int, W: int) -> List[Tuple[int, int]]:
    nbrs = []
    if y > 0:
        nbrs.append((y - 1, x))
    if y < H - 1:
        nbrs.append((y + 1, x))
    if x > 0:
        nbrs.append((y, x - 1))
    if x < W - 1:
        nbrs.append((y, x + 1))
    return nbrs


def enforce_connectivity(labels: np.ndarray, min_region_size: int) -> np.ndarray:
    H, W = labels.shape
    new_labels = -np.ones((H, W), dtype=np.int32)
    visited = np.zeros((H, W), dtype=bool)
    current_new_label = 0

    for y in range(H):
        for x in range(W):
            if visited[y, x]:
                continue

            original_label = labels[y, x]
            stack = [(y, x)]
            component = []
            visited[y, x] = True

            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for ny, nx in get_neighbors(cy, cx, H, W):
                    if not visited[ny, nx] and labels[ny, nx] == original_label:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            if len(component) >= min_region_size:
                for cy, cx in component:
                    new_labels[cy, cx] = current_new_label
                current_new_label += 1
            else:
                # merge small component to neighboring assigned label if possible
                neighbor_labels = []
                for cy, cx in component:
                    for ny, nx in get_neighbors(cy, cx, H, W):
                        if new_labels[ny, nx] >= 0:
                            neighbor_labels.append(new_labels[ny, nx])

                if neighbor_labels:
                    target = max(set(neighbor_labels), key=neighbor_labels.count)
                else:
                    target = current_new_label
                    current_new_label += 1

                for cy, cx in component:
                    new_labels[cy, cx] = target

    return relabel_contiguously(new_labels)


def relabel_contiguously(labels: np.ndarray) -> np.ndarray:
    unique = np.unique(labels)
    mapping = {old: new for new, old in enumerate(unique)}
    out = np.vectorize(mapping.get)(labels)
    return out.astype(np.int32)


def find_boundaries(labels: np.ndarray) -> np.ndarray:
    H, W = labels.shape
    boundary = np.zeros((H, W), dtype=bool)

    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]

    return boundary


def overlay_boundaries(image: np.ndarray, labels: np.ndarray, boundary_color: Tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    boundary = find_boundaries(labels)
    out[boundary] = np.array(boundary_color, dtype=np.uint8)
    return out


def compute_region_centroids(labels: np.ndarray) -> List[Tuple[int, float, float]]:
    result = []
    for label_id in np.unique(labels):
        ys, xs = np.where(labels == label_id)
        if len(xs) == 0:
            continue
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        result.append((int(label_id), cx, cy))
    return result


def draw_with_annotations(image: np.ndarray, labels: np.ndarray, annotate: bool) -> np.ndarray:
    fig_h = max(4, image.shape[0] / 200)
    fig_w = max(4, image.shape[1] / 200)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.imshow(image)
    ax.axis("off")

    if annotate:
        centroids = compute_region_centroids(labels)
        for label_id, cx, cy in centroids:
            ax.text(
                cx, cy, str(label_id),
                color="yellow",
                fontsize=5,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.5, edgecolor="none")
            )

    fig.tight_layout(pad=0)
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    rendered = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return rendered


def labels_to_color_image(labels: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(42)
    num_labels = int(labels.max()) + 1
    colors = rng.integers(0, 255, size=(num_labels, 3), dtype=np.uint8)
    return colors[labels]


def parse_rgb_string(s: str) -> Tuple[int, int, int]:
    parts = s.split(",")
    if len(parts) != 3:
        raise ValueError("boundary-color must be in R,G,B format")
    return tuple(int(p) for p in parts)


def main() -> None:
    args = parse_args()

    image = load_image(args.input)
    H, W, _ = image.shape

    lab = rgb_to_lab(image)
    labels = slic(
        lab=lab,
        num_superpixels=args.num_superpixels,
        compactness=args.compactness,
        max_iter=args.max_iter,
    )

    expected_region_size = (H * W) / max(1, args.num_superpixels)
    min_region_size = max(1, int(expected_region_size * args.min_region_size_factor))
    labels = enforce_connectivity(labels, min_region_size=min_region_size)

    boundary_color = parse_rgb_string(args.boundary_color)
    vis = overlay_boundaries(image, labels, boundary_color=boundary_color)
    vis = draw_with_annotations(vis, labels, annotate=args.annotate)
    save_image(args.output, vis)

    if args.save_label_map:
        label_map = labels_to_color_image(labels)
        save_image(args.save_label_map, label_map)

    print(f"[INFO] Input image: {args.input}")
    print(f"[INFO] Output visualization saved to: {args.output}")
    print(f"[INFO] Number of final superpixels: {labels.max() + 1}")
    if args.save_label_map:
        print(f"[INFO] Label map saved to: {args.save_label_map}")


if __name__ == "__main__":
    main()