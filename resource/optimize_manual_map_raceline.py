#!/usr/bin/env python3
"""Generate a curvature-optimized raceline from a manually drawn ROS map.

This script uses a rough/cleaned closed path as the route seed, then optimizes
small lateral offsets against the map distance field. It is intentionally
self-contained so it can run even when the full TUM/TPH optimizer stack is not
installed.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from scipy.interpolate import CubicSpline, splev, splprep
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.optimize import minimize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, default=Path("maps/test_track_7.yaml"))
    parser.add_argument(
        "--seed",
        type=Path,
        default=Path("../raceline_optimization_upstream/test_track_7_processed/test_track_7_rough_cycle_final_0.05m.csv"),
        help="Closed route seed CSV with x/y or x_m/y_m columns.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/test_track_7"))
    parser.add_argument("--spacing", type=float, default=0.20, help="Optimization point spacing [m].")
    parser.add_argument("--final-spacing", type=float, default=0.05, help="Export point spacing [m].")
    parser.add_argument("--clearance", type=float, default=0.30, help="Minimum center point wall clearance [m].")
    parser.add_argument("--max-offset", type=float, default=0.60, help="Maximum lateral search distance [m].")
    parser.add_argument("--v-max", type=float, default=5.0)
    parser.add_argument("--v-min", type=float, default=1.0)
    parser.add_argument("--a-lat-max", type=float, default=3.2)
    parser.add_argument("--a-long-max", type=float, default=2.5)
    parser.add_argument("--a-brake-max", type=float, default=3.5)
    parser.add_argument(
        "--racing-line-weight",
        type=float,
        default=0.24,
        help="How strongly to prefer outside-apex-outside F1-style corner geometry.",
    )
    parser.add_argument("--apex-bias", type=float, default=0.82, help="Fraction of safe inside width used at apex.")
    parser.add_argument("--outside-bias", type=float, default=0.36, help="Fraction of safe outside width used at corner entry.")
    parser.add_argument("--max-iter", type=int, default=300)
    return parser.parse_args()


def load_xy(path: Path) -> np.ndarray:
    with path.open(newline="") as stream:
        sample = stream.read(4096)
        stream.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        reader = csv.DictReader(stream, dialect=dialect)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        names = {name.strip().lstrip("#").strip().lower(): name for name in reader.fieldnames}
        x_name = names.get("x", names.get("x_m"))
        y_name = names.get("y", names.get("y_m"))
        if x_name is None or y_name is None:
            raise ValueError(f"{path} must contain x/y or x_m/y_m columns")
        pts = np.asarray([(float(row[x_name]), float(row[y_name])) for row in reader], dtype=float)
    if len(pts) < 4 or not np.isfinite(pts).all():
        raise ValueError("Seed path requires at least four finite x/y points")
    return remove_duplicate_endpoint(pts)


def remove_duplicate_endpoint(points: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > tol
    points = points[keep]
    if len(points) > 1 and np.linalg.norm(points[0] - points[-1]) <= tol:
        points = points[:-1]
    return points


def closed_arclength(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    points = remove_duplicate_endpoint(points)
    closed = np.vstack((points, points[0]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.insert(np.cumsum(lengths), 0, 0.0)
    if cumulative[-1] <= 0.0:
        raise ValueError("Path has zero length")
    return closed, cumulative, float(cumulative[-1])


def resample_closed(points: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    closed, cumulative, total = closed_arclength(points)
    targets = np.arange(0.0, total, spacing)
    if len(targets) < 8:
        raise ValueError("Path is too short for requested spacing")
    resampled = np.column_stack((
        np.interp(targets, cumulative, closed[:, 0]),
        np.interp(targets, cumulative, closed[:, 1]),
    ))
    return resampled, targets


class MapField:
    def __init__(self, map_yaml: Path) -> None:
        with map_yaml.open("r") as stream:
            self.meta = yaml.safe_load(stream)
        image_path = map_yaml.parent / self.meta["image"]
        self.image = np.asarray(Image.open(image_path).transpose(Image.FLIP_TOP_BOTTOM), dtype=float)
        if self.image.ndim > 2:
            self.image = self.image[:, :, 0]
        self.resolution = float(self.meta["resolution"])
        self.origin = np.asarray(self.meta["origin"][:2], dtype=float)
        self.free = self.image > 210.0
        self.distance_px = distance_transform_edt(self.free)
        self.distance_m = self.distance_px * self.resolution

    def world_to_pixel(self, points: np.ndarray) -> np.ndarray:
        return (points - self.origin) / self.resolution

    def distance_at(self, points: np.ndarray) -> np.ndarray:
        pix = self.world_to_pixel(points)
        coords = np.vstack((pix[:, 1], pix[:, 0]))
        return map_coordinates(self.distance_m, coords, order=1, mode="constant", cval=-1.0)

    @property
    def extent(self) -> list[float]:
        h, w = self.image.shape[:2]
        return [
            self.origin[0],
            self.origin[0] + w * self.resolution,
            self.origin[1],
            self.origin[1] + h * self.resolution,
        ]


def smooth_periodic(points: np.ndarray, spacing: float, smoothing: float = 0.0) -> np.ndarray:
    closed, cumulative, total = closed_arclength(points)
    if smoothing > 0.0:
        count = max(8, int(math.ceil(total / spacing)))
        tck, _ = splprep([closed[:, 0], closed[:, 1]], s=smoothing, per=True, k=min(3, len(points) - 1))
        u_vals = np.linspace(0.0, 1.0, max(count * 6, 240), endpoint=False)
        x_vals, y_vals = splev(u_vals, tck)
        dense = np.column_stack((x_vals, y_vals))
        return resample_closed(dense, spacing)[0]
    cs_x = CubicSpline(cumulative, closed[:, 0], bc_type="periodic")
    cs_y = CubicSpline(cumulative, closed[:, 1], bc_type="periodic")
    targets = np.arange(0.0, total, spacing)
    return np.column_stack((cs_x(targets), cs_y(targets)))


def rotate_to_straight_section(points: np.ndarray) -> np.ndarray:
    """Move the periodic seam onto a locally straight section."""
    points = remove_duplicate_endpoint(points)
    if len(points) < 8:
        return points
    abs_kappa = np.abs(curvature(points))
    window = max(5, len(points) // 18)
    straight_score = circular_smooth(abs_kappa, window)
    start_idx = int(np.argmin(straight_score))
    return np.roll(points, -start_idx, axis=0)


def fair_high_curvature_points(
    points: np.ndarray,
    field: MapField,
    clearance: float,
    kappa_limit: float = 2.2,
    iterations: int = 220,
) -> np.ndarray:
    fair = remove_duplicate_endpoint(points).copy()
    for _ in range(iterations):
        kappa = np.abs(curvature(fair))
        hot = np.where(kappa > kappa_limit)[0]
        if hot.size == 0:
            break
        updated = fair.copy()
        for idx in hot:
            prev_pt = fair[(idx - 1) % len(fair)]
            next_pt = fair[(idx + 1) % len(fair)]
            candidate = 0.25 * fair[idx] + 0.375 * (prev_pt + next_pt)
            if field.distance_at(candidate[None, :])[0] >= clearance:
                updated[idx] = candidate
        fair = updated
    return fair


def normals_for(points: np.ndarray) -> np.ndarray:
    prev_pts = np.roll(points, 1, axis=0)
    next_pts = np.roll(points, -1, axis=0)
    tangents = next_pts - prev_pts
    lens = np.linalg.norm(tangents, axis=1)
    lens[lens < 1e-9] = 1.0
    tangents /= lens[:, None]
    return np.column_stack((tangents[:, 1], -tangents[:, 0]))


def offset_bounds(field: MapField, base: np.ndarray, normals: np.ndarray, clearance: float, max_offset: float) -> list[tuple[float, float]]:
    samples = np.linspace(-max_offset, max_offset, int(max_offset / 0.02) * 2 + 1)
    bounds: list[tuple[float, float]] = []
    for point, normal in zip(base, normals):
        candidates = point[None, :] + samples[:, None] * normal[None, :]
        safe = field.distance_at(candidates) >= clearance
        zero_idx = int(np.argmin(np.abs(samples)))
        if not safe[zero_idx]:
            safe_dist = field.distance_at(point[None, :])[0]
            raise ValueError(
                f"Seed path violates clearance near ({point[0]:.2f}, {point[1]:.2f}); "
                f"distance={safe_dist:.3f}m, required={clearance:.3f}m"
            )
        lo = zero_idx
        while lo > 0 and safe[lo - 1]:
            lo -= 1
        hi = zero_idx
        while hi < len(samples) - 1 and safe[hi + 1]:
            hi += 1
        bounds.append((float(samples[lo]), float(samples[hi])))
    return bounds


def curvature(points: np.ndarray) -> np.ndarray:
    p_prev = np.roll(points, 1, axis=0)
    p_next = np.roll(points, -1, axis=0)
    a = np.linalg.norm(points - p_prev, axis=1)
    b = np.linalg.norm(p_next - points, axis=1)
    c = np.linalg.norm(p_next - p_prev, axis=1)
    cross = (points[:, 0] - p_prev[:, 0]) * (p_next[:, 1] - points[:, 1]) - (
        points[:, 1] - p_prev[:, 1]
    ) * (p_next[:, 0] - points[:, 0])
    denom = a * b * c
    denom[denom < 1e-9] = 1e-9
    return 2.0 * cross / denom


def circular_smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    kernel = np.hanning(window)
    kernel /= np.sum(kernel)
    pad = window // 2
    padded = np.r_[values[-pad:], values, values[:pad]]
    return np.convolve(padded, kernel, mode="valid")


@dataclass(frozen=True)
class TurnSegment:
    start: int
    end: int
    entry: int
    apex: int
    exit: int
    sign: float
    inside_sign: float
    outside_sign: float


def circular_span(start: int, end: int, n: int) -> int:
    return (end - start) % n


def circular_range(start: int, end: int, n: int) -> list[int]:
    span = circular_span(start, end, n)
    return [(start + step) % n for step in range(span + 1)]


def detect_turn_segments(base: np.ndarray) -> tuple[np.ndarray, list[TurnSegment]]:
    n = len(base)
    kappa = circular_smooth(curvature(base), max(7, n // 24))
    abs_kappa = np.abs(kappa)
    threshold = max(0.20, float(np.percentile(abs_kappa, 76)))
    corner_mask = abs_kappa >= threshold

    raw_segments: list[list[int]] = []
    visited = np.zeros(n, dtype=bool)
    for start in range(n):
        if visited[start] or not corner_mask[start]:
            continue
        seg = []
        idx = start
        while corner_mask[idx] and not visited[idx]:
            visited[idx] = True
            seg.append(idx)
            idx = (idx + 1) % n
        raw_segments.append(seg)

    if len(raw_segments) > 1 and raw_segments[0][0] == 0 and raw_segments[-1][-1] == n - 1:
        raw_segments[0] = raw_segments[-1] + raw_segments[0]
        raw_segments.pop()

    turns: list[TurnSegment] = []
    for seg in raw_segments:
        if len(seg) < 3:
            continue
        signed_strength = float(np.sum(kappa[seg]))
        sign = 1.0 if signed_strength >= 0.0 else -1.0
        raw_apex = max(seg, key=lambda i: abs_kappa[i])
        raw_pos = seg.index(raw_apex)
        apex = seg[min(len(seg) - 1, raw_pos + max(1, int(len(seg) * 0.18)))]
        pad = max(4, int(len(seg) * 0.70))
        entry = (seg[0] - pad) % n
        exit_idx = (seg[-1] + max(3, int(len(seg) * 0.25))) % n
        turns.append(
            TurnSegment(
                start=seg[0],
                end=seg[-1],
                entry=entry,
                apex=apex,
                exit=exit_idx,
                sign=sign,
                inside_sign=-sign,
                outside_sign=sign,
            )
        )

    turns.sort(key=lambda turn: turn.entry)
    return kappa, turns


def detect_straight_spans(
    base: np.ndarray,
    min_points: int = 8,
    min_length: float = 1.0,
) -> list[np.ndarray]:
    n = len(base)
    _, turns = detect_turn_segments(base)
    if len(turns) < 2:
        return []

    spans: list[np.ndarray] = []
    for idx, turn in enumerate(turns):
        next_turn = turns[(idx + 1) % len(turns)]
        span = circular_span(turn.exit, next_turn.entry, n)
        if span < min_points:
            continue

        indices = np.asarray(circular_range(turn.exit, next_turn.entry, n), dtype=np.int64)
        length = 0.0
        for step in range(len(indices) - 1):
            length += float(np.linalg.norm(base[indices[step + 1]] - base[indices[step]]))
        if length >= min_length:
            spans.append(indices)
    return spans


def safe_width(bounds: list[tuple[float, float]], idx: int, side_sign: float) -> float:
    lo, hi = bounds[idx]
    return hi if side_sign > 0.0 else -lo


def add_anchor(
    anchors: list[tuple[int, float, float]],
    bounds: list[tuple[float, float]],
    idx: int,
    side_sign: float,
    fraction: float,
    weight: float,
) -> None:
    width = safe_width(bounds, idx, side_sign)
    lo, hi = bounds[idx]
    value = float(np.clip(side_sign * width * fraction, lo, hi))
    anchors.append((idx, value, weight))


def build_racing_line_target(
    base: np.ndarray,
    bounds: list[tuple[float, float]],
    apex_bias: float,
    outside_bias: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a sequence-aware racing-line target.

    Positive alpha moves along the local right normal. For left turns
    (positive curvature), inside is negative alpha. For right turns, inside is
    positive alpha.

    Same-direction turn sequences intentionally migrate from the current inside
    to the next outside before clipping the next inside. Opposite-direction
    transitions avoid a forced weave: the previous inside is already the next
    outside, so the target stays smooth on that side and lets curvature
    minimization choose the straightest average path.
    """
    n = len(base)
    _, turns = detect_turn_segments(base)
    if not turns:
        return np.zeros(n), np.zeros(n)

    anchors: list[tuple[int, float, float]] = []
    for idx, turn in enumerate(turns):
        prev_turn = turns[(idx - 1) % len(turns)]
        next_turn = turns[(idx + 1) % len(turns)]
        prev_same = prev_turn.sign == turn.sign
        next_same = next_turn.sign == turn.sign

        entry_fraction = outside_bias * (1.18 if prev_same else 1.0)
        apex_fraction = apex_bias * (1.05 if next_same else 1.0)
        add_anchor(anchors, bounds, turn.entry, turn.outside_sign, min(entry_fraction, 0.90), 1.0)
        add_anchor(anchors, bounds, turn.apex, turn.inside_sign, min(apex_fraction, 0.94), 1.35)

        if next_same:
            # Same direction: after hugging inside, deliberately migrate back
            # outside so the following same-direction corner starts wide.
            next_entry = next_turn.entry
            add_anchor(anchors, bounds, next_entry, turn.outside_sign, min(outside_bias * 1.25, 0.90), 1.15)
        else:
            # Opposite direction: previous inside is already next outside.
            # Keep this section low-pressure so curvature minimization finds
            # the straightest average transition instead of weaving centerline.
            bridge_span = circular_span(turn.apex, next_turn.entry, n)
            if bridge_span > 2:
                mid = (turn.apex + bridge_span // 2) % n
                side = turn.inside_sign
                add_anchor(anchors, bounds, mid, side, outside_bias * 0.55, 0.35)

    if len(anchors) < 2:
        return np.zeros(n), np.zeros(n)

    # Sort anchors around the lap and interpolate target/weight together.
    anchors = sorted(anchors, key=lambda item: item[0])
    target = np.zeros(n)
    weights = np.zeros(n)
    for anchor_idx, (start_i, start_v, start_w) in enumerate(anchors):
        end_i, end_v, end_w = anchors[(anchor_idx + 1) % len(anchors)]
        span = (end_i - start_i) % n
        if span == 0:
            continue
        for step in range(span + 1):
            idx = (start_i + step) % n
            phase = step / span
            # Smoothstep avoids abrupt lateral acceleration at the anchors.
            blend = phase * phase * (3.0 - 2.0 * phase)
            target[idx] = (1.0 - blend) * start_v + blend * end_v
            weights[idx] = (1.0 - blend) * start_w + blend * end_w

    for idx, (lo, hi) in enumerate(bounds):
        target[idx] = float(np.clip(target[idx], lo, hi))

    target = circular_smooth(target, max(25, n // 6))
    weights = np.clip(circular_smooth(weights, max(25, n // 6)), 0.15, 1.35)
    return target, weights


def optimize_offsets(
    base: np.ndarray,
    normals: np.ndarray,
    bounds: list[tuple[float, float]],
    max_iter: int,
    racing_line_weight: float,
    apex_bias: float,
    outside_bias: float,
    field: MapField | None = None,
    preferred_clearance: float | None = None,
) -> np.ndarray:
    n = len(base)
    target_alpha, target_weights = build_racing_line_target(base, bounds, apex_bias, outside_bias)
    use_preferred_clearance = (
        field is not None and preferred_clearance is not None and preferred_clearance > 0.0
    )

    def objective(alpha: np.ndarray) -> float:
        pts = base + alpha[:, None] * normals
        kappa = curvature(pts)
        d2_alpha = np.roll(alpha, -1) - 2.0 * alpha + np.roll(alpha, 1)
        d1_alpha = np.roll(alpha, -1) - alpha
        seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        clearance_cost = 0.0
        if use_preferred_clearance:
            clearance_shortfall = np.maximum(preferred_clearance - field.distance_at(pts), 0.0)
            clearance_cost = float(np.mean(clearance_shortfall * clearance_shortfall) * 18.0)
        return float(
            np.mean(kappa * kappa) * 8.0
            + np.mean(d2_alpha * d2_alpha) * 28.0
            + np.mean(d1_alpha * d1_alpha) * 14.0
            + np.mean(target_weights * (alpha - target_alpha) ** 2) * racing_line_weight
            + np.mean(alpha * alpha) * 0.08
            + np.mean(seg) * 0.18
            + clearance_cost
        )

    result = minimize(
        objective,
        np.clip(target_alpha, [lo for lo, _ in bounds], [hi for _, hi in bounds]),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "maxfun": max_iter * n * 4, "ftol": 1e-9, "maxls": 30},
    )
    if not result.success:
        print(f"WARNING: optimizer stopped before declared convergence: {result.message}")
    return np.asarray(result.x, dtype=float)


def path_from_offsets(points: np.ndarray, field: MapField, final_spacing: float) -> np.ndarray:
    smooth = smooth_periodic(rotate_to_straight_section(points), final_spacing, smoothing=0.08)
    for _ in range(10):
        smooth = fair_high_curvature_points(smooth, field, clearance=0.24)
        smooth = resample_closed(smooth, final_spacing)[0]
        if float(np.max(np.abs(curvature(smooth)))) <= 3.0:
            break
    return smooth


def trajectory_from_path(points: np.ndarray, field: MapField, final_spacing: float, v_min: float, v_max: float,
                         a_lat_max: float, a_long_max: float, a_brake_max: float) -> np.ndarray:
    smooth = path_from_offsets(points, field, final_spacing)
    closed, cumulative, total = closed_arclength(smooth)
    s_vals = cumulative[:-1]
    pts = closed[:-1]
    prev_pts = np.roll(pts, 1, axis=0)
    next_pts = np.roll(pts, -1, axis=0)
    tangents = next_pts - prev_pts
    psi = np.arctan2(tangents[:, 1], tangents[:, 0])
    kappa = curvature(pts)
    abs_kappa = np.maximum(np.abs(kappa), 1e-4)
    vx = np.clip(np.sqrt(a_lat_max / abs_kappa), v_min, v_max)
    seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    for _ in range(3):
        for i in range(len(vx)):
            j = (i + 1) % len(vx)
            vx[j] = min(vx[j], math.sqrt(max(vx[i] ** 2 + 2.0 * a_long_max * seg[i], v_min ** 2)))
        for i in range(len(vx) - 1, -1, -1):
            j = (i - 1) % len(vx)
            vx[j] = min(vx[j], math.sqrt(max(vx[i] ** 2 + 2.0 * a_brake_max * seg[j], v_min ** 2)))
    ax = np.zeros_like(vx)
    for i in range(len(vx)):
        if seg[i] > 1e-6:
            ax[i] = (vx[(i + 1) % len(vx)] ** 2 - vx[i] ** 2) / (2.0 * seg[i])
    traj = np.column_stack((s_vals, pts[:, 0], pts[:, 1], psi, kappa, vx, ax))
    closed_traj = np.vstack((traj, traj[0]))
    closed_traj[-1, 0] = total
    return closed_traj


def save_overlay(path: Path, field: MapField, seed: np.ndarray, base: np.ndarray, opt: np.ndarray, clearance: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.imshow(field.image, cmap="gray", origin="lower", extent=field.extent)
    ax.plot(seed[:, 0], seed[:, 1], color="#1f77b4", linewidth=1.0, alpha=0.65, label="seed route")
    ax.plot(np.r_[base[:, 0], base[0, 0]], np.r_[base[:, 1], base[0, 1]], color="#ffbf00", linewidth=1.3, label="centerline")
    ax.plot(np.r_[opt[:, 0], opt[0, 0]], np.r_[opt[:, 1], opt[0, 1]], color="#00d12f", linewidth=2.3, label="optimized raceline")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("map x [m]")
    ax.set_ylabel("map y [m]")
    ax.set_title(f"Optimized raceline, clearance >= {clearance:.2f} m")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    field = MapField(args.map)
    seed = load_xy(args.seed)
    seed_smooth = smooth_periodic(seed, args.final_spacing)
    base = smooth_periodic(seed_smooth, args.spacing)
    normals = normals_for(base)
    bounds = offset_bounds(field, base, normals, args.clearance, args.max_offset)
    alpha = optimize_offsets(
        base,
        normals,
        bounds,
        args.max_iter,
        args.racing_line_weight,
        args.apex_bias,
        args.outside_bias,
    )
    opt_sparse = base + alpha[:, None] * normals
    traj = trajectory_from_path(
        opt_sparse,
        field,
        args.final_spacing,
        args.v_min,
        args.v_max,
        args.a_lat_max,
        args.a_long_max,
        args.a_brake_max,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    track_path = args.output_dir / "optimized_track.csv"
    traj_path = args.output_dir / "optimized_raceline.csv"
    overlay_path = args.output_dir / "optimized_raceline_overlay.png"

    widths = np.asarray([(hi - lo) * 0.5 for lo, hi in bounds])
    track = np.column_stack((base[:, 0], base[:, 1], widths, widths))
    np.savetxt(track_path, track, fmt="%.7f", delimiter=",", header="x_m,y_m,w_tr_right_m,w_tr_left_m")
    with traj_path.open("w") as stream:
        stream.write("# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2\n")
        np.savetxt(stream, traj, fmt="%.7f", delimiter=";")
    save_overlay(overlay_path, field, seed_smooth, base, traj[:-1, 1:3], args.clearance)

    min_clearance = float(np.min(field.distance_at(traj[:-1, 1:3])))
    length = float(traj[-1, 0])
    lap_time = float(np.sum(np.linalg.norm(np.diff(traj[:, 1:3], axis=0), axis=1) /
                            np.maximum(0.5 * (traj[:-1, 5] + traj[1:, 5]), 0.01)))
    print(f"wrote {track_path}")
    print(f"wrote {traj_path}")
    print(f"wrote {overlay_path}")
    print(f"points={len(traj)} length={length:.2f}m min_clearance={min_clearance:.3f}m")
    print(f"speed={np.min(traj[:-1, 5]):.2f}-{np.max(traj[:-1, 5]):.2f}m/s estimated_lap={lap_time:.2f}s")


if __name__ == "__main__":
    main()
