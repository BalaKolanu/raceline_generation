#!/usr/bin/env python3
"""Clean a ROS map and convert an input raceline into boundary-safe waypoints.

Pipeline:
1. Remove isolated occupied components containing at most three pixels.
2. Rigidly orient the input path to minimize overlap with cleaned boundaries.
3. Remove local self-overlap loops while preserving the full closed lap.
4. Select waypoints using steering-dependent spacing.
5. Connect consecutive waypoints by straight, shortest-distance segments.
6. Smooth the result below a steering limit and sample it every 0.05 m.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map_yaml", type=Path,
                        help="ROS map YAML whose image will be cleaned")
    parser.add_argument("raceline", type=Path,
                        help="Input CSV containing x/y or x_m/y_m columns")
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to <raceline_name>_output beside the raceline")
    parser.add_argument("--max-isolated-pixels", type=int, default=3)
    parser.add_argument("--wall-threshold", type=int, default=100)
    parser.add_argument("--boundary-clearance", type=float, default=0.4,
                        help="Minimum centerline distance from occupied pixels [m]")
    parser.add_argument("--wheelbase", type=float, default=0.324)
    parser.add_argument("--loop-radius", type=float, default=0.12,
                        help="Near-return distance used to identify local loops [m]")
    parser.add_argument("--max-local-loop", type=float, default=6.0,
                        help="Maximum arc length eligible for local-loop removal [m]")
    parser.add_argument("--search-translation", type=float, default=1.0,
                        help="Maximum automatic x/y registration adjustment [m]")
    parser.add_argument("--search-rotation-deg", type=float, default=15.0,
                        help="Maximum automatic registration rotation [degrees]")
    parser.add_argument("--dense-spacing", type=float, default=0.03,
                        help="Internal path sampling distance [m]")
    parser.add_argument("--max-final-steering", type=float, default=0.6,
                        help="Maximum absolute steering in the final CSV [rad]")
    parser.add_argument("--final-spacing", type=float, default=0.05,
                        help="Point spacing in the final smoothed CSV [m]")
    parser.add_argument("--speed-steering-min", type=float, default=0.0,
                        help="Steering angle that maps to max speed [rad]")
    parser.add_argument("--speed-steering-max", type=float, default=0.58,
                        help="Steering angle that maps to min speed [rad]")
    parser.add_argument("--min-speed", type=float, default=1.0,
                        help="Minimum output speed [m/s]")
    parser.add_argument("--max-speed", type=float, default=5.0,
                        help="Maximum output speed [m/s]")
    parser.add_argument("--speed-profile", choices=("piecewise", "linear"),
                        default="piecewise",
                        help="Speed assignment method for the final path")
    parser.add_argument("--straight-steering-threshold", type=float, default=0.08,
                        help="Maximum |steering| that still counts as a straight [rad]")
    parser.add_argument("--turn-angle-threshold-deg", type=float, default=30.0,
                        help="Minimum accumulated heading change that counts as a turn [deg]")
    parser.add_argument("--turn-distance-threshold-m", type=float, default=1.5,
                        help="Maximum path distance within which the turn angle must be reached [m]")
    parser.add_argument("--gentle-steering-threshold", type=float, default=0.20,
                        help="Maximum |steering| for the gentle-bend speed tier [rad]")
    parser.add_argument("--medium-steering-threshold", type=float, default=0.38,
                        help="Maximum |steering| for the medium-turn speed tier [rad]")
    parser.add_argument("--straight-speed", type=float, default=10.0,
                        help="Speed used on straights [m/s]")
    parser.add_argument("--gentle-speed", type=float, default=7.0,
                        help="Speed used on gentle bends [m/s]")
    parser.add_argument("--medium-speed", type=float, default=4.5,
                        help="Speed used on medium turns [m/s]")
    parser.add_argument("--corner-speed", type=float, default=2.0,
                        help="Speed used in tight corners [m/s]")
    return parser.parse_args()


def load_xy(path: Path) -> np.ndarray:
    with path.open(newline="") as stream:
        sample = stream.read(4096)
        stream.seek(0)
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;").delimiter
        reader = csv.DictReader(stream, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        names = {name.strip().lstrip("#").strip().lower(): name
                 for name in reader.fieldnames}
        x_name = names.get("x", names.get("x_m"))
        y_name = names.get("y", names.get("y_m"))
        if x_name is None or y_name is None:
            raise ValueError(f"{path} must contain x/y or x_m/y_m columns")
        points = np.asarray([
            (float(row[x_name]), float(row[y_name])) for row in reader
        ], dtype=float)
    if len(points) < 4 or not np.isfinite(points).all():
        raise ValueError("Raceline requires at least four finite points")
    return remove_duplicates(points)


def remove_duplicates(points: np.ndarray, tolerance: float = 1e-5) -> np.ndarray:
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > tolerance
    points = points[keep]
    if len(points) > 1 and np.linalg.norm(points[0] - points[-1]) <= tolerance:
        points = points[:-1]
    return points


def resample_closed(points: np.ndarray, spacing: float | None = None,
                    count: int | None = None) -> np.ndarray:
    points = remove_duplicates(points)
    closed = np.vstack((points, points[0]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.insert(np.cumsum(lengths), 0, 0.0)
    if cumulative[-1] <= 0.0:
        raise ValueError("Raceline has zero length")
    if count is None:
        if spacing is None or spacing <= 0.0:
            raise ValueError("Positive spacing or point count required")
        count = max(8, int(math.ceil(cumulative[-1] / spacing)))
    targets = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    return np.column_stack((
        np.interp(targets, cumulative, closed[:, 0]),
        np.interp(targets, cumulative, closed[:, 1]),
    ))


def resample_closed_exact(points: np.ndarray, spacing: float) -> np.ndarray:
    """Sample a closed polyline at exact distance increments from its start."""
    points = remove_duplicates(points)
    closed = np.vstack((points, points[0]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.insert(np.cumsum(lengths), 0, 0.0)
    targets = np.arange(0.0, cumulative[-1], spacing)
    return np.column_stack((
        np.interp(targets, cumulative, closed[:, 0]),
        np.interp(targets, cumulative, closed[:, 1]),
    ))


def clean_map(image: np.ndarray, threshold: int,
              max_component_size: int) -> tuple[np.ndarray, int, int]:
    occupied = (image <= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(occupied, 8)
    remove_ids = [
        component for component in range(1, count)
        if stats[component, cv2.CC_STAT_AREA] <= max_component_size
    ]
    cleaned = image.copy()
    if remove_ids:
        cleaned[np.isin(labels, remove_ids)] = 255
    removed_pixels = int(sum(stats[item, cv2.CC_STAT_AREA] for item in remove_ids))
    return cleaned, len(remove_ids), removed_pixels


class MapField:
    def __init__(self, image: np.ndarray, metadata: dict, threshold: int) -> None:
        self.image = image
        self.resolution = float(metadata["resolution"])
        self.origin = np.asarray(metadata["origin"][:2], dtype=float)
        if abs(float(metadata["origin"][2])) > 1e-9:
            raise ValueError("Rotated map origins are not supported")
        free = (image > threshold).astype(np.uint8)
        occupied = 1 - free
        self.distance = cv2.distanceTransform(free, cv2.DIST_L2, 5) * self.resolution
        inside = cv2.distanceTransform(occupied, cv2.DIST_L2, 5) * self.resolution
        self.signed_distance = self.distance - inside
        gx = cv2.Sobel(self.signed_distance, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(self.signed_distance, cv2.CV_32F, 0, 1, ksize=3)
        self.gradient_x = gx
        self.gradient_y = -gy

    @property
    def extent(self) -> list[float]:
        return [
            self.origin[0], self.origin[0] + self.image.shape[1] * self.resolution,
            self.origin[1], self.origin[1] + self.image.shape[0] * self.resolution,
        ]

    def sample(self, array: np.ndarray, points: np.ndarray,
               outside_value: float = -1.0) -> np.ndarray:
        cols = ((points[:, 0] - self.origin[0]) / self.resolution).astype(np.float32)
        rows = (self.image.shape[0] - 1
                - (points[:, 1] - self.origin[1]) / self.resolution).astype(np.float32)
        return cv2.remap(
            array.astype(np.float32), cols[:, None], rows[:, None], cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=outside_value,
        ).ravel()

    def distance_and_gradient(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distance = self.sample(self.signed_distance, points)
        gradient = np.column_stack((
            self.sample(self.gradient_x, points, 0.0),
            self.sample(self.gradient_y, points, 0.0),
        ))
        norm = np.linalg.norm(gradient, axis=1)
        valid = norm > 1e-8
        gradient[valid] /= norm[valid, None]
        gradient[~valid] = 0.0
        return distance, gradient


def transform(points: np.ndarray, angle: float, translation: np.ndarray,
              pivot: np.ndarray) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    return (points - pivot) @ rotation.T + pivot + translation


def alignment_score(points: np.ndarray, field: MapField, clearance: float,
                    angle: float, translation: np.ndarray,
                    translation_limit: float, rotation_limit: float) -> float:
    distance = field.sample(field.signed_distance, points)
    deficit = np.maximum(clearance - distance, 0.0)
    collision_cost = np.mean(deficit ** 2) * 5000.0 + np.mean(distance <= 0.0) * 500.0
    # Prefer the smallest correction among similarly collision-free candidates.
    motion_cost = 0.03 * np.sum((translation / max(translation_limit, 1e-6)) ** 2)
    motion_cost += 0.03 * (angle / max(rotation_limit, 1e-6)) ** 2
    return float(collision_cost + motion_cost)


def orient_path(points: np.ndarray, field: MapField, args: argparse.Namespace) -> tuple[np.ndarray, float, np.ndarray]:
    pivot = np.mean(points, axis=0)
    rotation_limit = math.radians(args.search_rotation_deg)
    best = (math.inf, 0.0, np.zeros(2), points)
    stages = [
        (rotation_limit, args.search_translation, 9, 11),
        (math.radians(3.0), 0.25, 9, 9),
        (math.radians(0.75), 0.06, 7, 7),
    ]
    center_angle = 0.0
    center_translation = np.zeros(2)
    for angle_radius, translation_radius, angle_steps, translation_steps in stages:
        angles = center_angle + np.linspace(-angle_radius, angle_radius, angle_steps)
        offsets = np.linspace(-translation_radius, translation_radius, translation_steps)
        for angle in angles:
            for dx in offsets:
                for dy in offsets:
                    translation = center_translation + np.asarray([dx, dy])
                    candidate = transform(points, angle, translation, pivot)
                    score = alignment_score(
                        candidate, field, args.boundary_clearance, angle, translation,
                        args.search_translation, rotation_limit,
                    )
                    if score < best[0]:
                        best = (score, angle, translation, candidate)
        center_angle, center_translation = best[1], best[2]
    return best[3], best[1], best[2]


def erase_near_return_loops(points: np.ndarray, radius: float,
                            maximum_loop_length: float) -> np.ndarray:
    result: list[np.ndarray] = []
    minimum_index_gap = 5
    for point in points:
        if len(result) > minimum_index_gap:
            previous = np.asarray(result[:-minimum_index_gap])
            distances = np.linalg.norm(previous - point, axis=1)
            matches = np.flatnonzero(distances < radius)
            if len(matches):
                match = int(matches[-1])
                loop = np.asarray(result[match:] + [point])
                loop_length = np.linalg.norm(np.diff(loop, axis=0), axis=1).sum()
                if 2.0 * radius < loop_length <= maximum_loop_length:
                    result = result[:match + 1]
        result.append(point)
    return remove_duplicates(np.asarray(result))


def segment_intersection(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                         d: np.ndarray) -> tuple[bool, np.ndarray]:
    r = b - a
    s = d - c
    denominator = r[0] * s[1] - r[1] * s[0]
    if abs(denominator) < 1e-10:
        return False, np.zeros(2)
    delta = c - a
    t = (delta[0] * s[1] - delta[1] * s[0]) / denominator
    u = (delta[0] * r[1] - delta[1] * r[0]) / denominator
    return 1e-6 < t < 1.0 - 1e-6 and 1e-6 < u < 1.0 - 1e-6, a + t * r


def erase_crossing_loops(points: np.ndarray, maximum_loop_length: float) -> np.ndarray:
    current = points.copy()
    for _ in range(50):
        count = len(current)
        cumulative = np.insert(np.cumsum(np.linalg.norm(
            np.diff(np.vstack((current, current[0])), axis=0), axis=1)), 0, 0.0)
        changed = False
        for first in range(count):
            a, b = current[first], current[(first + 1) % count]
            for second in range(first + 2, count):
                if first == 0 and second == count - 1:
                    continue
                c, d = current[second], current[(second + 1) % count]
                intersects, crossing = segment_intersection(a, b, c, d)
                if not intersects:
                    continue
                forward_length = cumulative[second] - cumulative[first + 1]
                total_length = cumulative[-1]
                complementary_length = total_length - forward_length
                if forward_length <= complementary_length and forward_length <= maximum_loop_length:
                    current = np.vstack((current[:first + 1], crossing, current[second + 1:]))
                    changed = True
                    break
                if complementary_length < forward_length and complementary_length <= maximum_loop_length:
                    current = np.vstack((crossing, current[first + 1:second + 1]))
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break
    return remove_duplicates(current)


def project_from_boundaries(points: np.ndarray, field: MapField,
                            clearance: float) -> np.ndarray:
    projected = points.copy()
    for _ in range(80):
        distance, gradient = field.distance_and_gradient(projected)
        deficit = np.maximum(clearance - distance, 0.0)
        if deficit.max(initial=0.0) < 1e-4:
            break
        projected += np.minimum(deficit + 0.002, 0.04)[:, None] * gradient
    return projected


def curvature(points: np.ndarray) -> np.ndarray:
    previous = np.roll(points, 1, axis=0)
    following = np.roll(points, -1, axis=0)
    a = points - previous
    b = following - points
    c = following - previous
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(c, axis=1)
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    result = np.zeros(len(points))
    valid = denominator > 1e-9
    result[valid] = 2.0 * cross[valid] / denominator[valid]
    return result


def spacing_for_steering(steering: float) -> float:
    value = abs(steering)
    if value < 0.1 or value > 0.6:
        return 0.5
    if value < 0.25:
        return 0.3
    return 0.1


def adaptive_waypoints(dense: np.ndarray, steering: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    segments = np.linalg.norm(np.roll(dense, -1, axis=0) - dense, axis=1)
    selected = [0]
    used_spacing = [spacing_for_steering(steering[0])]
    travelled = 0.0
    index = 0
    while True:
        target = spacing_for_steering(steering[index])
        next_index = (index + 1) % len(dense)
        travelled += segments[index]
        index = next_index
        if index == 0:
            break
        if travelled >= target:
            selected.append(index)
            used_spacing.append(target)
            travelled = 0.0
    return dense[np.asarray(selected)], np.asarray(used_spacing)


def straight_segment(a: np.ndarray, b: np.ndarray, spacing: float = 0.02) -> np.ndarray:
    distance = np.linalg.norm(b - a)
    count = max(2, int(math.ceil(distance / spacing)) + 1)
    return np.linspace(a, b, count)


def insert_safety_points(waypoints: np.ndarray, source: np.ndarray, field: MapField,
                         clearance: float) -> np.ndarray:
    """Keep straight links, inserting source points only when a shortcut hits a wall."""
    result = [waypoints[0]]
    for index in range(len(waypoints)):
        start = result[-1]
        goal = waypoints[(index + 1) % len(waypoints)]
        samples = straight_segment(start, goal)
        if np.min(field.sample(field.signed_distance, samples)) >= clearance:
            if index + 1 < len(waypoints):
                result.append(goal)
            continue
        # Find the nearest source point near the middle of the unsafe shortcut.
        midpoint = 0.5 * (start + goal)
        source_index = int(np.argmin(np.linalg.norm(source - midpoint, axis=1)))
        candidate = source[source_index]
        if np.linalg.norm(candidate - start) > 1e-4 and np.linalg.norm(candidate - goal) > 1e-4:
            result.append(candidate)
        if index + 1 < len(waypoints):
            result.append(goal)
    return remove_duplicates(np.asarray(result))


def steering_angles(points: np.ndarray, wheelbase: float) -> np.ndarray:
    return np.arctan(wheelbase * curvature(points))


def speed_from_previous_steering(steering: np.ndarray, steering_min: float,
                                 steering_max: float, speed_min: float,
                                 speed_max: float) -> np.ndarray:
    if steering_min < 0.0:
        raise ValueError("Minimum steering for speed mapping must be non-negative")
    if steering_max <= steering_min:
        raise ValueError("Maximum steering for speed mapping must exceed minimum steering")
    if speed_max < speed_min:
        raise ValueError("Maximum speed must be greater than or equal to minimum speed")

    previous_steering = np.abs(np.roll(steering, 1))
    ratio = (previous_steering - steering_min) / (steering_max - steering_min)
    ratio = np.clip(ratio, 0.0, 1.0)
    return speed_max - ratio * (speed_max - speed_min)


def piecewise_speed_from_previous_steering(
    steering: np.ndarray,
    straight_threshold: float,
    gentle_threshold: float,
    medium_threshold: float,
    straight_speed: float,
    gentle_speed: float,
    medium_speed: float,
    corner_speed: float,
) -> np.ndarray:
    if straight_threshold < 0.0:
        raise ValueError("Straight steering threshold must be non-negative")
    if not straight_threshold <= gentle_threshold <= medium_threshold:
        raise ValueError(
            "Piecewise steering thresholds must satisfy straight <= gentle <= medium"
        )
    if not straight_speed >= gentle_speed >= medium_speed >= corner_speed:
        raise ValueError(
            "Piecewise speeds must satisfy straight >= gentle >= medium >= corner"
        )

    previous_steering = np.abs(np.roll(steering, 1))
    speed = np.full(previous_steering.shape, corner_speed, dtype=float)
    speed[previous_steering <= medium_threshold] = medium_speed
    speed[previous_steering <= gentle_threshold] = gentle_speed
    speed[previous_steering <= straight_threshold] = straight_speed
    return speed


def classify_turn_states(
    heading: np.ndarray,
    spacing: float,
    turn_angle_threshold_deg: float,
    turn_distance_threshold_m: float,
) -> list[str]:
    if spacing <= 0.0:
        raise ValueError("Classification spacing must be positive")
    if turn_angle_threshold_deg <= 0.0:
        raise ValueError("Turn angle threshold must be positive")
    if turn_distance_threshold_m <= 0.0:
        raise ValueError("Turn distance threshold must be positive")

    angle_threshold = math.radians(turn_angle_threshold_deg)
    window_count = max(1, int(math.ceil(turn_distance_threshold_m / spacing)))
    half_window_count = max(1, window_count // 2)
    deltas = np.arctan2(
        np.sin(np.roll(heading, -1) - heading),
        np.cos(np.roll(heading, -1) - heading),
    )

    def accumulated_change(start: int, steps: int) -> float:
        total = 0.0
        for offset in range(steps):
            total += deltas[(start + offset) % len(deltas)]
        return total

    labels: list[str] = []
    for index in range(len(heading)):
        centered_start = (index - half_window_count) % len(heading)
        centered_change = accumulated_change(centered_start, window_count)
        if abs(centered_change) >= angle_threshold:
            labels.append("L" if centered_change > 0.0 else "R")
            continue

        forward_change = accumulated_change(index, window_count)
        if forward_change >= angle_threshold:
            labels.append("ASL")
        elif forward_change <= -angle_threshold:
            labels.append("ASR")
        else:
            labels.append("S")
    return labels


def smooth_with_steering_limit(points: np.ndarray, field: MapField,
                               clearance: float, wheelbase: float,
                               spacing: float, maximum_steering: float) -> np.ndarray:
    """Fair a closed path while retaining clearance and a steering margin."""
    if spacing <= 0.0:
        raise ValueError("Final spacing must be positive")
    if not 0.0 < maximum_steering < math.pi / 2.0:
        raise ValueError("Maximum final steering must be between 0 and pi/2")

    # Keeping each vertex this far away guarantees that the line segment between
    # adjacent vertices also retains the requested clearance.
    vertex_clearance = clearance + 0.5 * spacing + 0.002
    smoothed = resample_closed_exact(points, spacing)
    smoothed = project_from_boundaries(smoothed, field, vertex_clearance)
    reference = smoothed.copy()
    # The margin absorbs small curvature changes introduced by final resampling.
    target = maximum_steering - min(0.02, maximum_steering * 0.05)
    for iteration in range(4000):
        steering = np.abs(steering_angles(smoothed, wheelbase))
        if iteration > 20 and float(np.max(steering)) <= target:
            break

        previous = np.roll(smoothed, 1, axis=0)
        following = np.roll(smoothed, -1, axis=0)
        laplacian = 0.5 * (previous + following) - smoothed
        violation = np.clip((steering - target) / max(target, 1e-6), 0.0, 2.0)
        strength = 0.10 + 0.30 * violation
        proposal = smoothed + strength[:, None] * laplacian
        # A weak tether prevents unconstrained Laplacian shrinkage on long straights.
        proposal += 0.002 * (reference - smoothed)
        smoothed = project_from_boundaries(proposal, field, vertex_clearance)

        if (iteration + 1) % 20 == 0:
            smoothed = resample_closed(smoothed, count=len(reference))
            smoothed = project_from_boundaries(smoothed, field, vertex_clearance)

    final = resample_closed_exact(smoothed, spacing)
    final = project_from_boundaries(final, field, vertex_clearance)
    # A few final uniform fairing passes remove kinks caused by wall projection.
    for _ in range(300):
        maximum = float(np.max(np.abs(steering_angles(final, wheelbase))))
        if maximum < maximum_steering:
            break
        laplacian = 0.5 * (np.roll(final, 1, axis=0)
                           + np.roll(final, -1, axis=0)) - final
        final = project_from_boundaries(
            final + 0.16 * laplacian, field, vertex_clearance,
        )
    final = resample_closed_exact(final, spacing)
    return project_from_boundaries(final, field, vertex_clearance)


def export_waypoints(path: Path, waypoints: np.ndarray, wheelbase: float,
                     include_spacing_rule: bool = True,
                     fixed_spacing: float | None = None,
                     include_speed: bool = False,
                     include_turn_state: bool = False,
                     speed_profile: str = "piecewise",
                     speed_steering_min: float = 0.0,
                     speed_steering_max: float = 0.58,
                     min_speed: float = 1.0,
                     max_speed: float = 5.0,
                     straight_steering_threshold: float = 0.08,
                     turn_angle_threshold_deg: float = 30.0,
                     turn_distance_threshold_m: float = 2.5,
                     gentle_steering_threshold: float = 0.20,
                     medium_steering_threshold: float = 0.38,
                     straight_speed: float = 10.0,
                     gentle_speed: float = 7.0,
                     medium_speed: float = 4.5,
                     corner_speed: float = 2.0) -> None:
    closed = np.vstack((waypoints, waypoints[0]))
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    if fixed_spacing is None:
        distance = np.insert(np.cumsum(segment_lengths[:-1]), 0, 0.0)
    else:
        distance = np.arange(len(waypoints), dtype=float) * fixed_spacing
    heading = np.arctan2(np.roll(waypoints, -1, axis=0)[:, 1] - waypoints[:, 1],
                         np.roll(waypoints, -1, axis=0)[:, 0] - waypoints[:, 0])
    kappa = curvature(waypoints)
    steering = np.arctan(wheelbase * kappa)
    rows: list[list[object]] = []
    header = ["index", "s", "x", "y", "yaw", "kappa", "steering"]
    spacing_rule = None
    if include_spacing_rule:
        spacing_rule = np.asarray([spacing_for_steering(value) for value in steering])
        header.append("spacing_rule")
    turn_state = None
    if include_turn_state:
        point_spacing = fixed_spacing if fixed_spacing is not None else float(np.mean(segment_lengths))
        turn_state = classify_turn_states(
            heading,
            point_spacing,
            turn_angle_threshold_deg,
            turn_distance_threshold_m,
        )
        header.append("turn_state")
    speed = None
    if include_speed:
        if speed_profile == "linear":
            speed = speed_from_previous_steering(
                steering, speed_steering_min, speed_steering_max, min_speed, max_speed,
            )
        elif speed_profile == "piecewise":
            speed = piecewise_speed_from_previous_steering(
                steering,
                straight_steering_threshold,
                gentle_steering_threshold,
                medium_steering_threshold,
                straight_speed,
                gentle_speed,
                medium_speed,
                corner_speed,
            )
        else:
            raise ValueError(f"Unsupported speed profile: {speed_profile}")
        header.append("speed")

    for index in range(len(waypoints)):
        row: list[object] = [
            index,
            distance[index],
            waypoints[index, 0],
            waypoints[index, 1],
            heading[index],
            kappa[index],
            steering[index],
        ]
        if spacing_rule is not None:
            row.append(spacing_rule[index])
        if turn_state is not None:
            row.append(turn_state[index])
        if speed is not None:
            row.append(speed[index])
        rows.append(row)

    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for row in rows:
            formatted: list[object] = [f"{value:d}" if isinstance(value, int) else value for value in row[:1]]
            for value in row[1:]:
                if isinstance(value, str):
                    formatted.append(value)
                else:
                    formatted.append(f"{float(value):.8f}")
            writer.writerow(formatted)


def save_overlay(path: Path, field: MapField, final_path: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 10))
    axis.imshow(field.image, cmap="gray", vmin=0, vmax=255, origin="upper",
                extent=field.extent)
    closed_final = np.vstack((final_path, final_path[0]))
    axis.plot(closed_final[:, 0], closed_final[:, 1], color="lime",
              linewidth=2.2, label="final smoothed 0.05 m path")
    axis.set_aspect("equal")
    axis.set_xlabel("map x [m]")
    axis.set_ylabel("map y [m]")
    axis.set_title("Cleaned map and final raceline")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    args.map_yaml = args.map_yaml.expanduser().resolve()
    args.raceline = args.raceline.expanduser().resolve()
    if not args.map_yaml.is_file():
        raise FileNotFoundError(f"Map YAML not found: {args.map_yaml}")
    if not args.raceline.is_file():
        raise FileNotFoundError(f"Raceline CSV not found: {args.raceline}")
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir
                  else args.raceline.parent / f"{args.raceline.stem}_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    with args.map_yaml.open() as stream:
        metadata = yaml.safe_load(stream)
    image_path = Path(metadata["image"])
    if not image_path.is_absolute():
        image_path = args.map_yaml.parent / image_path
    original_image = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
    cleaned_image, removed_components, removed_pixels = clean_map(
        original_image, args.wall_threshold, args.max_isolated_pixels,
    )

    map_stem = image_path.stem
    cleaned_stem = f"{map_stem}_cleaned"
    cleaned_pgm = output_dir / f"{cleaned_stem}.pgm"
    cleaned_png = output_dir / f"{cleaned_stem}.png"
    cleaned_yaml = output_dir / f"{cleaned_stem}.yaml"
    Image.fromarray(cleaned_image).save(cleaned_pgm)
    Image.fromarray(cleaned_image).save(cleaned_png)
    cleaned_metadata = dict(metadata)
    cleaned_metadata["image"] = cleaned_pgm.name
    with cleaned_yaml.open("w") as stream:
        yaml.safe_dump(cleaned_metadata, stream, sort_keys=False)

    field = MapField(cleaned_image, cleaned_metadata, args.wall_threshold)
    input_path = load_xy(args.raceline)
    registration_source = resample_closed(input_path, spacing=args.dense_spacing)
    oriented, angle, translation = orient_path(registration_source, field, args)
    loop_cleaned = erase_near_return_loops(
        oriented, args.loop_radius, args.max_local_loop,
    )
    loop_cleaned = erase_crossing_loops(loop_cleaned, args.max_local_loop)
    loop_cleaned = project_from_boundaries(loop_cleaned, field, args.boundary_clearance)
    dense = resample_closed(loop_cleaned, spacing=args.dense_spacing)
    dense = project_from_boundaries(dense, field, args.boundary_clearance)
    kappa = curvature(dense)
    steering = np.arctan(args.wheelbase * kappa)
    waypoints, _ = adaptive_waypoints(dense, steering)
    for _ in range(10):
        previous_count = len(waypoints)
        waypoints = insert_safety_points(
            waypoints, dense, field, args.boundary_clearance,
        )
        if len(waypoints) == previous_count:
            break

    raceline_stem = args.raceline.stem
    adaptive_csv = output_dir / f"{raceline_stem}_adaptive.csv"
    final_csv = output_dir / f"{raceline_stem}_final_0.05m.csv"
    final_overlay_png = output_dir / f"{raceline_stem}_final_0.05m_overlay.png"
    legacy_overlay_png = output_dir / f"{raceline_stem}_processed_overlay.png"
    # Do not leave older final artifacts behind if this run fails validation.
    final_csv.unlink(missing_ok=True)
    final_overlay_png.unlink(missing_ok=True)
    legacy_overlay_png.unlink(missing_ok=True)
    export_waypoints(adaptive_csv, waypoints, args.wheelbase)

    final_waypoints = smooth_with_steering_limit(
        waypoints, field, args.boundary_clearance, args.wheelbase,
        args.final_spacing, args.max_final_steering,
    )
    straight_samples = np.vstack([
        straight_segment(waypoints[index], waypoints[(index + 1) % len(waypoints)])
        for index in range(len(waypoints))
    ])
    minimum_clearance = float(np.min(field.sample(field.signed_distance, straight_samples)))
    final_samples = np.vstack([
        straight_segment(
            final_waypoints[index],
            final_waypoints[(index + 1) % len(final_waypoints)],
            spacing=min(0.01, args.final_spacing / 5.0),
        )
        for index in range(len(final_waypoints))
    ])
    final_clearance = float(np.min(field.sample(field.signed_distance, final_samples)))
    final_max_steering = float(np.max(np.abs(
        steering_angles(final_waypoints, args.wheelbase),
    )))
    print(f"cleaned map: {cleaned_yaml}")
    print(f"removed components: {removed_components} ({removed_pixels} pixels)")
    print(f"registration rotation [deg]: {math.degrees(angle):.4f}")
    print(f"registration translation [m]: {translation[0]:.4f}, {translation[1]:.4f}")
    print(f"input points: {len(input_path)}")
    print(f"adaptive output waypoints: {len(waypoints)}")
    print(f"{args.final_spacing:.2f} m output waypoints: {len(final_waypoints)}")
    print(f"minimum straight-segment boundary clearance [m]: {minimum_clearance:.4f}")
    print(f"minimum final-path boundary clearance [m]: {final_clearance:.4f}")
    print(f"maximum final steering [rad]: {final_max_steering:.6f}")
    print(f"adaptive raceline: {adaptive_csv}")
    if minimum_clearance + 1e-4 < args.boundary_clearance:
        print("WARNING: intermediate adaptive segments are below the configured "
              "boundary clearance; final-path validation still applies")
    if final_clearance + 1e-4 < args.boundary_clearance:
        print("ERROR: smoothed path does not satisfy the configured boundary clearance")
        return 2
    if final_max_steering >= args.max_final_steering:
        print("ERROR: smoothed path exceeds the configured steering limit")
        return 2
    export_waypoints(final_csv, final_waypoints, args.wheelbase,
                     include_spacing_rule=False, fixed_spacing=args.final_spacing,
                     include_turn_state=True,
                     speed_profile=args.speed_profile,
                     include_speed=True,
                     speed_steering_min=args.speed_steering_min,
                     speed_steering_max=args.speed_steering_max,
                     min_speed=args.min_speed, max_speed=args.max_speed,
                     straight_steering_threshold=args.straight_steering_threshold,
                     turn_angle_threshold_deg=args.turn_angle_threshold_deg,
                     turn_distance_threshold_m=args.turn_distance_threshold_m,
                     gentle_steering_threshold=args.gentle_steering_threshold,
                     medium_steering_threshold=args.medium_steering_threshold,
                     straight_speed=args.straight_speed,
                     gentle_speed=args.gentle_speed,
                     medium_speed=args.medium_speed,
                     corner_speed=args.corner_speed)
    save_overlay(final_overlay_png, field, final_waypoints)
    print(f"FINAL RACELINE PNG: {final_overlay_png}")
    # Keep the final line machine- and human-friendly: this is the requested
    # uniformly sampled CSV users should consume.
    print(f"FINAL RACELINE CSV: {final_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
