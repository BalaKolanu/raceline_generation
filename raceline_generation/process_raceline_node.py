#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import math
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node


RACELINE_GENERATION_REPO_ROOT = Path('/home/nvidia/f1tenth_ws/src/raceline_generation')
DEFAULT_MAP_DIR = RACELINE_GENERATION_REPO_ROOT / 'maps'
DEFAULT_MANUAL_PATH_DIR = RACELINE_GENERATION_REPO_ROOT / 'manual_paths'
DEFAULT_OUTPUT_DIR = RACELINE_GENERATION_REPO_ROOT / 'paths'
DEFAULT_ARCHIVE_DIR = RACELINE_GENERATION_REPO_ROOT / 'old_paths'
DEFAULT_LANE_DIR = RACELINE_GENERATION_REPO_ROOT / 'lanes'
ACTIVE_RACELINE_FILENAME = 'raceline.csv'
ACTIVE_LANE_FILENAME = 'lane.csv'
DEFAULT_PREVIEW_PNG = DEFAULT_LANE_DIR / 'lane_preview.png'


def _find_single_file(directory: Path, patterns: tuple[str, ...], asset_name: str) -> Path:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(path for path in directory.glob(pattern) if path.is_file())
    unique_matches = sorted(set(matches))
    if not unique_matches:
        raise FileNotFoundError(
            f'No {asset_name} file found in {directory}. Expected exactly one matching '
            f'{", ".join(patterns)}.'
        )
    if len(unique_matches) > 1:
        raise RuntimeError(
            f'Multiple {asset_name} files found in {directory}: '
            f'{", ".join(str(path) for path in unique_matches)}. Expected exactly one.'
        )
    return unique_matches[0]


def _resolve_processor_script() -> Path:
    source_script = RACELINE_GENERATION_REPO_ROOT / 'resource' / 'process_raceline.py'
    if source_script.is_file():
        return source_script

    share_dir = Path(get_package_share_directory('raceline_generation'))
    installed_script = share_dir / 'resource' / 'process_raceline.py'
    if installed_script.is_file():
        return installed_script

    raise FileNotFoundError('Could not locate resource/process_raceline.py in source or install space.')


def _resolve_optimizer_script() -> Path:
    script = RACELINE_GENERATION_REPO_ROOT / 'resource' / 'optimize_manual_map_raceline.py'
    if script.is_file():
        return script
    raise FileNotFoundError(f'Could not locate optimizer script at {script}.')


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load module {module_name} from {module_path}.')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _unique_archive_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    candidate = base_dir / timestamp
    suffix = 1
    while candidate.exists():
        candidate = base_dir / f'{timestamp}_{suffix:02d}'
        suffix += 1
    return candidate


def _archive_existing_outputs(output_dir: Path, archive_root: Path) -> list[Path]:
    archive_root_resolved = archive_root.resolve()
    items_to_archive = sorted(
        path
        for path in output_dir.iterdir()
        if path.name != 'README.md' and path.resolve() != archive_root_resolved
    )
    if not items_to_archive:
        return []

    archive_dir = _unique_archive_dir(archive_root)
    archive_dir.mkdir(parents=True, exist_ok=False)
    archived_paths: list[Path] = []
    for path in items_to_archive:
        destination = archive_dir / path.name
        shutil.move(str(path), str(destination))
        archived_paths.append(destination)
    return archived_paths


def _normalize_final_csv(source_csv: Path, destination_csv: Path) -> None:
    with source_csv.open(newline='') as source_stream:
        reader = csv.DictReader(source_stream)
        if reader.fieldnames is None:
            raise ValueError(f'{source_csv} has no header row.')
        required_fields = {'s', 'x', 'y', 'yaw', 'kappa'}
        missing_fields = sorted(required_fields.difference(reader.fieldnames))
        if missing_fields:
            raise ValueError(
                f'{source_csv} is missing required columns: {", ".join(missing_fields)}.'
            )
        with destination_csv.open('w', newline='') as destination_stream:
            writer = csv.writer(destination_stream, delimiter=';', lineterminator='\n')
            writer.writerow(['s', 'x', 'y', 'yaw', 'kappa', 'segment'])
            for row in reader:
                writer.writerow([
                    row['s'],
                    row['x'],
                    row['y'],
                    row['yaw'],
                    row['kappa'],
                    'segment_0',
                ])


def _roll_points_to_match_seed(points: np.ndarray, seed_start: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    start_idx = int(np.argmin(np.linalg.norm(points - seed_start, axis=1)))
    return np.roll(points, -start_idx, axis=0)


def _load_lane_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]], np.ndarray, np.ndarray]:
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f'{path} has no header row.')
        required = {'center_x', 'center_y', 'normal_x', 'normal_y', 'w_left', 'w_right'}
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f'{path} is missing required lane columns: {", ".join(missing)}.')

        center = []
        normals = []
        bounds = []
        left = []
        right = []
        for row in reader:
            point = np.asarray([float(row['center_x']), float(row['center_y'])], dtype=np.float64)
            normal = np.asarray([float(row['normal_x']), float(row['normal_y'])], dtype=np.float64)
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-9:
                raise ValueError(f'{path} contains a zero normal near ({point[0]:.3f}, {point[1]:.3f}).')
            normal /= norm
            w_left = max(0.0, float(row['w_left']))
            w_right = max(0.0, float(row['w_right']))
            center.append(point)
            normals.append(normal)
            bounds.append((-w_left, w_right))
            left.append(point - w_left * normal)
            right.append(point + w_right * normal)

    if len(center) < 4:
        raise ValueError(f'{path} must contain at least four lane samples.')
    return (
        np.asarray(center, dtype=np.float64),
        np.asarray(normals, dtype=np.float64),
        bounds,
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
    )


def _clip_points_to_lane(
    points: np.ndarray,
    lane_center: np.ndarray,
    lane_normals: np.ndarray,
    lane_bounds: list[tuple[float, float]],
) -> np.ndarray:
    clipped = points.copy()
    bounds = np.asarray(lane_bounds, dtype=np.float64)
    for index, point in enumerate(clipped):
        nearest = int(np.argmin(np.linalg.norm(lane_center - point, axis=1)))
        normal = lane_normals[nearest]
        alpha = float(np.dot(point - lane_center[nearest], normal))
        clipped_alpha = float(np.clip(alpha, bounds[nearest, 0], bounds[nearest, 1]))
        clipped[index] = point + (clipped_alpha - alpha) * normal
    return clipped


def _straighten_sparse_straights(
    points: np.ndarray,
    straight_spans: list[np.ndarray],
    processor,
    field,
    lane_center: np.ndarray,
    lane_normals: np.ndarray,
    lane_bounds: list[tuple[float, float]],
    minimum_clearance: float,
    max_deviation: float,
    lane_margin: float,
    strength: float,
) -> tuple[np.ndarray, int]:
    if not straight_spans:
        return points, 0

    straightened = points.copy()
    bounds = np.asarray(lane_bounds, dtype=np.float64)
    changed = 0
    for indices in straight_spans:
        if len(indices) < 4:
            continue
        span_points = straightened[indices]
        chord = span_points[-1] - span_points[0]
        chord_length = float(np.linalg.norm(chord))
        if chord_length < 1e-6:
            continue

        relative = span_points - span_points[0]
        lateral_error = (
            relative[:, 0] * chord[1] - relative[:, 1] * chord[0]
        ) / chord_length
        if float(np.max(np.abs(lateral_error))) > max_deviation:
            continue

        station_steps = np.linalg.norm(
            lane_center[indices[1:]] - lane_center[indices[:-1]],
            axis=1,
        )
        cumulative = np.insert(np.cumsum(station_steps), 0, 0.0)
        if cumulative[-1] < 1e-6:
            continue
        phase = cumulative / cumulative[-1]
        chord_candidate = span_points[0] + phase[:, None] * chord
        blend = strength * np.sin(np.pi * phase)
        candidate = span_points + blend[:, None] * (chord_candidate - span_points)

        alphas = np.einsum(
            'ij,ij->i',
            candidate - lane_center[indices],
            lane_normals[indices],
        )
        span_bounds = bounds[indices]
        if np.any(alphas < span_bounds[:, 0] + lane_margin):
            continue
        if np.any(alphas > span_bounds[:, 1] - lane_margin):
            continue

        samples = np.vstack([
            processor.straight_segment(
                candidate[index],
                candidate[index + 1],
                spacing=0.01,
            )
            for index in range(len(candidate) - 1)
        ])
        clearance = float(np.min(field.sample(field.signed_distance, samples)))
        if clearance + 1e-4 < minimum_clearance:
            continue

        straightened[indices[1:-1]] = candidate[1:-1]
        changed += 1

    return straightened, changed


def _save_raceline_preview(
    path: Path,
    processor,
    field,
    lane_center: np.ndarray,
    lane_left: np.ndarray,
    lane_right: np.ndarray,
    optimized_points: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    center_display = np.vstack((lane_center, lane_center[0]))
    left_display = np.vstack((lane_left, lane_left[0]))
    right_display = np.vstack((lane_right, lane_right[0]))
    raceline_display = np.vstack((optimized_points, optimized_points[0]))

    fig, axis = plt.subplots(figsize=(10, 10))
    axis.imshow(
        field.image,
        cmap='gray',
        vmin=0,
        vmax=255,
        origin='upper',
        extent=field.extent,
    )
    axis.plot(left_display[:, 0], left_display[:, 1], color='#1f77b4', linewidth=1.2, label='left bound')
    axis.plot(right_display[:, 0], right_display[:, 1], color='#ff7f0e', linewidth=1.2, label='right bound')
    axis.plot(center_display[:, 0], center_display[:, 1], color='#00d12f', linewidth=1.3, alpha=0.75, label='lane center')
    axis.plot(raceline_display[:, 0], raceline_display[:, 1], color='#d62728', linewidth=2.0, label='optimized raceline')
    stride = max(1, len(lane_center) // 90)
    for index in range(0, len(lane_center), stride):
        axis.plot(
            [lane_left[index, 0], lane_right[index, 0]],
            [lane_left[index, 1], lane_right[index, 1]],
            color='#1f77b4',
            linewidth=0.6,
            alpha=0.20,
        )
    axis.set_aspect('equal', adjustable='box')
    axis.set_xlabel('map x [m]')
    axis.set_ylabel('map y [m]')
    axis.set_title('Lane-bounded optimized raceline')
    axis.grid(alpha=0.2)
    axis.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


class ProcessRacelineNode(Node):
    def __init__(self) -> None:
        super().__init__('process_raceline')

        self.declare_parameter('map_directory', str(DEFAULT_MAP_DIR))
        self.declare_parameter('manual_path_directory', str(DEFAULT_MANUAL_PATH_DIR))
        self.declare_parameter('output_directory', str(DEFAULT_OUTPUT_DIR))
        self.declare_parameter('archive_directory', str(DEFAULT_ARCHIVE_DIR))
        self.declare_parameter('lane_directory', str(DEFAULT_LANE_DIR))
        self.declare_parameter('lane_csv', '')
        self.declare_parameter('use_lane_bounds', True)
        self.declare_parameter('save_preview', True)
        self.declare_parameter('preview_png', str(DEFAULT_PREVIEW_PNG))
        self.declare_parameter('max_isolated_pixels', 3)
        self.declare_parameter('wall_threshold', 100)
        self.declare_parameter('boundary_clearance', 0.25)
        self.declare_parameter('wheelbase', 0.324)
        self.declare_parameter('loop_radius', 0.12)
        self.declare_parameter('max_local_loop', 6.0)
        self.declare_parameter('search_translation', 1.0)
        self.declare_parameter('search_rotation_deg', 15.0)
        self.declare_parameter('dense_spacing', 0.03)
        self.declare_parameter('loop_cleanup_spacing', 0.15)
        self.declare_parameter('max_final_steering', 0.6)
        self.declare_parameter('final_spacing', 0.05)
        self.declare_parameter('optimizer_spacing', 0.20)
        self.declare_parameter('optimizer_clearance', 0.25)
        self.declare_parameter('optimizer_preferred_clearance', 0.50)
        self.declare_parameter('optimizer_max_offset', 0.60)
        self.declare_parameter('optimizer_racing_line_weight', 0.24)
        self.declare_parameter('optimizer_apex_bias', 0.82)
        self.declare_parameter('optimizer_outside_bias', 0.36)
        self.declare_parameter('optimizer_straighten_straights', False)
        self.declare_parameter('optimizer_straighten_min_length', 1.20)
        self.declare_parameter('optimizer_straighten_max_deviation', 0.22)
        self.declare_parameter('optimizer_straighten_lane_margin', 0.01)
        self.declare_parameter('optimizer_straighten_strength', 0.15)
        self.declare_parameter('optimizer_max_iter', 300)

        self._startup_timer = self.create_timer(0.1, self._run_once)
        self._completed = False
        self._exit_code = 0

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def exit_code(self) -> int:
        return self._exit_code

    def _run_once(self) -> None:
        if self._completed:
            return
        self._startup_timer.cancel()
        try:
            self._process_raceline()
        except Exception as exc:  # pragma: no cover - runtime logging path
            self._exit_code = 1
            self.get_logger().error(str(exc))
        finally:
            self._completed = True

    def _process_raceline(self) -> None:
        map_directory = Path(str(self.get_parameter('map_directory').value)).expanduser().resolve()
        manual_path_directory = Path(
            str(self.get_parameter('manual_path_directory').value)
        ).expanduser().resolve()
        output_directory = Path(str(self.get_parameter('output_directory').value)).expanduser().resolve()
        archive_directory = Path(
            str(self.get_parameter('archive_directory').value)
        ).expanduser().resolve()
        lane_directory = Path(str(self.get_parameter('lane_directory').value)).expanduser().resolve()
        lane_csv_param = str(self.get_parameter('lane_csv').value)
        lane_csv = (
            Path(lane_csv_param).expanduser().resolve()
            if lane_csv_param
            else lane_directory / ACTIVE_LANE_FILENAME
        )
        use_lane_bounds = bool(self.get_parameter('use_lane_bounds').value)
        preview_png = Path(str(self.get_parameter('preview_png').value)).expanduser().resolve()

        map_yaml = _find_single_file(map_directory, ('*.yaml', '*.yml'), 'map')
        manual_path = _find_single_file(manual_path_directory, ('*.csv',), 'manual path')
        processor_script = _resolve_processor_script()
        optimizer_script = _resolve_optimizer_script()
        processor = _load_module('raceline_generation_process_raceline', processor_script)
        optimizer = _load_module('racestack_bala_raceline_optimizer', optimizer_script)

        output_directory.mkdir(parents=True, exist_ok=True)
        archive_directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='raceline_generation_process_raceline_') as temp_dir:
            temp_output_dir = Path(temp_dir)
            self.get_logger().info(
                f'Processing manual path {manual_path} against map {map_yaml}.'
            )

            with map_yaml.open() as stream:
                metadata = processor.yaml.safe_load(stream)
            image_path = Path(metadata['image'])
            if not image_path.is_absolute():
                image_path = map_yaml.parent / image_path
            original_image = np.asarray(processor.Image.open(image_path).convert('L'), dtype=np.uint8)
            cleaned_image, removed_components, removed_pixels = processor.clean_map(
                original_image,
                int(self.get_parameter('wall_threshold').value),
                int(self.get_parameter('max_isolated_pixels').value),
            )
            cleaned_pgm = temp_output_dir / f'{image_path.stem}_cleaned.pgm'
            cleaned_yaml = temp_output_dir / f'{image_path.stem}_cleaned.yaml'
            processor.Image.fromarray(cleaned_image).save(cleaned_pgm)
            cleaned_metadata = dict(metadata)
            cleaned_metadata['image'] = cleaned_pgm.name
            with cleaned_yaml.open('w') as stream:
                processor.yaml.safe_dump(cleaned_metadata, stream, sort_keys=False)

            boundary_clearance = float(self.get_parameter('boundary_clearance').value)
            wheelbase = float(self.get_parameter('wheelbase').value)
            final_spacing = float(self.get_parameter('final_spacing').value)
            field = processor.MapField(
                cleaned_image,
                cleaned_metadata,
                int(self.get_parameter('wall_threshold').value),
            )
            input_path = processor.load_xy(manual_path)
            self.get_logger().info(
                f'Loaded manual path with {len(input_path)} points. '
                f'Using minimum_clearance={boundary_clearance:.2f} m.'
            )
            registration_source = processor.resample_closed(
                input_path,
                spacing=float(self.get_parameter('dense_spacing').value),
            )
            self.get_logger().info(
                f'Resampled registration seed to {len(registration_source)} points '
                f'at dense_spacing={float(self.get_parameter("dense_spacing").value):.2f} m.'
            )
            oriented, angle, translation = processor.orient_path(
                registration_source,
                field,
                type(
                    'ProcessorArgs',
                    (),
                    {
                        'boundary_clearance': boundary_clearance,
                        'search_translation': float(self.get_parameter('search_translation').value),
                        'search_rotation_deg': float(self.get_parameter('search_rotation_deg').value),
                    },
                )(),
            )
            self.get_logger().info(
                'Finished path registration; beginning loop cleanup and wall projection.'
            )
            loop_cleanup_spacing = max(
                float(self.get_parameter('loop_cleanup_spacing').value),
                float(self.get_parameter('dense_spacing').value),
            )
            loop_cleanup_source = processor.resample_closed(
                oriented,
                spacing=loop_cleanup_spacing,
            )
            loop_cleaned = processor.erase_near_return_loops(
                loop_cleanup_source,
                float(self.get_parameter('loop_radius').value),
                float(self.get_parameter('max_local_loop').value),
            )
            loop_cleaned = processor.erase_crossing_loops(
                loop_cleaned,
                float(self.get_parameter('max_local_loop').value),
            )
            loop_cleaned = processor.project_from_boundaries(
                loop_cleaned,
                field,
                boundary_clearance,
            )
            dense = processor.resample_closed(
                loop_cleaned,
                spacing=float(self.get_parameter('dense_spacing').value),
            )
            dense = processor.project_from_boundaries(dense, field, boundary_clearance)
            self.get_logger().info(
                f'Generated dense projected path with {len(dense)} points; '
                'building adaptive waypoint set.'
            )
            steering = np.arctan(wheelbase * processor.curvature(dense))
            waypoints, _ = processor.adaptive_waypoints(dense, steering)
            for _ in range(10):
                previous_count = len(waypoints)
                waypoints = processor.insert_safety_points(
                    waypoints,
                    dense,
                    field,
                    boundary_clearance,
                )
                if len(waypoints) == previous_count:
                    break
            self.get_logger().info(
                f'Adaptive waypoint stage complete with {len(waypoints)} points; '
                'starting steering-limited smoothing.'
            )

            smoothed_centerline = processor.smooth_with_steering_limit(
                waypoints,
                field,
                boundary_clearance,
                wheelbase,
                final_spacing,
                float(self.get_parameter('max_final_steering').value),
            )

            straight_samples = np.vstack([
                processor.straight_segment(
                    smoothed_centerline[index],
                    smoothed_centerline[(index + 1) % len(smoothed_centerline)],
                    spacing=min(0.01, final_spacing / 5.0),
                )
                for index in range(len(smoothed_centerline))
            ])
            centerline_clearance = float(np.min(field.sample(field.signed_distance, straight_samples)))
            centerline_max_steering = float(np.max(np.abs(
                processor.steering_angles(smoothed_centerline, wheelbase),
            )))
            if centerline_clearance + 1e-4 < boundary_clearance:
                raise RuntimeError(
                    'Smoothed centerline does not satisfy the configured boundary clearance '
                    f'({centerline_clearance:.4f} m < {boundary_clearance:.4f} m).'
                )
            if centerline_max_steering >= float(self.get_parameter('max_final_steering').value):
                raise RuntimeError(
                    'Smoothed centerline exceeds the configured steering limit '
                    f'({centerline_max_steering:.6f} rad).'
                )

            optimizer_field = optimizer.MapField(cleaned_yaml)
            optimizer_clearance = float(self.get_parameter('optimizer_clearance').value)
            optimizer_preferred_clearance = float(
                self.get_parameter('optimizer_preferred_clearance').value
            )
            optimizer_spacing = float(self.get_parameter('optimizer_spacing').value)
            lane_left = None
            lane_right = None
            if use_lane_bounds:
                if not lane_csv.is_file():
                    raise FileNotFoundError(
                        f'Lane CSV not found at {lane_csv}. Run generate_lanes first or set use_lane_bounds:=false.'
                    )
                optimizer_base, optimizer_normals, optimizer_bounds, lane_left, lane_right = _load_lane_csv(lane_csv)
                optimizer_spacing = float(
                    np.mean(np.linalg.norm(np.roll(optimizer_base, -1, axis=0) - optimizer_base, axis=1))
                )
                self.get_logger().info(
                    f'Loaded lane bounds from {lane_csv}; launching optimizer on {len(optimizer_base)} '
                    f'lane stations.'
                )
            else:
                optimizer_seed = optimizer.smooth_periodic(smoothed_centerline, final_spacing)
                optimizer_base = optimizer.smooth_periodic(optimizer_seed, optimizer_spacing)
                optimizer_base = processor.project_from_boundaries(
                    optimizer_base,
                    field,
                    optimizer_clearance,
                )
                optimizer_normals = optimizer.normals_for(optimizer_base)
                self.get_logger().info(
                    f'Smoothed centerline valid; launching optimizer on {len(optimizer_base)} base points '
                    f'with minimum_clearance={optimizer_clearance:.2f} m.'
                )
                optimizer_bounds = optimizer.offset_bounds(
                    optimizer_field,
                    optimizer_base,
                    optimizer_normals,
                    optimizer_clearance,
                    float(self.get_parameter('optimizer_max_offset').value),
                )
            optimizer_offsets = optimizer.optimize_offsets(
                optimizer_base,
                optimizer_normals,
                optimizer_bounds,
                int(self.get_parameter('optimizer_max_iter').value),
                float(self.get_parameter('optimizer_racing_line_weight').value),
                float(self.get_parameter('optimizer_apex_bias').value),
                float(self.get_parameter('optimizer_outside_bias').value),
                optimizer_field,
                optimizer_preferred_clearance,
            )
            optimized_sparse = optimizer_base + optimizer_offsets[:, None] * optimizer_normals
            if use_lane_bounds and bool(self.get_parameter('optimizer_straighten_straights').value):
                straight_spans = optimizer.detect_straight_spans(
                    optimizer_base,
                    min_length=float(self.get_parameter('optimizer_straighten_min_length').value),
                )
                optimized_sparse, straightened_count = _straighten_sparse_straights(
                    optimized_sparse,
                    straight_spans,
                    processor,
                    field,
                    optimizer_base,
                    optimizer_normals,
                    optimizer_bounds,
                    boundary_clearance,
                    float(self.get_parameter('optimizer_straighten_max_deviation').value),
                    float(self.get_parameter('optimizer_straighten_lane_margin').value),
                    float(self.get_parameter('optimizer_straighten_strength').value),
                )
                if straightened_count:
                    self.get_logger().info(
                        f'Straightened {straightened_count} lane-bounded straight span(s).'
                    )
            optimized_points = optimizer.path_from_offsets(
                optimized_sparse,
                optimizer_field,
                final_spacing,
            )
            optimized_points = _roll_points_to_match_seed(optimized_points, optimizer_base[0])
            optimized_points = processor.resample_closed_exact(optimized_points, final_spacing)
            optimized_vertex_clearance = boundary_clearance + 0.5 * final_spacing + 0.002
            optimized_clearance = float('-inf')
            for _ in range(8):
                if use_lane_bounds:
                    optimized_points = _clip_points_to_lane(
                        optimized_points,
                        optimizer_base,
                        optimizer_normals,
                        optimizer_bounds,
                    )
                optimized_points = processor.project_from_boundaries(
                    optimized_points,
                    field,
                    optimized_vertex_clearance,
                )
                optimized_points = processor.resample_closed_exact(optimized_points, final_spacing)
                optimized_samples = np.vstack([
                    processor.straight_segment(
                        optimized_points[index],
                        optimized_points[(index + 1) % len(optimized_points)],
                        spacing=min(0.01, final_spacing / 5.0),
                    )
                    for index in range(len(optimized_points))
                ])
                optimized_clearance = float(np.min(field.sample(field.signed_distance, optimized_samples)))
                if optimized_clearance + 1e-4 >= boundary_clearance:
                    break

            if use_lane_bounds:
                optimized_points = _clip_points_to_lane(
                    optimized_points,
                    optimizer_base,
                    optimizer_normals,
                    optimizer_bounds,
                )
                optimized_points = processor.resample_closed_exact(optimized_points, final_spacing)

            optimized_samples = np.vstack([
                processor.straight_segment(
                    optimized_points[index],
                    optimized_points[(index + 1) % len(optimized_points)],
                    spacing=min(0.01, final_spacing / 5.0),
                )
                for index in range(len(optimized_points))
            ])
            optimized_clearance = float(np.min(field.sample(field.signed_distance, optimized_samples)))
            if optimized_clearance + 1e-4 < boundary_clearance:
                raise RuntimeError(
                    'Optimized raceline does not satisfy the configured boundary clearance '
                    f'({optimized_clearance:.4f} m < {boundary_clearance:.4f} m).'
                )
            optimized_csv = temp_output_dir / ACTIVE_RACELINE_FILENAME
            processor.export_waypoints(
                optimized_csv,
                optimized_points,
                wheelbase,
                include_spacing_rule=False,
                fixed_spacing=final_spacing,
                include_turn_state=False,
                include_segment=True,
                include_speed=False,
            )

            archived_paths = _archive_existing_outputs(output_directory, archive_directory)
            destination = output_directory / ACTIVE_RACELINE_FILENAME
            _normalize_final_csv(optimized_csv, destination)
            if bool(self.get_parameter('save_preview').value) and lane_left is not None and lane_right is not None:
                _save_raceline_preview(
                    preview_png,
                    processor,
                    field,
                    optimizer_base,
                    lane_left,
                    lane_right,
                    optimized_points,
                )

            if archived_paths:
                self.get_logger().info(
                    f'Archived {len(archived_paths)} existing item(s) into {archived_paths[0].parent}.'
                )
            self.get_logger().info(
                f'Removed isolated map components: {removed_components} ({removed_pixels} pixels).'
            )
            self.get_logger().info(
                'Manual path registration adjustment: '
                f'rotation={math.degrees(angle):.4f} deg '
                f'translation=({translation[0]:.4f}, {translation[1]:.4f}) m.'
            )
            self.get_logger().info(
                f'Smoothed centerline points: {len(smoothed_centerline)} '
                f'clearance={centerline_clearance:.4f} m.'
            )
            self.get_logger().info(
                f'Optimized raceline points: {len(optimized_points)} '
                f'clearance={optimized_clearance:.4f} m '
                f'(minimum={boundary_clearance:.2f} m).'
            )
            if bool(self.get_parameter('save_preview').value) and lane_left is not None and lane_right is not None:
                self.get_logger().info(f'Lane-bounded raceline preview saved to {preview_png}.')
            self.get_logger().info(
                f'Active processed raceline saved to {destination} as s;x;y;yaw;kappa;segment.'
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ProcessRacelineNode()
    try:
        while rclpy.ok() and not node.completed:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if node.exit_code != 0:
        raise SystemExit(node.exit_code)


if __name__ == '__main__':
    main()
