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


RACESTACK_REPO_ROOT = Path('/home/nvidia/spring26_ws/src/racestack')
RACELINE_GENERATION_REPO_ROOT = Path('/home/nvidia/spring26_ws/src/raceline_generation')
DEFAULT_MAP_DIR = RACESTACK_REPO_ROOT / 'maps'
DEFAULT_MANUAL_PATH_DIR = RACESTACK_REPO_ROOT / 'manual_paths'
DEFAULT_OUTPUT_DIR = RACESTACK_REPO_ROOT / 'paths'
DEFAULT_ARCHIVE_DIR = RACESTACK_REPO_ROOT / 'old_paths'
ACTIVE_RACELINE_FILENAME = 'raceline.csv'


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
    script = RACESTACK_REPO_ROOT / 'resource' / 'optimize_manual_map_raceline.py'
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
        speed_field = 'speed' if 'speed' in reader.fieldnames else None
        turn_state_field = 'turn_state' if 'turn_state' in reader.fieldnames else None

        with destination_csv.open('w', newline='') as destination_stream:
            writer = csv.writer(destination_stream, delimiter=';')
            header = ['s', 'x', 'y', 'yaw', 'kappa', 'v']
            if turn_state_field is not None:
                header.append('turn_state')
            writer.writerow(header)
            for row in reader:
                output_row = [
                    row['s'],
                    row['x'],
                    row['y'],
                    row['yaw'],
                    row['kappa'],
                    row[speed_field] if speed_field is not None else '10.00000000',
                ]
                if turn_state_field is not None:
                    output_row.append(row[turn_state_field])
                writer.writerow(output_row)


def _roll_points_to_match_seed(points: np.ndarray, seed_start: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    start_idx = int(np.argmin(np.linalg.norm(points - seed_start, axis=1)))
    return np.roll(points, -start_idx, axis=0)


class ProcessRacelineNode(Node):
    def __init__(self) -> None:
        super().__init__('process_raceline')

        self.declare_parameter('map_directory', str(DEFAULT_MAP_DIR))
        self.declare_parameter('manual_path_directory', str(DEFAULT_MANUAL_PATH_DIR))
        self.declare_parameter('output_directory', str(DEFAULT_OUTPUT_DIR))
        self.declare_parameter('archive_directory', str(DEFAULT_ARCHIVE_DIR))
        self.declare_parameter('max_isolated_pixels', 3)
        self.declare_parameter('wall_threshold', 100)
        self.declare_parameter('boundary_clearance', 0.35)
        self.declare_parameter('wheelbase', 0.324)
        self.declare_parameter('loop_radius', 0.12)
        self.declare_parameter('max_local_loop', 6.0)
        self.declare_parameter('search_translation', 1.0)
        self.declare_parameter('search_rotation_deg', 15.0)
        self.declare_parameter('dense_spacing', 0.03)
        self.declare_parameter('loop_cleanup_spacing', 0.15)
        self.declare_parameter('max_final_steering', 0.6)
        self.declare_parameter('final_spacing', 0.05)
        self.declare_parameter('speed_steering_min', 0.0)
        self.declare_parameter('speed_steering_max', 0.58)
        self.declare_parameter('min_speed', 1.0)
        self.declare_parameter('max_speed', 10.0)
        self.declare_parameter('speed_profile', 'piecewise')
        self.declare_parameter('straight_steering_threshold', 0.08)
        self.declare_parameter('turn_angle_threshold_deg', 30.0)
        self.declare_parameter('turn_distance_threshold_m', 1.5)
        self.declare_parameter('gentle_steering_threshold', 0.20)
        self.declare_parameter('medium_steering_threshold', 0.38)
        self.declare_parameter('straight_speed', 10.0)
        self.declare_parameter('gentle_speed', 6.0)
        self.declare_parameter('medium_speed', 4.0)
        self.declare_parameter('corner_speed', 1.5)
        self.declare_parameter('optimizer_spacing', 0.20)
        self.declare_parameter('optimizer_clearance', 0.35)
        self.declare_parameter('optimizer_max_offset', 0.60)
        self.declare_parameter('optimizer_racing_line_weight', 0.24)
        self.declare_parameter('optimizer_apex_bias', 0.82)
        self.declare_parameter('optimizer_outside_bias', 0.36)
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
            registration_source = processor.resample_closed(
                input_path,
                spacing=float(self.get_parameter('dense_spacing').value),
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
            optimizer_spacing = float(self.get_parameter('optimizer_spacing').value)
            optimizer_seed = optimizer.smooth_periodic(smoothed_centerline, final_spacing)
            optimizer_base = optimizer.smooth_periodic(optimizer_seed, optimizer_spacing)
            optimizer_normals = optimizer.normals_for(optimizer_base)
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
            )
            optimized_sparse = optimizer_base + optimizer_offsets[:, None] * optimizer_normals
            optimized_traj = optimizer.trajectory_from_path(
                optimized_sparse,
                optimizer_field,
                final_spacing,
                float(self.get_parameter('min_speed').value),
                float(self.get_parameter('max_speed').value),
                3.2,
                2.5,
                3.5,
            )
            optimized_points = optimizer.remove_duplicate_endpoint(optimized_traj[:, 1:3])
            optimized_points = _roll_points_to_match_seed(optimized_points, smoothed_centerline[0])
            optimized_points = processor.resample_closed_exact(optimized_points, final_spacing)
            optimized_vertex_clearance = boundary_clearance + 0.5 * final_spacing + 0.002
            optimized_clearance = float('-inf')
            for _ in range(8):
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
                include_turn_state=True,
                speed_profile=str(self.get_parameter('speed_profile').value),
                include_speed=True,
                speed_steering_min=float(self.get_parameter('speed_steering_min').value),
                speed_steering_max=float(self.get_parameter('speed_steering_max').value),
                min_speed=float(self.get_parameter('min_speed').value),
                max_speed=float(self.get_parameter('max_speed').value),
                straight_steering_threshold=float(self.get_parameter('straight_steering_threshold').value),
                turn_angle_threshold_deg=float(self.get_parameter('turn_angle_threshold_deg').value),
                turn_distance_threshold_m=float(self.get_parameter('turn_distance_threshold_m').value),
                gentle_steering_threshold=float(self.get_parameter('gentle_steering_threshold').value),
                medium_steering_threshold=float(self.get_parameter('medium_steering_threshold').value),
                straight_speed=float(self.get_parameter('straight_speed').value),
                gentle_speed=float(self.get_parameter('gentle_speed').value),
                medium_speed=float(self.get_parameter('medium_speed').value),
                corner_speed=float(self.get_parameter('corner_speed').value),
            )

            archived_paths = _archive_existing_outputs(output_directory, archive_directory)
            destination = output_directory / ACTIVE_RACELINE_FILENAME
            _normalize_final_csv(optimized_csv, destination)

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
                f'clearance={optimized_clearance:.4f} m.'
            )
            self.get_logger().info(
                f'Active processed raceline saved to {destination} as s;x;y;yaw;kappa;v.'
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
