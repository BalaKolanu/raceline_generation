#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node


RACESTACK_REPO_ROOT = Path('/home/nvidia/spring26_ws/src/racestack')
RACELINE_GENERATION_REPO_ROOT = Path('/home/nvidia/spring26_ws/src/raceline_generation')
DEFAULT_MAP_DIR = RACESTACK_REPO_ROOT / 'maps'
DEFAULT_MANUAL_PATH_DIR = RACESTACK_REPO_ROOT / 'manual_paths'
DEFAULT_OUTPUT_DIR = RACESTACK_REPO_ROOT / 'paths'
DEFAULT_ARCHIVE_DIR = RACESTACK_REPO_ROOT / 'old_paths'


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


def _extract_final_csv(stdout: str) -> Path:
    for line in reversed(stdout.splitlines()):
        if line.startswith('FINAL RACELINE CSV:'):
            return Path(line.split(':', 1)[1].strip())
    raise RuntimeError('Processor completed without reporting a final raceline CSV.')


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

        with destination_csv.open('w', newline='') as destination_stream:
            writer = csv.writer(destination_stream, delimiter=';')
            writer.writerow(['s', 'x', 'y', 'yaw', 'kappa', 'v'])
            for row in reader:
                writer.writerow([
                    row['s'],
                    row['x'],
                    row['y'],
                    row['yaw'],
                    row['kappa'],
                    row[speed_field] if speed_field is not None else '10.00000000',
                ])


class ProcessRacelineNode(Node):
    def __init__(self) -> None:
        super().__init__('process_raceline')

        self.declare_parameter('map_directory', str(DEFAULT_MAP_DIR))
        self.declare_parameter('manual_path_directory', str(DEFAULT_MANUAL_PATH_DIR))
        self.declare_parameter('output_directory', str(DEFAULT_OUTPUT_DIR))
        self.declare_parameter('archive_directory', str(DEFAULT_ARCHIVE_DIR))
        self.declare_parameter('python_executable', sys.executable)
        self.declare_parameter('max_isolated_pixels', 3)
        self.declare_parameter('wall_threshold', 100)
        self.declare_parameter('boundary_clearance', 0.25)
        self.declare_parameter('wheelbase', 0.25)
        self.declare_parameter('loop_radius', 0.12)
        self.declare_parameter('max_local_loop', 6.0)
        self.declare_parameter('search_translation', 1.0)
        self.declare_parameter('search_rotation_deg', 15.0)
        self.declare_parameter('dense_spacing', 0.03)
        self.declare_parameter('max_final_steering', 0.6)
        self.declare_parameter('final_spacing', 0.05)
        self.declare_parameter('speed_steering_min', 0.0)
        self.declare_parameter('speed_steering_max', 0.58)
        self.declare_parameter('min_speed', 1.0)
        self.declare_parameter('max_speed', 10.0)
        self.declare_parameter('speed_profile', 'piecewise')
        self.declare_parameter('straight_steering_threshold', 0.08)
        self.declare_parameter('gentle_steering_threshold', 0.20)
        self.declare_parameter('medium_steering_threshold', 0.38)
        self.declare_parameter('straight_speed', 10.0)
        self.declare_parameter('gentle_speed', 6.0)
        self.declare_parameter('medium_speed', 4.0)
        self.declare_parameter('corner_speed', 1.5)

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
        python_executable = str(self.get_parameter('python_executable').value)

        map_yaml = _find_single_file(map_directory, ('*.yaml', '*.yml'), 'map')
        manual_path = _find_single_file(manual_path_directory, ('*.csv',), 'manual path')
        processor_script = _resolve_processor_script()

        output_directory.mkdir(parents=True, exist_ok=True)
        archive_directory.mkdir(parents=True, exist_ok=True)

        command = [
            python_executable,
            str(processor_script),
            str(map_yaml),
            str(manual_path),
            '--output-dir',
        ]
        with tempfile.TemporaryDirectory(prefix='raceline_generation_process_raceline_') as temp_dir:
            temp_output_dir = Path(temp_dir)
            command.append(str(temp_output_dir))
            for flag, parameter_name in (
                ('--max-isolated-pixels', 'max_isolated_pixels'),
                ('--wall-threshold', 'wall_threshold'),
                ('--boundary-clearance', 'boundary_clearance'),
                ('--wheelbase', 'wheelbase'),
                ('--loop-radius', 'loop_radius'),
                ('--max-local-loop', 'max_local_loop'),
                ('--search-translation', 'search_translation'),
                ('--search-rotation-deg', 'search_rotation_deg'),
                ('--dense-spacing', 'dense_spacing'),
                ('--max-final-steering', 'max_final_steering'),
                ('--final-spacing', 'final_spacing'),
                ('--speed-steering-min', 'speed_steering_min'),
                ('--speed-steering-max', 'speed_steering_max'),
                ('--min-speed', 'min_speed'),
                ('--max-speed', 'max_speed'),
                ('--speed-profile', 'speed_profile'),
                ('--straight-steering-threshold', 'straight_steering_threshold'),
                ('--gentle-steering-threshold', 'gentle_steering_threshold'),
                ('--medium-steering-threshold', 'medium_steering_threshold'),
                ('--straight-speed', 'straight_speed'),
                ('--gentle-speed', 'gentle_speed'),
                ('--medium-speed', 'medium_speed'),
                ('--corner-speed', 'corner_speed'),
            ):
                command.extend((flag, str(self.get_parameter(parameter_name).value)))

            self.get_logger().info(
                f'Processing manual path {manual_path} against map {map_yaml}.'
            )
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.stdout.strip():
                for line in result.stdout.strip().splitlines():
                    self.get_logger().info(line)
            if result.stderr.strip():
                for line in result.stderr.strip().splitlines():
                    self.get_logger().warn(line)
            if result.returncode != 0:
                raise RuntimeError(
                    f'process_raceline.py failed with exit code {result.returncode}. '
                    'The previous active path was left untouched.'
                )

            final_csv = _extract_final_csv(result.stdout)
            if not final_csv.is_file():
                raise FileNotFoundError(f'Final raceline CSV not found: {final_csv}')

            archived_paths = _archive_existing_outputs(output_directory, archive_directory)
            destination = output_directory / final_csv.name
            _normalize_final_csv(final_csv, destination)

            if archived_paths:
                self.get_logger().info(
                    f'Archived {len(archived_paths)} existing item(s) into {archived_paths[0].parent}.'
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
