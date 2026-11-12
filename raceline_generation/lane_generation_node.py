#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy


RACELINE_GENERATION_REPO_ROOT = Path('/home/nvidia/f1tenth_ws/src/raceline_generation')
DEFAULT_MAP_DIR = RACELINE_GENERATION_REPO_ROOT / 'maps'
DEFAULT_MANUAL_PATH_DIR = RACELINE_GENERATION_REPO_ROOT / 'manual_paths'
DEFAULT_OUTPUT_DIR = RACELINE_GENERATION_REPO_ROOT / 'lanes'
ACTIVE_LANE_FILENAME = 'lane.csv'
LANE_PREVIEW_FILENAME = 'lane_preview.png'


@dataclass(frozen=True)
class LaneData:
    center: np.ndarray
    left: np.ndarray
    right: np.ndarray
    left_width: np.ndarray
    right_width: np.ndarray
    normals: np.ndarray
    center_clearance: np.ndarray


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


def _path_msg(points: np.ndarray, frame_id: str, stamp_msg) -> PathMsg:
    msg = PathMsg()
    msg.header.stamp = stamp_msg
    msg.header.frame_id = frame_id
    display_points = np.vstack((points, points[0])) if len(points) > 2 else points

    for index, point in enumerate(display_points):
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose.position.x = float(point[0])
        pose.pose.position.y = float(point[1])
        pose.pose.position.z = 0.0

        next_index = min(index + 1, len(display_points) - 1)
        prev_index = max(index - 1, 0)
        if len(display_points) > 1:
            delta = display_points[next_index] - display_points[prev_index]
            yaw = math.atan2(float(delta[1]), float(delta[0]))
            pose.pose.orientation.z = math.sin(0.5 * yaw)
            pose.pose.orientation.w = math.cos(0.5 * yaw)
        else:
            pose.pose.orientation.w = 1.0
        msg.poses.append(pose)
    return msg


def _smooth_closed_values(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values.copy()
    if window % 2 == 0:
        window += 1
    window = min(window, len(values) if len(values) % 2 == 1 else len(values) - 1)
    if window < 3:
        return values.copy()
    kernel = np.hanning(window)
    kernel_sum = float(np.sum(kernel))
    if kernel_sum <= 1e-9:
        return values.copy()
    kernel /= kernel_sum
    pad = window // 2
    padded = np.r_[values[-pad:], values, values[:pad]]
    return np.convolve(padded, kernel, mode='valid')


def _rolling_min_closed(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values.copy()
    if window % 2 == 0:
        window += 1
    window = min(window, len(values) if len(values) % 2 == 1 else len(values) - 1)
    if window < 3:
        return values.copy()
    pad = window // 2
    padded = np.r_[values[-pad:], values, values[:pad]]
    out = np.empty_like(values)
    for index in range(len(values)):
        out[index] = float(np.min(padded[index:index + window]))
    return out


def _limit_closed_growth(values: np.ndarray, max_growth: float, passes: int = 3) -> np.ndarray:
    if max_growth <= 0.0 or len(values) < 3:
        return values.copy()
    limited = values.copy()
    n = len(limited)
    start = int(np.argmin(limited))
    for _ in range(max(1, passes)):
        for step in range(1, n + 1):
            idx = (start + step) % n
            prev_idx = (idx - 1) % n
            limited[idx] = min(limited[idx], limited[prev_idx] + max_growth)
        for step in range(1, n + 1):
            idx = (start - step) % n
            next_idx = (idx + 1) % n
            limited[idx] = min(limited[idx], limited[next_idx] + max_growth)
    return limited


def _offset_bounds_allowing_recenter(
    field,
    base: np.ndarray,
    normals: np.ndarray,
    clearance: float,
    max_offset: float,
    sample_spacing: float = 0.02,
) -> list[tuple[float, float]]:
    samples = np.linspace(
        -max_offset,
        max_offset,
        int(max_offset / sample_spacing) * 2 + 1,
    )
    zero_idx = int(np.argmin(np.abs(samples)))
    bounds: list[tuple[float, float]] = []

    for point, normal in zip(base, normals):
        candidates = point[None, :] + samples[:, None] * normal[None, :]
        distances = field.distance_at(candidates)
        safe = distances >= clearance
        if not np.any(safe):
            best_idx = int(np.argmax(distances))
            best_point = candidates[best_idx]
            raise ValueError(
                'No legal lane strip found near '
                f'({point[0]:.2f}, {point[1]:.2f}); '
                f'best distance={distances[best_idx]:.3f}m, required={clearance:.3f}m.'
            )

        runs: list[tuple[int, int]] = []
        idx = 0
        while idx < len(samples):
            if not safe[idx]:
                idx += 1
                continue
            start = idx
            while idx + 1 < len(samples) and safe[idx + 1]:
                idx += 1
            runs.append((start, idx))
            idx += 1

        containing_zero = [run for run in runs if run[0] <= zero_idx <= run[1]]
        if containing_zero:
            lo_idx, hi_idx = containing_zero[0]
        else:
            lo_idx, hi_idx = min(
                runs,
                key=lambda run: (
                    min(abs(samples[run[0]]), abs(samples[run[1]])),
                    -abs(samples[run[1]] - samples[run[0]]),
                ),
            )
        bounds.append((float(samples[lo_idx]), float(samples[hi_idx])))

    return bounds


class LaneGenerationNode(Node):
    def __init__(self) -> None:
        super().__init__('lane_generation')

        self.declare_parameter('map_directory', str(DEFAULT_MAP_DIR))
        self.declare_parameter('manual_path_directory', str(DEFAULT_MANUAL_PATH_DIR))
        self.declare_parameter('output_directory', str(DEFAULT_OUTPUT_DIR))
        self.declare_parameter('lane_csv', '')
        self.declare_parameter('lane_preview_png', '')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('publish_lanes', True)
        self.declare_parameter('publish_boundary_paths', False)
        self.declare_parameter('center_path_topic', '/lane/center')
        self.declare_parameter('left_path_topic', '/lane/left_boundary')
        self.declare_parameter('right_path_topic', '/lane/right_boundary')
        self.declare_parameter('max_isolated_pixels', 3)
        self.declare_parameter('wall_threshold', 100)
        self.declare_parameter('boundary_clearance', 0.25)
        self.declare_parameter('lane_spacing', 0.20)
        self.declare_parameter('lane_center_smoothing', 4.0)
        self.declare_parameter('dense_spacing', 0.03)
        self.declare_parameter('loop_cleanup_spacing', 0.15)
        self.declare_parameter('loop_radius', 0.12)
        self.declare_parameter('max_local_loop', 6.0)
        self.declare_parameter('search_translation', 1.0)
        self.declare_parameter('search_rotation_deg', 15.0)
        self.declare_parameter('wheelbase', 0.324)
        self.declare_parameter('max_centerline_steering', 0.6)
        self.declare_parameter('lane_max_offset', 1.20)
        self.declare_parameter('lane_recenter_iterations', 4)
        self.declare_parameter('lane_width_envelope_window', 3)
        self.declare_parameter('lane_width_max_growth_per_station', 0.08)
        self.declare_parameter('lane_width_smoothing_window', 15)
        self.declare_parameter('save_preview', True)

        path_qos = QoSProfile(depth=1)
        path_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        path_qos.reliability = ReliabilityPolicy.RELIABLE

        self.center_pub = self.create_publisher(
            PathMsg,
            str(self.get_parameter('center_path_topic').value),
            path_qos,
        )
        self.left_pub = self.create_publisher(
            PathMsg,
            str(self.get_parameter('left_path_topic').value),
            path_qos,
        )
        self.right_pub = self.create_publisher(
            PathMsg,
            str(self.get_parameter('right_path_topic').value),
            path_qos,
        )

        self._timer = self.create_timer(0.1, self._run_once)
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
        self._timer.cancel()
        try:
            self._generate_lanes()
        except Exception as exc:  # pragma: no cover - runtime logging path
            self._exit_code = 1
            self.get_logger().error(str(exc))
        finally:
            self._completed = True

    def _generate_lanes(self) -> None:
        map_directory = Path(str(self.get_parameter('map_directory').value)).expanduser().resolve()
        manual_path_directory = Path(
            str(self.get_parameter('manual_path_directory').value)
        ).expanduser().resolve()
        output_directory = Path(str(self.get_parameter('output_directory').value)).expanduser().resolve()

        lane_csv_param = str(self.get_parameter('lane_csv').value)
        preview_param = str(self.get_parameter('lane_preview_png').value)
        lane_csv = Path(lane_csv_param).expanduser().resolve() if lane_csv_param else output_directory / ACTIVE_LANE_FILENAME
        preview_png = (
            Path(preview_param).expanduser().resolve()
            if preview_param
            else output_directory / LANE_PREVIEW_FILENAME
        )

        map_yaml = _find_single_file(map_directory, ('*.yaml', '*.yml'), 'map')
        manual_path = _find_single_file(manual_path_directory, ('*.csv',), 'manual path')
        processor = _load_module('raceline_generation_process_raceline_for_lanes', _resolve_processor_script())
        optimizer = _load_module('racestack_lane_bounds_optimizer', _resolve_optimizer_script())

        output_directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='lane_generation_') as temp_dir:
            temp_output_dir = Path(temp_dir)
            cleaned_yaml, cleaned_image, cleaned_metadata, removed_components, removed_pixels = self._clean_map(
                processor,
                map_yaml,
                temp_output_dir,
            )
            lane = self._build_lane(
                processor,
                optimizer,
                cleaned_yaml,
                cleaned_image,
                cleaned_metadata,
                manual_path,
            )

            self._write_lane_csv(lane_csv, lane)
            if bool(self.get_parameter('save_preview').value):
                self._save_preview(processor, cleaned_image, cleaned_metadata, lane, preview_png)
            if bool(self.get_parameter('publish_lanes').value):
                self._publish_lane(lane)

        self.get_logger().info(
            f'Generated lane from manual seed {manual_path} against map {map_yaml}.'
        )
        self.get_logger().info(
            f'Removed isolated map components: {removed_components} ({removed_pixels} pixels).'
        )
        self.get_logger().info(
            'Lane widths: '
            f'left={float(np.min(lane.left_width)):.3f}-{float(np.max(lane.left_width)):.3f} m, '
            f'right={float(np.min(lane.right_width)):.3f}-{float(np.max(lane.right_width)):.3f} m.'
        )
        self.get_logger().info(f'Lane CSV saved to {lane_csv}.')
        if bool(self.get_parameter('save_preview').value):
            self.get_logger().info(f'Lane preview saved to {preview_png}.')

    def _clean_map(self, processor, map_yaml: Path, temp_output_dir: Path):
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
        return cleaned_yaml, cleaned_image, cleaned_metadata, removed_components, removed_pixels

    def _build_lane(
        self,
        processor,
        optimizer,
        cleaned_yaml: Path,
        cleaned_image: np.ndarray,
        cleaned_metadata: dict,
        manual_path: Path,
    ) -> LaneData:
        boundary_clearance = float(self.get_parameter('boundary_clearance').value)
        dense_spacing = float(self.get_parameter('dense_spacing').value)
        lane_spacing = float(self.get_parameter('lane_spacing').value)
        wheelbase = float(self.get_parameter('wheelbase').value)

        processor_field = processor.MapField(
            cleaned_image,
            cleaned_metadata,
            int(self.get_parameter('wall_threshold').value),
        )
        input_path = processor.load_xy(manual_path)
        registration_source = processor.resample_closed(input_path, spacing=dense_spacing)
        oriented, angle, translation = processor.orient_path(
            registration_source,
            processor_field,
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
            'Manual seed registration adjustment: '
            f'rotation={math.degrees(angle):.4f} deg '
            f'translation=({translation[0]:.4f}, {translation[1]:.4f}) m.'
        )

        loop_cleanup_spacing = max(float(self.get_parameter('loop_cleanup_spacing').value), dense_spacing)
        loop_source = processor.resample_closed(oriented, spacing=loop_cleanup_spacing)
        loop_cleaned = processor.erase_near_return_loops(
            loop_source,
            float(self.get_parameter('loop_radius').value),
            float(self.get_parameter('max_local_loop').value),
        )
        loop_cleaned = processor.erase_crossing_loops(
            loop_cleaned,
            float(self.get_parameter('max_local_loop').value),
        )
        loop_cleaned = processor.project_from_boundaries(loop_cleaned, processor_field, boundary_clearance)
        dense = processor.resample_closed(loop_cleaned, spacing=dense_spacing)
        dense = processor.project_from_boundaries(dense, processor_field, boundary_clearance)
        steering = np.arctan(wheelbase * processor.curvature(dense))
        waypoints, _ = processor.adaptive_waypoints(dense, steering)
        for _ in range(10):
            previous_count = len(waypoints)
            waypoints = processor.insert_safety_points(
                waypoints,
                dense,
                processor_field,
                boundary_clearance,
            )
            if len(waypoints) == previous_count:
                break

        centerline = processor.smooth_with_steering_limit(
            waypoints,
            processor_field,
            boundary_clearance,
            wheelbase,
            min(dense_spacing, 0.05),
            float(self.get_parameter('max_centerline_steering').value),
        )
        centerline = processor.project_from_boundaries(centerline, processor_field, boundary_clearance)

        optimizer_field = optimizer.MapField(cleaned_yaml)
        center_smoothing = max(0.0, float(self.get_parameter('lane_center_smoothing').value))
        center = optimizer.smooth_periodic(centerline, lane_spacing, smoothing=center_smoothing)
        center = processor.project_from_boundaries(center, processor_field, boundary_clearance)
        lane_max_offset = float(self.get_parameter('lane_max_offset').value)
        for _ in range(max(0, int(self.get_parameter('lane_recenter_iterations').value))):
            normals = optimizer.normals_for(center)
            bounds = _offset_bounds_allowing_recenter(
                optimizer_field,
                center,
                normals,
                boundary_clearance,
                lane_max_offset,
            )
            lo = np.asarray([item[0] for item in bounds], dtype=np.float64)
            hi = np.asarray([item[1] for item in bounds], dtype=np.float64)
            midpoint_offsets = 0.5 * (lo + hi)
            if float(np.max(np.abs(midpoint_offsets))) < 1e-3:
                break
            center = center + midpoint_offsets[:, None] * normals
            center = optimizer.smooth_periodic(center, lane_spacing, smoothing=center_smoothing)
            center = processor.project_from_boundaries(center, processor_field, boundary_clearance)

        normals = optimizer.normals_for(center)
        bounds = _offset_bounds_allowing_recenter(
            optimizer_field,
            center,
            normals,
            boundary_clearance,
            lane_max_offset,
        )

        lo = np.asarray([item[0] for item in bounds], dtype=np.float64)
        hi = np.asarray([item[1] for item in bounds], dtype=np.float64)
        raw_left_width = np.maximum(-lo, 0.0)
        raw_right_width = np.maximum(hi, 0.0)
        envelope_window = max(1, int(self.get_parameter('lane_width_envelope_window').value))
        max_growth = max(0.0, float(self.get_parameter('lane_width_max_growth_per_station').value))
        width_window = max(1, int(self.get_parameter('lane_width_smoothing_window').value))
        left_envelope = _limit_closed_growth(
            _rolling_min_closed(raw_left_width, envelope_window),
            max_growth,
        )
        right_envelope = _limit_closed_growth(
            _rolling_min_closed(raw_right_width, envelope_window),
            max_growth,
        )
        left_width = np.minimum(_smooth_closed_values(left_envelope, width_window), raw_left_width)
        right_width = np.minimum(_smooth_closed_values(right_envelope, width_window), raw_right_width)
        left_width = np.maximum(left_width, 0.0)
        right_width = np.maximum(right_width, 0.0)
        left = center - left_width[:, None] * normals
        right = center + right_width[:, None] * normals
        center_clearance = processor_field.sample(processor_field.signed_distance, center)
        if float(np.min(center_clearance)) + 1e-4 < boundary_clearance:
            raise RuntimeError(
                'Generated lane center violates the configured boundary clearance '
                f'({float(np.min(center_clearance)):.4f} m < {boundary_clearance:.4f} m).'
            )

        return LaneData(
            center=center,
            left=left,
            right=right,
            left_width=left_width,
            right_width=right_width,
            normals=normals,
            center_clearance=center_clearance,
        )

    def _write_lane_csv(self, lane_csv: Path, lane: LaneData) -> None:
        lane_csv.parent.mkdir(parents=True, exist_ok=True)
        segment_lengths = np.linalg.norm(
            np.roll(lane.center, -1, axis=0) - lane.center,
            axis=1,
        )
        s_values = np.insert(np.cumsum(segment_lengths[:-1]), 0, 0.0)
        with lane_csv.open('w', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow([
                's',
                'center_x',
                'center_y',
                'left_x',
                'left_y',
                'right_x',
                'right_y',
                'w_left',
                'w_right',
                'normal_x',
                'normal_y',
                'center_clearance',
            ])
            for index in range(len(lane.center)):
                writer.writerow([
                    f'{s_values[index]:.8f}',
                    f'{lane.center[index, 0]:.8f}',
                    f'{lane.center[index, 1]:.8f}',
                    f'{lane.left[index, 0]:.8f}',
                    f'{lane.left[index, 1]:.8f}',
                    f'{lane.right[index, 0]:.8f}',
                    f'{lane.right[index, 1]:.8f}',
                    f'{lane.left_width[index]:.8f}',
                    f'{lane.right_width[index]:.8f}',
                    f'{lane.normals[index, 0]:.8f}',
                    f'{lane.normals[index, 1]:.8f}',
                    f'{lane.center_clearance[index]:.8f}',
                ])

    def _save_preview(self, processor, cleaned_image: np.ndarray, cleaned_metadata: dict, lane: LaneData, preview_png: Path) -> None:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        preview_png.parent.mkdir(parents=True, exist_ok=True)
        field = processor.MapField(
            cleaned_image,
            cleaned_metadata,
            int(self.get_parameter('wall_threshold').value),
        )

        fig, axis = plt.subplots(figsize=(10, 10))
        axis.imshow(
            field.image,
            cmap='gray',
            vmin=0,
            vmax=255,
            origin='upper',
            extent=field.extent,
        )
        center_display = np.vstack((lane.center, lane.center[0]))
        left_display = np.vstack((lane.left, lane.left[0]))
        right_display = np.vstack((lane.right, lane.right[0]))
        axis.plot(center_display[:, 0], center_display[:, 1], color='#00d12f', linewidth=1.8, label='center')
        axis.plot(left_display[:, 0], left_display[:, 1], color='#1f77b4', linewidth=1.2, alpha=0.85, label='left bound')
        axis.plot(right_display[:, 0], right_display[:, 1], color='#ff7f0e', linewidth=1.2, alpha=0.85, label='right bound')
        stride = max(1, len(lane.center) // 90)
        for index in range(0, len(lane.center), stride):
            axis.plot(
                [lane.left[index, 0], lane.right[index, 0]],
                [lane.left[index, 1], lane.right[index, 1]],
                color='#1f77b4',
                linewidth=0.7,
                alpha=0.28,
            )
        axis.scatter(lane.left[:, 0], lane.left[:, 1], color='#1f77b4', s=3, alpha=0.35)
        axis.scatter(lane.right[:, 0], lane.right[:, 1], color='#ff7f0e', s=3, alpha=0.35)
        axis.set_aspect('equal', adjustable='box')
        axis.set_xlabel('map x [m]')
        axis.set_ylabel('map y [m]')
        axis.set_title('Generated lane from manual seed')
        axis.grid(alpha=0.2)
        axis.legend(loc='upper right')
        fig.tight_layout()
        fig.savefig(preview_png, dpi=180)
        plt.close(fig)

    def _publish_lane(self, lane: LaneData) -> None:
        stamp = self.get_clock().now().to_msg()
        frame_id = str(self.get_parameter('frame_id').value)
        self.center_pub.publish(_path_msg(lane.center, frame_id, stamp))
        if bool(self.get_parameter('publish_boundary_paths').value):
            self.left_pub.publish(_path_msg(lane.left, frame_id, stamp))
            self.right_pub.publish(_path_msg(lane.right, frame_id, stamp))


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = LaneGenerationNode()
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
