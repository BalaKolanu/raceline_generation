#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:  # pragma: no cover - available in ROS runtime
    get_package_share_directory = None


RACESTACK_REPO_ROOT = Path('/home/nvidia/spring26_ws/src/racestack')
RACELINE_GENERATION_REPO_ROOT = Path('/home/nvidia/spring26_ws/src/raceline_generation')
DEFAULT_MAP_YAML = RACESTACK_REPO_ROOT / 'maps' / 'racetrack.yaml'
DEFAULT_RACELINE = RACESTACK_REPO_ROOT / 'paths' / 'raceline.csv'


def _resolve_processor_script() -> Path:
    source_script = RACELINE_GENERATION_REPO_ROOT / 'resource' / 'process_raceline.py'
    if source_script.is_file():
        return source_script

    if get_package_share_directory is None:
        raise FileNotFoundError(
            'Could not locate resource/process_raceline.py in source space, and '
            'ament_index_python is unavailable for install-space lookup.'
        )

    share_dir = Path(get_package_share_directory('raceline_generation'))
    installed_script = share_dir / 'resource' / 'process_raceline.py'
    if installed_script.is_file():
        return installed_script

    raise FileNotFoundError('Could not locate resource/process_raceline.py in source or install space.')


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load module {module_name} from {module_path}.')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_xy_csv(path: Path) -> np.ndarray:
    with path.open(newline='') as stream:
        sample = stream.read(4096)
        stream.seek(0)
        delimiter = csv.Sniffer().sniff(sample, delimiters=',;').delimiter
        reader = csv.DictReader(stream, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f'{path} has no header.')
        names = {name.strip().lstrip('#').strip().lower(): name for name in reader.fieldnames}
        x_name = names.get('x', names.get('x_m'))
        y_name = names.get('y', names.get('y_m'))
        if x_name is None or y_name is None:
            raise ValueError(f'{path} must contain x/y or x_m/y_m columns.')
        points = np.asarray(
            [(float(row[x_name]), float(row[y_name])) for row in reader],
            dtype=float,
        )
    if len(points) < 4:
        raise ValueError(f'{path} must contain at least four points.')
    return points


def _segment_lengths(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)


def _control_spacing(points: np.ndarray, target_spacing: float) -> float:
    lengths = _segment_lengths(points)
    lap_length = float(np.sum(lengths))
    if lap_length <= 1e-6:
        return max(target_spacing, 0.5)
    control_count = max(12, int(round(lap_length / max(target_spacing, 0.25))))
    return lap_length / control_count


def _save_raceline_csv(
    path: Path,
    points: np.ndarray,
    processor,
    *,
    spacing: float,
    wheelbase: float,
    speed_profile: str,
    speed_steering_min: float,
    speed_steering_max: float,
    min_speed: float,
    max_speed: float,
    straight_steering_threshold: float,
    turn_angle_threshold_deg: float,
    turn_distance_threshold_m: float,
    gentle_steering_threshold: float,
    medium_steering_threshold: float,
    straight_speed: float,
    gentle_speed: float,
    medium_speed: float,
    corner_speed: float,
) -> None:
    heading = np.arctan2(
        np.roll(points, -1, axis=0)[:, 1] - points[:, 1],
        np.roll(points, -1, axis=0)[:, 0] - points[:, 0],
    )
    kappa = processor.curvature(points)
    steering = np.arctan(wheelbase * kappa)
    if speed_profile == 'linear':
        speed = processor.speed_from_previous_steering(
            steering,
            speed_steering_min,
            speed_steering_max,
            min_speed,
            max_speed,
        )
    else:
        speed = processor.piecewise_speed_from_previous_steering(
            steering,
            straight_steering_threshold,
            gentle_steering_threshold,
            medium_steering_threshold,
            straight_speed,
            gentle_speed,
            medium_speed,
            corner_speed,
        )
    turn_state = processor.classify_turn_states(
        heading,
        spacing,
        turn_angle_threshold_deg,
        turn_distance_threshold_m,
    )
    segment_ref = processor.build_segment_refs(turn_state)
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter=';')
        writer.writerow(['s', 'x', 'y', 'yaw', 'kappa', 'v', 'turn_state', 'segment_ref'])
        for index, point in enumerate(points):
            writer.writerow([
                f'{index * spacing:.8f}',
                f'{point[0]:.8f}',
                f'{point[1]:.8f}',
                f'{heading[index]:.8f}',
                f'{kappa[index]:.8f}',
                f'{speed[index]:.8f}',
                turn_state[index],
                segment_ref[index],
            ])


@dataclass
class EditorConfig:
    map_yaml: Path
    raceline_csv: Path
    output_csv: Path
    boundary_clearance: float
    wall_threshold: int
    max_isolated_pixels: int
    wheelbase: float
    final_spacing: float
    control_spacing: float
    max_final_steering: float
    smoothing_iterations: int
    speed_profile: str
    speed_steering_min: float
    speed_steering_max: float
    min_speed: float
    max_speed: float
    straight_steering_threshold: float
    turn_angle_threshold_deg: float
    turn_distance_threshold_m: float
    gentle_steering_threshold: float
    medium_steering_threshold: float
    straight_speed: float
    gentle_speed: float
    medium_speed: float
    corner_speed: float


class RacelineEditor:
    def __init__(self, processor, config: EditorConfig) -> None:
        self.processor = processor
        self.config = config

        self.original_points = processor.resample_closed_exact(
            _load_xy_csv(config.raceline_csv),
            config.final_spacing,
        )
        self.points = self.original_points.copy()
        self.undo_stack: list[np.ndarray] = []
        self.control_spacing = _control_spacing(self.points, config.control_spacing)
        self.control_points = self._controls_from_path(self.points)
        self.preview_points = self.points.copy()
        self.preview_valid = True
        self.drag_idx: int | None = None
        self.preview_line = None
        self.preview_info = ''

        with config.map_yaml.open() as stream:
            self.metadata = processor.yaml.safe_load(stream)
        image_path = Path(self.metadata['image'])
        if not image_path.is_absolute():
            image_path = config.map_yaml.parent / image_path
        original_image = np.asarray(processor.Image.open(image_path).convert('L'), dtype=np.uint8)
        cleaned_image, _, _ = processor.clean_map(
            original_image,
            config.wall_threshold,
            config.max_isolated_pixels,
        )
        self.field = processor.MapField(cleaned_image, self.metadata, config.wall_threshold)
        self.extent = self.field.extent
        self.map_display = np.flipud(cleaned_image)

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.imshow(
            self.map_display,
            cmap='gray',
            extent=self.extent,
            origin='lower',
            interpolation='nearest',
        )
        self.ax.set_aspect('equal')
        self.ax.set_title(
            'Drag control vertices to reshape the raceline | enter apply | u undo | '
            's save | q quit'
        )
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')
        self.ax.grid(True, alpha=0.2)

        self.line, = self.ax.plot(
            self.points[:, 0],
            self.points[:, 1],
            color='#00d12f',
            linewidth=2.0,
            label='raceline',
        )
        self.original_line, = self.ax.plot(
            self.original_points[:, 0],
            self.original_points[:, 1],
            color='#1f77b4',
            linewidth=1.0,
            alpha=0.35,
            label='original',
        )
        self.control_line, = self.ax.plot(
            np.r_[self.control_points[:, 0], self.control_points[0, 0]],
            np.r_[self.control_points[:, 1], self.control_points[0, 1]],
            color='#ff6b00',
            linewidth=1.2,
            alpha=0.9,
            label='control mesh',
        )
        self.control_scatter = self.ax.scatter(
            self.control_points[:, 0],
            self.control_points[:, 1],
            c='#ff6b00',
            s=26,
            zorder=5,
        )
        self.preview_line, = self.ax.plot([], [], color='#ffd000', linewidth=1.6, alpha=0.85)
        self.status_text = self.ax.text(
            0.01,
            0.01,
            self._status_message(),
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none'},
        )
        self.ax.legend(loc='upper right')

        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_move)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _status_message(self) -> str:
        active = self.preview_points if self.preview_valid else self.points
        clearance = self._min_clearance(active)
        steering = float(np.max(np.abs(self.processor.steering_angles(active, self.config.wheelbase))))
        edit_state = 'preview ready' if self.preview_valid and self.preview_info else 'drag a control point'
        if self.drag_idx is not None:
            edit_state = f'dragging control {self.drag_idx + 1}/{len(self.control_points)}'
        return (
            f'min clearance {clearance:.3f} m | max steering {steering:.3f} rad | '
            f'controls {len(self.control_points)} | {edit_state}\n'
            'Left-drag a control vertex, inspect the yellow preview, then press Enter to commit smoothing.'
        )

    def _min_clearance(self, points: np.ndarray) -> float:
        samples = np.vstack([
            self.processor.straight_segment(
                points[index],
                points[(index + 1) % len(points)],
                spacing=min(0.01, self.config.final_spacing / 5.0),
            )
            for index in range(len(points))
        ])
        return float(np.min(self.field.sample(self.field.signed_distance, samples)))

    def _redraw(self) -> None:
        self.line.set_data(self.points[:, 0], self.points[:, 1])
        self.control_line.set_data(
            np.r_[self.control_points[:, 0], self.control_points[0, 0]],
            np.r_[self.control_points[:, 1], self.control_points[0, 1]],
        )
        self.control_scatter.set_offsets(self.control_points)
        self.status_text.set_text(self._status_message())
        self.fig.canvas.draw_idle()

    def _controls_from_path(self, points: np.ndarray) -> np.ndarray:
        return self.processor.resample_closed_exact(points, self.control_spacing)

    def _nearest_control_index(self, point: np.ndarray) -> int:
        distances = np.linalg.norm(self.control_points - point[None, :], axis=1)
        return int(np.argmin(distances))

    def _on_press(self, event) -> None:
        if event.inaxes != self.ax or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        point = np.asarray([float(event.xdata), float(event.ydata)], dtype=float)
        idx = self._nearest_control_index(point)
        threshold = max(0.35, 0.6 * self.control_spacing)
        if np.linalg.norm(self.control_points[idx] - point) > threshold:
            return
        self.drag_idx = idx
        self.preview_info = ''
        self.preview_line.set_data([], [])
        self._redraw()

    def _on_move(self, event) -> None:
        if self.drag_idx is None or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self.control_points[self.drag_idx] = np.asarray([float(event.xdata), float(event.ydata)], dtype=float)
        self._preview_edit()

    def _on_release(self, _event) -> None:
        if self.drag_idx is None:
            return
        self.drag_idx = None
        self._preview_edit()

    def _on_key(self, event) -> None:
        if event.key == 'enter':
            self._apply_preview()
        elif event.key == 'u':
            self._undo()
        elif event.key == 'r':
            self.control_points = self._controls_from_path(self.points)
            self.preview_points = self.points.copy()
            self.preview_line.set_data([], [])
            self.preview_info = ''
            self.preview_valid = True
            self._redraw()
        elif event.key == 's':
            self._save()
        elif event.key == 'q':
            plt.close(self.fig)

    def _build_candidate(self) -> np.ndarray:
        candidate = self.processor.resample_closed_exact(
            self.control_points,
            self.config.final_spacing,
        )
        if len(candidate) < 8:
            raise ValueError('Need at least eight raceline points after resampling.')
        return candidate

    def _fair_path(self, candidate: np.ndarray) -> np.ndarray:
        current = candidate.copy()
        reference = candidate.copy()
        vertex_clearance = self.config.boundary_clearance + 0.5 * self.config.final_spacing + 0.002
        tether_strength = 0.045

        for _ in range(self.config.smoothing_iterations):
            previous = np.roll(current, 1, axis=0)
            following = np.roll(current, -1, axis=0)
            laplacian = 0.5 * (previous + following) - current
            proposal = current + 0.18 * laplacian + tether_strength * (reference - current)
            current = self.processor.project_from_boundaries(proposal, self.field, vertex_clearance)
            if _ % 20 == 19:
                current = self.processor.resample_closed(current, count=len(reference))

        steering = np.abs(self.processor.steering_angles(current, self.config.wheelbase))
        if float(np.max(steering)) > self.config.max_final_steering:
            current = self.processor.smooth_with_steering_limit(
                current,
                self.field,
                self.config.boundary_clearance,
                self.config.wheelbase,
                self.config.final_spacing,
                self.config.max_final_steering,
            )
        return self.processor.resample_closed_exact(current, self.config.final_spacing)

    def _preview_edit(self) -> None:
        try:
            candidate = self._fair_path(self._build_candidate())
        except Exception as exc:
            self.preview_line.set_data([], [])
            self.preview_info = str(exc)
            self.preview_valid = False
            self.status_text.set_text(f'{self._status_message()}\npreview failed: {exc}')
            self.fig.canvas.draw_idle()
            return

        self.preview_points = candidate
        self.preview_valid = True
        self.preview_line.set_data(candidate[:, 0], candidate[:, 1])
        clearance = self._min_clearance(candidate)
        steering = float(np.max(np.abs(self.processor.steering_angles(candidate, self.config.wheelbase))))
        self.preview_info = f'preview clearance {clearance:.3f} m | preview steering {steering:.3f} rad'
        self.status_text.set_text(f'{self._status_message()}\n{self.preview_info}')
        self.fig.canvas.draw_idle()

    def _apply_preview(self) -> None:
        if not self.preview_valid:
            return
        self.undo_stack.append(self.points.copy())
        self.points = self.preview_points.copy()
        self.control_points = self._controls_from_path(self.points)
        self.preview_line.set_data([], [])
        self.preview_info = ''
        self.preview_valid = True
        self._redraw()

    def _undo(self) -> None:
        if not self.undo_stack:
            return
        self.points = self.undo_stack.pop()
        self.control_points = self._controls_from_path(self.points)
        self.preview_points = self.points.copy()
        self.preview_line.set_data([], [])
        self.preview_info = ''
        self.preview_valid = True
        self._redraw()

    def _save(self) -> None:
        _save_raceline_csv(
            self.config.output_csv,
            self.points,
            self.processor,
            spacing=self.config.final_spacing,
            wheelbase=self.config.wheelbase,
            speed_profile=self.config.speed_profile,
            speed_steering_min=self.config.speed_steering_min,
            speed_steering_max=self.config.speed_steering_max,
            min_speed=self.config.min_speed,
            max_speed=self.config.max_speed,
            straight_steering_threshold=self.config.straight_steering_threshold,
            turn_angle_threshold_deg=self.config.turn_angle_threshold_deg,
            turn_distance_threshold_m=self.config.turn_distance_threshold_m,
            gentle_steering_threshold=self.config.gentle_steering_threshold,
            medium_steering_threshold=self.config.medium_steering_threshold,
            straight_speed=self.config.straight_speed,
            gentle_speed=self.config.gentle_speed,
            medium_speed=self.config.medium_speed,
            corner_speed=self.config.corner_speed,
        )
        clearance = self._min_clearance(self.points)
        print(f'Saved raceline to {self.config.output_csv} with min clearance {clearance:.3f} m.')

    def run(self) -> None:
        plt.show()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Interactively redraw local sections of a raceline on top of the map.',
    )
    parser.add_argument('--map', type=Path, default=DEFAULT_MAP_YAML, dest='map_yaml')
    parser.add_argument('--raceline', type=Path, default=DEFAULT_RACELINE)
    parser.add_argument('--output', type=Path, default=DEFAULT_RACELINE)
    parser.add_argument('--boundary-clearance', type=float, default=0.15)
    parser.add_argument('--wall-threshold', type=int, default=100)
    parser.add_argument('--max-isolated-pixels', type=int, default=3)
    parser.add_argument('--wheelbase', type=float, default=0.324)
    parser.add_argument('--final-spacing', type=float, default=0.05)
    parser.add_argument('--control-spacing', type=float, default=1.0)
    parser.add_argument('--max-final-steering', type=float, default=0.6)
    parser.add_argument('--smoothing-iterations', type=int, default=220)
    parser.add_argument('--speed-profile', choices=('piecewise', 'linear'), default='piecewise')
    parser.add_argument('--speed-steering-min', type=float, default=0.0)
    parser.add_argument('--speed-steering-max', type=float, default=0.58)
    parser.add_argument('--min-speed', type=float, default=1.0)
    parser.add_argument('--max-speed', type=float, default=10.0)
    parser.add_argument('--straight-steering-threshold', type=float, default=0.08)
    parser.add_argument('--turn-angle-threshold-deg', type=float, default=45.0)
    parser.add_argument('--turn-distance-threshold-m', type=float, default=1.5)
    parser.add_argument('--gentle-steering-threshold', type=float, default=0.20)
    parser.add_argument('--medium-steering-threshold', type=float, default=0.38)
    parser.add_argument('--straight-speed', type=float, default=10.0)
    parser.add_argument('--gentle-speed', type=float, default=6.0)
    parser.add_argument('--medium-speed', type=float, default=4.0)
    parser.add_argument('--corner-speed', type=float, default=1.5)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    processor = _load_module('raceline_generation_process_raceline_editor', _resolve_processor_script())
    config = EditorConfig(
        map_yaml=args.map_yaml.expanduser().resolve(),
        raceline_csv=args.raceline.expanduser().resolve(),
        output_csv=args.output.expanduser().resolve(),
        boundary_clearance=float(args.boundary_clearance),
        wall_threshold=int(args.wall_threshold),
        max_isolated_pixels=int(args.max_isolated_pixels),
        wheelbase=float(args.wheelbase),
        final_spacing=float(args.final_spacing),
        control_spacing=float(args.control_spacing),
        max_final_steering=float(args.max_final_steering),
        smoothing_iterations=int(args.smoothing_iterations),
        speed_profile=str(args.speed_profile),
        speed_steering_min=float(args.speed_steering_min),
        speed_steering_max=float(args.speed_steering_max),
        min_speed=float(args.min_speed),
        max_speed=float(args.max_speed),
        straight_steering_threshold=float(args.straight_steering_threshold),
        turn_angle_threshold_deg=float(args.turn_angle_threshold_deg),
        turn_distance_threshold_m=float(args.turn_distance_threshold_m),
        gentle_steering_threshold=float(args.gentle_steering_threshold),
        medium_steering_threshold=float(args.medium_steering_threshold),
        straight_speed=float(args.straight_speed),
        gentle_speed=float(args.gentle_speed),
        medium_speed=float(args.medium_speed),
        corner_speed=float(args.corner_speed),
    )
    editor = RacelineEditor(processor, config)
    editor.run()


if __name__ == '__main__':
    main()
