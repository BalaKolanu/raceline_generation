# raceline_generation

## Purpose

ROS 2 package for generating lane bounds and optimizing a raceline from a map
and a manually recorded seed path.

## Use

The normal pipeline is launched by f1tenth_bringup:

~~~bash
ros2 launch f1tenth_bringup process_raceline.launch.py
~~~

The pipeline is:

~~~text
map + manual seed
       |
       v
generate_lanes
       |
       v
lane.csv
       |
       v
process_raceline
       |
       v
paths/raceline.csv
~~~

Individual nodes can also be run:

~~~bash
ros2 run raceline_generation generate_lanes
ros2 run raceline_generation process_raceline
~~~

## Configuration

The bringup launch controls:

- map_directory
- manual_path_directory
- fallback_path_directory
- lane_directory
- output_directory
- archive_directory
- preview_png

The generated files are stored under f1tenth_bringup:

- maps/
- manual_paths/
- lanes/
- paths/
- old_paths/

The optimizer implementation is resource/optimize_manual_map_raceline.py.
The shared map-processing functions are in resource/process_raceline.py.

## Interfaces and behavior

The pipeline is offline with respect to ROS vehicle topics. It reads a map and
closed seed CSV, cleans isolated map pixels, registers the seed to the map,
computes safe lane bounds, optimizes lateral offsets, checks wall clearance,
and writes a fixed-spacing raceline CSV plus preview images.

Canonical entry points are generate_lanes, process_raceline, and edit_raceline.
The duplicate process_raceline_node entry point is intentionally not registered.

## Theory and further reading

The optimizer uses distance fields, curvature-aware path processing, lane-bound
constraints, and steering-limited smoothing. See the
[F1TENTH raceline resources](https://f1tenth.github.io/website/learn/) and the
[minimum-curvature racing-line overview](https://docs.nvidia.com/drive/driveworks-3.5/dwx-vehicle-motion.html).
