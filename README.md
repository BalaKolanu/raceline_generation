# raceline_generation

ROS 2 Python package for raceline generation tools and nodes.

This package now owns the raceline-processing pipeline:

- `process_raceline`
- `process_raceline_node`
- `generate_lanes`
- `resource/process_raceline.py`

The processing code still reads its input map and manual path from the `racestack`
repository and writes the active processed path back to `racestack/paths`.

## Lane Generation

`generate_lanes` uses the manual path only as a route seed. It registers and
cleans the seed against the map, creates a clearance-safe centerline, then
samples left/right safe offsets along the centerline normals.

Default inputs:
- map: `/home/nvidia/f1tenth_ws/src/racestack/maps/*.yaml`
- manual seed path: `/home/nvidia/f1tenth_ws/src/racestack/manual_paths/*.csv`

Default outputs:
- lane CSV: `/home/nvidia/f1tenth_ws/src/racestack/lanes/lane.csv`
- preview PNG: `/home/nvidia/f1tenth_ws/src/racestack/lanes/lane_preview.png`
- ROS path: `/lane/center`

The CSV is the main lane artifact. Each row stores a center point, local normal,
left/right safe widths, and left/right sample points. Boundary `Path` publishing
can be enabled with `publish_boundary_paths:=true`, but for optimization the
station-based widths are the safer representation.

Run:
```bash
ros2 run raceline_generation generate_lanes
```

Then generate the active raceline inside those lane bounds:
```bash
ros2 run raceline_generation process_raceline
```

`process_raceline` uses `racestack/lanes/lane.csv` by default when
`use_lane_bounds:=true`, and updates the same preview PNG with the optimized
raceline overlay.
