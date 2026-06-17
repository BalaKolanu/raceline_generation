# raceline_generation

ROS 2 Python package for raceline generation tools and nodes.

This package now owns the raceline-processing pipeline:

- `process_raceline`
- `process_raceline_node`
- `resource/process_raceline.py`

The processing code still reads its input map and manual path from the `racestack`
repository and writes the active processed path back to `racestack/paths`.
