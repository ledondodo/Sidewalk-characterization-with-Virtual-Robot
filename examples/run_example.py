"""Run the navigator on one of the supplied sample LAZ recordings.

Usage from the repository root:
    uv run python examples/run_example.py

Pass --input, --start and --goal to select another recording or route.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import laspy
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.centerline_navigation import CenterlineNavigator, NavigationConfig


SAMPLE_DIR = REPO_ROOT.parents[1] / "Data Samples" / "LiDAR Recordings"
DEFAULT_INPUT = (
    REPO_ROOT / "data" / "example.las"
    if (REPO_ROOT / "data" / "example.las").is_file()
    else SAMPLE_DIR
    / "classified_litept-l-mls-5cm"
    / "SlamProject-35-03_classified_with_modelclass.laz"
)


def make_sample_route(las_path: Path) -> tuple[list[float], list[float]]:
    """Choose endpoints on the ground class along the recording's long axis."""
    reader = laspy.open(las_path)
    dims = set(reader.header.point_format.dimension_names)
    if "ModelClass" in dims:
        class_field = "ModelClass"
        ground_code = 1
    elif "Label" in dims:
        class_field = "Label"
        ground_code = 1111
    elif "classification" in dims:
        class_field = "classification"
        ground_code = 2
    else:
        raise ValueError("Could not find ModelClass, Label, or classification in this point cloud")

    # Keep a small deterministic ground sample while streaming through large files.
    rng = np.random.default_rng(7)
    ground_chunks = []
    total_ground = 0
    for chunk in reader.chunk_iterator(1_000_000):
        mask = np.asarray(getattr(chunk, class_field)) == ground_code
        indices = np.flatnonzero(mask)
        total_ground += len(indices)
        # Sample every chunk so route selection represents the whole recording.
        if len(indices) > 3_000:
            indices = np.sort(rng.choice(indices, size=3_000, replace=False))
        ground_chunks.append(
            np.column_stack((chunk.x[indices], chunk.y[indices], chunk.z[indices]))
        )
    reader.close()
    ground = np.concatenate(ground_chunks, axis=0) if ground_chunks else np.empty((0, 3))
    if len(ground) < 2:
        raise ValueError(f"No ground points found (expected class {ground_code})")
    if len(ground) > 150_000:
        ground = ground[rng.choice(len(ground), size=150_000, replace=False)]

    print(f"Sampled {len(ground)} of {total_ground} ground points for route selection")
    center = ground[:, :2].mean(axis=0)
    _, _, vectors = np.linalg.svd(ground[:, :2] - center, full_matrices=False)
    axis = vectors[0]
    projection = (ground[:, :2] - center) @ axis
    mid = np.median(projection)
    # A short representative route is practical even for city-scale recordings.
    start_idx = np.argmin(np.abs(projection - (mid - 12.5)))
    goal_idx = np.argmin(np.abs(projection - (mid + 12.5)))
    return ground[start_idx].tolist(), ground[goal_idx].tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input .las/.laz file")
    parser.add_argument("--start", type=float, nargs=3, help="Start x y z; defaults to an automatic sample route")
    parser.add_argument("--goal", type=float, nargs=3, help="Goal x y z; defaults to an automatic sample route")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum navigation steps")
    parser.add_argument("--max-points", type=int, help="Randomly retain at most this many points when loading a large cloud")
    parser.add_argument("--route-padding", type=float, default=30.0, help="Spatial margin around explicit start/goal, in meters")
    parser.add_argument("--open-space-grid-size", type=float, default=0.05, help="Obstacle proximity threshold in meters")
    parser.add_argument("--output", type=Path, help="CSV output path; defaults to <input name>.csv")
    args = parser.parse_args()

    las_path = args.input.expanduser().resolve()
    if not las_path.is_file():
        raise FileNotFoundError(
            f"Sample file not found: {las_path}\n"
            "Use --input to point to a .las/.laz file, or put the sample recordings beside this repository."
        )
    if (args.start is None) != (args.goal is None):
        parser.error("--start and --goal must be supplied together")
    start, goal = (args.start, args.goal) if args.start is not None else make_sample_route(las_path)
    print(f"Route: {start} -> {goal}")
    bounds = None
    if args.start is not None:
        if args.route_padding < 0:
            parser.error("--route-padding cannot be negative")
        bounds = (
            min(start[0], goal[0]) - args.route_padding,
            min(start[1], goal[1]) - args.route_padding,
            max(start[0], goal[0]) + args.route_padding,
            max(start[1], goal[1]) + args.route_padding,
        )

    input_name = las_path.stem
    output_file = args.output or (REPO_ROOT / "results" / f"{input_name}.csv")

    config = NavigationConfig(
        triangle_side_length=0.8,
        step_size=0.3,
        robot_width=0.8,
        sensor_range=3.0,
        min_passage_width=0.6,
        desired_side_distance=0.3,
        min_side_distance=0.2,
        open_space_grid_size=args.open_space_grid_size,
        ray_coarse_step=0.10,
        width_fan_rays=3,
    )
    navigator = CenterlineNavigator(config)
    navigator.load_point_cloud(str(las_path), max_points=args.max_points, bounds=bounds)
    navigator.initialize_robot(start, goal)
    navigator.plan_ground_route()
    success = navigator.navigate(max_steps=args.max_steps)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    navigator.save_results(str(output_file))
    print(f"Navigation {'reached the goal' if success else 'stopped before the goal'}; output: {output_file}")


if __name__ == "__main__":
    main()
