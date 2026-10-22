from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import heapq
import numpy as np
import laspy
import pandas as pd
from scipy.ndimage import binary_closing, distance_transform_edt
from scipy.spatial import KDTree


class NavigationState(Enum):
    FIND_PATH = 1
    FOLLOW_CENTERLINE = 2
    NAVIGATE_BETWEEN = 3
    AVOID_OBSTACLE = 4
    GOAL_REACHED = 5
    BLOCKED = 6


@dataclass
class NavigationConfig:
    """Configuration parameters for sidewalk centerline navigation"""
    # Robot parameters
    triangle_side_length: float = 0.8       # ~robot footprint
    step_size: float = 0.3                  # 30 cm
    robot_width: float = 0.8                # sidewalk footprint model

    # Navigation parameters
    sensor_range: float = 5.0               # maximum distance to search for obstacles/edges 
    goal_tolerance: float = 0.2             # how close to the goal is considered “reached” (m)

    # Point cloud parameters
    navigable_classes: List[float] = field(
        default_factory=lambda: [1111.00, 1112.00, 1113.00, 1114.00, 1115.00, 1116.00, 1810.00]
    )
    open_space_grid_size: float = 0.05      # spatial resolution used when checking free space
    point_density: float = 0.05

    # Centerline navigation parameters
    min_passage_width: float = 1.2          # min usable sidewalk width
    obstacle_avoidance_distance: float = 0.5 # preferred minimum distance from obstacles (m)
    centerline_samples: int = 36            # number of radial directions sampled around the

    # Safety / mode parameters
    safe_distance_from_obstacles: float = 0.4
    mode_switch_improvement: float = 0.5

    # Desired lateral clearance to each edge (centerline target)
    desired_side_distance: float = 0.5      # want ~0.5 m to both sides
    min_side_distance: float = 0.45          # hard minimum, ~40 cm

    # Anti‑zigzag smoothing / gains
    heading_smooth_alpha: float = 0.2       # 0=no smoothing, 1=instant
    center_error_gain: float = 0.1          # penalise imbalance (smaller = smoother)
    distance_error_gain: float = 0.2        # penalise distance error

    # High‑precision path width measurement
    ray_coarse_step: float = 0.05           # 5 cm coarse step
    ray_refine_iters: int = 6               # binary search refinement steps
    width_fan_half_angle_deg: float = 15.0  # fan ±15° around perpendicular
    width_fan_rays: int = 5                 # rays per side in the fan
    
    # New: radius where we stop enforcing centerline/side distances
    relax_centerline_radius: float = 1.0   # e.g. last 1 m before goal

    def __post_init__(self):
        if self.navigable_classes is None:
            self.navigable_classes = [
                1111.00, 1112.00, 1113.00,
                1114.00, 1115.00, 1116.00
            ]


@dataclass
class TerrainFeatures:
    step_id: int
    vertex1: np.ndarray
    vertex2: np.ndarray
    vertex3: np.ndarray
    centroid: np.ndarray
    step_length: float
    open_space_width_left: float
    open_space_width_right: float
    surface_normal: np.ndarray
    slope_angle: float
    cross_slope_angle: float
    surface_class: float
    heading_direction: np.ndarray
    distance_to_goal: float
    navigation_state: str
    distance_to_nearest_obstacle: float
    obstacle_distances: Dict[str, float]
    path_width: float
    centerline_offset: float


class CenterlineNavigator:
    def __init__(self, config: NavigationConfig):
        self.config = config
        self.point_cloud = None
        self.kdtree = None
        self.navigable_points = None
        self.obstacle_points = None
        self.obstacle_kdtree = None

        self.current_position = None
        self.goal_position = None
        self.current_heading = None
        self.state = NavigationState.FIND_PATH

        self.path_points: List[np.ndarray] = []
        self.terrain_features: List[TerrainFeatures] = []
        self.step_counter = 0
        self.planned_route: Optional[np.ndarray] = None
        self.route_index = 0
        self.route_grid: Optional[np.ndarray] = None
        self.route_grid_origin: Optional[np.ndarray] = None
        self.route_grid_size: Optional[float] = None

    # ------------------------------------------------------------------
    # Point cloud loading
    # ------------------------------------------------------------------
    def load_point_cloud(
        self,
        las_file_path: str,
        max_points: Optional[int] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ):
        print(f"Loading point cloud from {las_file_path}...")
        reader = laspy.open(las_file_path)
        total_points = reader.header.point_count
        dimensions = set(reader.header.point_format.dimension_names)
        class_field = next(
            (name for name in ("Label", "ModelClass", "classification") if name in dimensions),
            None,
        )
        if class_field is None:
            raise ValueError(
                "No semantic class dimension found. Expected `Label`, `ModelClass`, "
                "or LAS `classification`."
            )
        navigable_classes = self.config.navigable_classes
        if class_field == "ModelClass" and navigable_classes == [1111.00, 1112.00, 1113.00, 1114.00, 1115.00, 1116.00, 1810.00]:
            navigable_classes = [1]  # sample-data model legend: 1 = ground
            print("Using ModelClass=1 as the navigable ground surface")
        elif class_field == "classification" and navigable_classes == [1111.00, 1112.00, 1113.00, 1114.00, 1115.00, 1116.00, 1810.00]:
            navigable_classes = [2]  # LAS standard: 2 = ground
            print("Using LAS classification=2 as the navigable ground surface")
        # Keep all later navigation queries consistent with the selected field.
        self.config.navigable_classes = list(navigable_classes)

        # Read in chunks so large LAZ files can be sampled without first
        # expanding the entire recording in memory.
        rng = np.random.default_rng(42)
        stride = max(1, int(np.ceil(total_points / max_points))) if max_points else 1
        point_chunks = []
        class_chunks = []
        for las_chunk in reader.chunk_iterator(1_000_000):
            x_all = np.asarray(las_chunk.x)
            y_all = np.asarray(las_chunk.y)
            if bounds is None:
                selected = np.arange(len(las_chunk))
            else:
                xmin, ymin, xmax, ymax = bounds
                selected = np.flatnonzero(
                    (x_all >= xmin) & (x_all <= xmax)
                    & (y_all >= ymin) & (y_all <= ymax)
                )
            if stride > 1 and len(selected) > 1:
                count = max(1, int(round(len(selected) / stride)))
                selected = np.sort(rng.choice(selected, size=count, replace=False))
            if len(selected) == 0:
                continue
            x = x_all[selected]
            y = y_all[selected]
            z = np.asarray(las_chunk.z[selected])
            cls = np.asarray(getattr(las_chunk, class_field)[selected], dtype=np.float32)
            point_chunks.append(np.column_stack((x, y, z)))
            class_chunks.append(cls)
        reader.close()
        if not point_chunks:
            raise ValueError("No points found in the requested spatial bounds")
        points = np.concatenate(point_chunks, axis=0)
        classifications = np.concatenate(class_chunks, axis=0)
        if stride > 1:
            print(f"Sampled {len(points)} of {total_points} points (max_points={max_points})")
        elif bounds is not None:
            print(f"Loaded {len(points)} points in the requested spatial bounds")

        navigable_mask = np.isin(classifications, navigable_classes)
        if not np.any(navigable_mask):
            raise ValueError(
                f"No points match navigable classes {navigable_classes} in `{class_field}`. "
                "Set NavigationConfig.navigable_classes to the class values for this file."
            )
        self.navigable_points = points[navigable_mask]
        self.obstacle_points = points[~navigable_mask]

        self.kdtree = KDTree(points)
        self.navigable_kdtree = KDTree(self.navigable_points)
        if len(self.obstacle_points) > 0:
            self.obstacle_kdtree = KDTree(self.obstacle_points)
        else:
            self.obstacle_kdtree = None

        self.point_cloud = {
            "points": points,
            "classifications": classifications,
        }

        print(f"Loaded {len(points)} points ({class_field})")
        print(f"Navigable points: {len(self.navigable_points)}")
        print(f"Obstacle points: {len(self.obstacle_points)}")

    # ------------------------------------------------------------------
    # Robot init
    # ------------------------------------------------------------------
    def initialize_robot(self, start_point: List[float], end_point: List[float]):
        self.current_position = np.array(start_point, dtype=float)
        self.goal_position = np.array(end_point, dtype=float)

        direction_to_goal = self.goal_position - self.current_position
        direction_to_goal[2] = 0.0
        self.current_heading = direction_to_goal / np.linalg.norm(direction_to_goal)

        print(f"Robot initialized at: {self.current_position}")
        print(f"Goal position: {self.goal_position}")
        print(
            f"Initial distance to goal: "
            f"{np.linalg.norm(self.goal_position - self.current_position):.2f}m"
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def create_triangle_vertices(
        self, center: np.ndarray, heading: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        L = self.config.triangle_side_length
        h = L * np.sqrt(3.0) / 2.0

        local_v1 = np.array([2.0 * h / 3.0, 0.0, 0.0])
        local_v2 = np.array([-h / 3.0, L / 2.0, 0.0])
        local_v3 = np.array([-h / 3.0, -L / 2.0, 0.0])

        angle = np.arctan2(heading[1], heading[0])
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, -sin_a, 0.0],
                      [sin_a,  cos_a, 0.0],
                      [0.0,    0.0,   1.0]])

        v1 = center + R @ local_v1
        v2 = center + R @ local_v2
        v3 = center + R @ local_v3

        v1 = self.conform_to_surface(v1)
        v2 = self.conform_to_surface(v2)
        v3 = self.conform_to_surface(v3)
        return v1, v2, v3

    def conform_to_surface(self, point: np.ndarray) -> np.ndarray:
        d, idx = self.navigable_kdtree.query(point, k=5)
        if len(idx) > 0 and d[0] < np.inf:
            w = 1.0 / (d + 1e-6)
            w /= w.sum()
            nearby = self.navigable_points[idx]
            z = np.average(nearby[:, 2], weights=w)
            return np.array([point[0], point[1], z])
        return point

    # ------------------------------------------------------------------
    # Navigable / obstacle queries (sidewalk edges act as walls)
    # ------------------------------------------------------------------
    def is_on_navigable_surface(self, position: np.ndarray) -> bool:
        # Query the navigable tree directly. Looking up the nearest point in
        # the all-points tree can reject a valid ground point whenever a
        # nearby non-ground label is marginally closer (common at 5 cm density).
        if self.navigable_kdtree is None:
            return False
        distance, _ = self.navigable_kdtree.query(position, k=1)
        return bool(distance < 0.1)

    def is_safe_segment(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Require the entire robot step to stay within classified ground."""
        length = float(np.linalg.norm(end[:2] - start[:2]))
        sample_count = max(1, int(np.ceil(length / 0.05)))
        for fraction in np.linspace(0.0, 1.0, sample_count + 1):
            sample = start + fraction * (end - start)
            if self.route_grid is not None:
                cell = np.floor(
                    (sample[:2] - self.route_grid_origin) / self.route_grid_size
                ).astype(int)
                if not (
                    0 <= cell[0] < self.route_grid.shape[1]
                    and 0 <= cell[1] < self.route_grid.shape[0]
                    and self.route_grid[cell[1], cell[0]]
                ):
                    return False
                continue
            sample = self.conform_to_surface(sample)
            if not self.is_on_navigable_surface(sample):
                return False
        return True

    def plan_ground_route(self, grid_size: float = 0.2) -> np.ndarray:
        """Plan a global A* route through ground cells with robot clearance."""
        if self.navigable_points is None or self.current_position is None:
            raise RuntimeError("Load the point cloud and initialize start/goal first")

        points = self.navigable_points
        origin = points[:, :2].min(axis=0) - grid_size
        shape_xy = np.ceil((points[:, :2].max(axis=0) - origin) / grid_size).astype(int) + 1
        occupied = np.zeros((shape_xy[1], shape_xy[0]), dtype=bool)
        cells = np.floor((points[:, :2] - origin) / grid_size).astype(int)
        occupied[cells[:, 1], cells[:, 0]] = True
        # Fill only sub-grid acquisition gaps; building footprints remain
        # disconnected because their empty regions are many cells wide.
        occupied = binary_closing(occupied, structure=np.ones((3, 3), dtype=bool))

        clearance = distance_transform_edt(occupied) * grid_size
        required_clearance = self.config.robot_width / 2.0
        traversable = occupied & (clearance >= required_clearance)
        traversable_points = np.argwhere(traversable)
        if len(traversable_points) == 0:
            raise ValueError("No ground cells have enough clearance for the robot")

        centers_xy = origin + (traversable_points[:, ::-1] + 0.5) * grid_size
        cell_tree = KDTree(centers_xy)
        start_cell_idx = int(cell_tree.query(self.current_position[:2])[1])
        goal_cell_idx = int(cell_tree.query(self.goal_position[:2])[1])
        start_cell = tuple(traversable_points[start_cell_idx])
        goal_cell = tuple(traversable_points[goal_cell_idx])

        start_snap = np.linalg.norm(centers_xy[start_cell_idx] - self.current_position[:2])
        goal_snap = np.linalg.norm(centers_xy[goal_cell_idx] - self.goal_position[:2])
        if start_snap > 2.0 or goal_snap > 2.0:
            raise ValueError(
                f"Could not connect start/goal to a clear ground cell "
                f"(snaps {start_snap:.2f}m, {goal_snap:.2f}m)"
            )

        directions = [
            (-1, -1, np.sqrt(2.0)), (-1, 0, 1.0), (-1, 1, np.sqrt(2.0)),
            (0, -1, 1.0), (0, 1, 1.0),
            (1, -1, np.sqrt(2.0)), (1, 0, 1.0), (1, 1, np.sqrt(2.0)),
        ]
        frontier = [(0.0, start_cell)]
        came_from = {start_cell: None}
        cost_so_far = {start_cell: 0.0}
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal_cell:
                break
            for dy, dx, step_cost in directions:
                nxt = (current[0] + dy, current[1] + dx)
                if not (0 <= nxt[0] < occupied.shape[0] and 0 <= nxt[1] < occupied.shape[1]):
                    continue
                if not traversable[nxt]:
                    continue
                if dx and dy and (not traversable[current[0], nxt[1]] or not traversable[nxt[0], current[1]]):
                    continue
                local_clearance = max(clearance[nxt] - required_clearance, grid_size)
                new_cost = cost_so_far[current] + step_cost * (1.0 + 0.15 / local_clearance)
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    heuristic = np.hypot(goal_cell[0] - nxt[0], goal_cell[1] - nxt[1])
                    heapq.heappush(frontier, (new_cost + heuristic, nxt))
                    came_from[nxt] = current

        if goal_cell not in came_from:
            raise ValueError("No connected ground route with robot clearance exists between these endpoints")

        self.route_grid = traversable
        self.route_grid_origin = origin
        self.route_grid_size = grid_size

        grid_path = []
        cell = goal_cell
        while cell is not None:
            grid_path.append(cell)
            cell = came_from[cell]
        grid_path.reverse()
        xy_path = origin + (np.asarray(grid_path)[:, ::-1] + 0.5) * grid_size
        route = [self.current_position.copy()]
        route.extend(
            self.conform_to_surface(np.array([xy[0], xy[1], self.current_position[2]]))
            for xy in xy_path
        )
        route.append(self.goal_position.copy())

        # Resample the A* polyline at the robot's step length.
        sampled = [route[0]]
        remaining = self.config.step_size
        for a, b in zip(route[:-1], route[1:]):
            segment = b - a
            length = float(np.linalg.norm(segment[:2]))
            if length < 1e-9:
                continue
            direction = segment / length
            travelled = 0.0
            while length - travelled >= remaining:
                travelled += remaining
                sampled.append(a + direction * travelled)
                remaining = self.config.step_size
            remaining -= length - travelled
        if np.linalg.norm(sampled[-1] - route[-1]) > 1e-6:
            sampled.append(route[-1])
        self.planned_route = np.asarray(sampled)
        self.route_index = 1 if len(sampled) > 1 else 0
        print(
            f"Planned ground route: {len(self.planned_route)} waypoints, "
            f"{sum(np.linalg.norm(b[:2] - a[:2]) for a, b in zip(self.planned_route[:-1], self.planned_route[1:])):.2f}m"
        )
        return self.planned_route

    def cast_ray(
        self, origin: np.ndarray, direction: np.ndarray, max_distance: float = None
    ) -> float:
        """
        Ray that stops when navigable surface ends or obstacle is hit.
        Uses coarse stepping + binary search refinement for higher accuracy.
        """
        cfg = self.config
        if max_distance is None:
            max_distance = cfg.sensor_range

        step = cfg.ray_coarse_step
        d = step
        prev_d = 0.0

        while d <= max_distance:
            p = origin + direction * d

            on_nav = self.is_on_navigable_surface(p)

            if self.obstacle_kdtree is not None:
                od, _ = self.obstacle_kdtree.query(p, k=1)
                if od < cfg.open_space_grid_size:
                    on_nav = False

            if not on_nav:
                # boundary between prev_d and d → refine
                lo = prev_d
                hi = d
                for _ in range(cfg.ray_refine_iters):
                    mid = 0.5 * (lo + hi)
                    pm = origin + direction * mid
                    on_mid = self.is_on_navigable_surface(pm)
                    if self.obstacle_kdtree is not None:
                        odm, _ = self.obstacle_kdtree.query(pm, k=1)
                        if odm < cfg.open_space_grid_size:
                            on_mid = False
                    if on_mid:
                        lo = mid
                    else:
                        hi = mid
                return lo

            prev_d = d
            d += step

        return max_distance

    def calculate_obstacle_distances(
        self, position: np.ndarray
    ) -> Dict[str, float]:
        if self.obstacle_kdtree is None and self.kdtree is None:
            return {
                f"dir_{i}": self.config.sensor_range
                for i in range(self.config.centerline_samples)
            }

        angles = np.linspace(
            0.0, 2.0 * np.pi, self.config.centerline_samples, endpoint=False
        )
        dists: Dict[str, float] = {}
        for i, ang in enumerate(angles):
            dir_vec = np.array([np.cos(ang), np.sin(ang), 0.0])
            dists[f"dir_{i}"] = self.cast_ray(position, dir_vec)
        return dists

    # ------------------------------------------------------------------
    # High‑accuracy sidewalk width and centerline
    # ------------------------------------------------------------------
    def _side_clearances(
        self, pos: np.ndarray, heading: np.ndarray
    ) -> Tuple[float, float]:
        """
        Legacy single‑ray clearances (kept for logging / safety).
        Not used for final path_width; main width uses fan‑based method.
        """
        perp = np.array([-heading[1], heading[0], 0.0])
        perp /= np.linalg.norm(perp) + 1e-6
        left = self.cast_ray(pos, perp)
        right = self.cast_ray(pos, -perp)
        return left, right

    def _fan_side_clearances(
        self, pos: np.ndarray, heading: np.ndarray
    ) -> Tuple[float, float]:
        """
        Fan‑based clearance for left/right, projected onto true perpendicular.
        """
        cfg = self.config

        # True perpendicular
        perp = np.array([-heading[1], heading[0], 0.0])
        perp /= np.linalg.norm(perp) + 1e-6

        half_angle = np.radians(cfg.width_fan_half_angle_deg)
        n_rays = cfg.width_fan_rays if cfg.width_fan_rays > 0 else 1

        if n_rays == 1:
            local_angles = np.array([0.0])
        else:
            local_angles = np.linspace(-half_angle, +half_angle, n_rays)

        h2d = np.array([heading[0], heading[1], 0.0])
        h2d /= np.linalg.norm(h2d) + 1e-6
        side = perp
        forward = h2d

        left_effective: List[float] = []
        right_effective: List[float] = []

        for ang in local_angles:
            ca, sa = np.cos(ang), np.sin(ang)

            # Left side
            dir_left = ca * side + sa * forward
            dir_left /= np.linalg.norm(dir_left) + 1e-6
            d_left = self.cast_ray(pos, dir_left)
            eff_left = d_left * np.dot(dir_left, side)
            if eff_left > 0.0:
                left_effective.append(eff_left)

            # Right side
            dir_right = -side
            dir_right = ca * dir_right + sa * forward
            dir_right /= np.linalg.norm(dir_right) + 1e-6
            d_right = self.cast_ray(pos, dir_right)
            eff_right = d_right * np.dot(dir_right, -side)
            if eff_right > 0.0:
                right_effective.append(eff_right)

        left = min(left_effective) if left_effective else 0.0
        right = min(right_effective) if right_effective else 0.0
        return left, right

    def calculate_path_width(self, position: np.ndarray, heading: np.ndarray) -> float:
        """
        Measure sidewalk width using a fan of rays around the perpendicular.
        Returns width in meters (left + right).
        """
        left, right = self._fan_side_clearances(position, heading)
        return left + right

    def calculate_centerline_offset(
        self, position: np.ndarray, heading: np.ndarray
    ) -> float:
        """
        Offset from centerline: (left - right) / 2, using same fan‑based rays.
        """
        left, right = self._fan_side_clearances(position, heading)
        return (left - right) / 2.0

    # ------------------------------------------------------------------
    # Centerline heading selection
    # ------------------------------------------------------------------
    def find_centerline_direction(self, position: np.ndarray) -> np.ndarray:
        """Choose heading that keeps robot near sidewalk centerline."""
        cfg = self.config

        angles = np.linspace(
            0.0, 2.0 * np.pi, cfg.centerline_samples, endpoint=False
        )
        goal_vec = self.goal_position - position
        goal_vec[2] = 0.0
        goal_vec = goal_vec / (np.linalg.norm(goal_vec) + 1e-6)

        best_score = -np.inf
        best_dir = self.current_heading if self.current_heading is not None else goal_vec

        for ang in angles:
            direction = np.array([np.cos(ang), np.sin(ang), 0.0])

            if np.dot(direction, goal_vec) < 0.1:
                continue

            lookahead_pos = position + direction * (cfg.step_size * 2.0)

            if not self.is_on_navigable_surface(lookahead_pos):
                continue

            left, right = self._fan_side_clearances(lookahead_pos, direction)
            width = left + right

            required_width = max(
                cfg.min_passage_width,
                cfg.robot_width + 2.0 * cfg.min_side_distance,
            )
            if width < required_width:
                continue

            d_left = left - cfg.desired_side_distance
            d_right = right - cfg.desired_side_distance
            center_error = d_left - d_right

            balance_cost = abs(center_error)
            distance_cost = abs(d_left) + abs(d_right)

            soft_penalty = 0.0
            if left < cfg.min_side_distance or right < cfg.min_side_distance:
                soft_penalty += 5.0 * max(
                    0.0, cfg.min_side_distance - min(left, right)
                )

            align = np.dot(direction, goal_vec)

            score = (
                3.0 * align
                - cfg.center_error_gain * balance_cost
                - cfg.distance_error_gain * distance_cost
                - soft_penalty
            )

            if score > best_score:
                best_score = score
                best_dir = direction

        if self.current_heading is None:
            smoothed = best_dir
        else:
            alpha = cfg.heading_smooth_alpha
            smoothed = (1.0 - alpha) * self.current_heading + alpha * best_dir
            smoothed /= np.linalg.norm(smoothed) + 1e-6

        return smoothed

    def find_gap_between_obstacles(
        self, position: np.ndarray
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Gap logic kept for completeness; centerline is primary."""
        cfg = self.config
        obs_dists = self.calculate_obstacle_distances(position)
        angles = np.linspace(
            0.0, 2.0 * np.pi, cfg.centerline_samples, endpoint=False
        )

        required_width = max(
            cfg.min_passage_width,
            cfg.robot_width + 2.0 * cfg.min_side_distance,
        )

        gaps = []
        current = None
        for i, ang in enumerate(angles):
            dist = obs_dists[f"dir_{i}"]
            if dist >= cfg.min_side_distance:
                if current is None:
                    current = {"start": i, "end": i, "min_width": dist}
                else:
                    current["end"] = i
                    current["min_width"] = min(current["min_width"], dist)
            else:
                if current is not None:
                    gaps.append(current)
                    current = None

        if current is not None:
            if gaps and gaps[0]["start"] == 0:
                gaps[0]["start"] = current["start"]
                gaps[0]["min_width"] = min(
                    gaps[0]["min_width"], current["min_width"]
                )
            else:
                gaps.append(current)

        if not gaps:
            return None

        goal_vec = self.goal_position - position
        goal_vec[2] = 0.0
        goal_vec = goal_vec / (np.linalg.norm(goal_vec) + 1e-6)

        best_gap = None
        best_score = -np.inf
        for g in gaps:
            c_idx = (g["start"] + g["end"]) // 2
            ang = angles[c_idx % len(angles)]
            direction = np.array([np.cos(ang), np.sin(ang), 0.0])

            ang_width = (g["end"] - g["start"]) * (2.0 * np.pi / len(angles))
            gap_width = 2.0 * g["min_width"] * np.sin(ang_width / 2.0)

            if gap_width < required_width:
                continue

            align = np.dot(direction, goal_vec)
            score = gap_width * (1.0 + align) / 2.0

            if score > best_score:
                best_score = score
                best_gap = (direction, gap_width)

        return best_gap

    # ------------------------------------------------------------------
    # Navigation step: sidewalk centerline first, with smoothing
    # ------------------------------------------------------------------
    def centerline_navigation_step(self) -> bool:
        cfg = self.config

        if self.planned_route is not None:
            if self.route_index >= len(self.planned_route):
                self.state = NavigationState.GOAL_REACHED
                return True
            next_pos = self.planned_route[self.route_index]
            delta = next_pos - self.current_position
            delta[2] = 0.0
            self.current_heading = delta / (np.linalg.norm(delta) + 1e-6)
            if not self.is_safe_segment(self.current_position, next_pos):
                self.state = NavigationState.BLOCKED
                return True
            self.state = NavigationState.FOLLOW_CENTERLINE
            self.current_position = self.conform_to_surface(next_pos)
            self.route_index += 1
            return False

        dist_goal = np.linalg.norm(self.goal_position - self.current_position)

        # If we are very close: just go straight to goal, ignore centerline
        if dist_goal < cfg.goal_tolerance:
            self.state = NavigationState.GOAL_REACHED
            return True

        if dist_goal < cfg.relax_centerline_radius:
            # Direct goal‑seeking mode, no centering
            goal_vec = self.goal_position - self.current_position
            goal_vec[2] = 0.0
            goal_dir = goal_vec / (np.linalg.norm(goal_vec) + 1e-6)

            # You can still smooth a bit to avoid a sharp last turn
            alpha = self.config.heading_smooth_alpha
            if self.current_heading is None:
                self.current_heading = goal_dir
            else:
                self.current_heading = (
                    (1.0 - alpha) * self.current_heading + alpha * goal_dir
                )
                self.current_heading /= np.linalg.norm(self.current_heading) + 1e-6

            self.state = NavigationState.FOLLOW_CENTERLINE  # or a new NEAR_GOAL state

        else:
            # Normal centerline behaviour (unchanged)
            self.current_heading = self.find_centerline_direction(self.current_position)

            left, right = self._side_clearances(
                self.current_position, self.current_heading
            )
            min_side = min(left, right)
            if min_side < cfg.min_side_distance * 0.7:
                self.state = NavigationState.AVOID_OBSTACLE
            else:
                self.state = NavigationState.FOLLOW_CENTERLINE
            # `find_centerline_direction` already evaluates candidate forward
            # directions against the goal and the available passage. Rotating
            # that result toward a perpendicular whenever one side is close
            # to an obstacle discards the goal direction and can make the
            # robot circle in place at narrow or noisy sections.

        # Validate the complete step, not just its endpoint. If the preferred
        # heading crosses a gap, try nearby ground-supported headings before
        # declaring the route blocked.
        goal_vec = self.goal_position - self.current_position
        goal_vec[2] = 0.0
        goal_vec /= np.linalg.norm(goal_vec) + 1e-6
        candidate_headings = [self.current_heading]
        offsets = np.radians(np.arange(10.0, 181.0, 10.0))
        for offset in offsets:
            for sign in (-1.0, 1.0):
                angle = sign * offset
                ca, sa = np.cos(angle), np.sin(angle)
                heading = np.array([
                    ca * self.current_heading[0] - sa * self.current_heading[1],
                    sa * self.current_heading[0] + ca * self.current_heading[1],
                    0.0,
                ])
                candidate_headings.append(heading)

        safe_candidates = []
        for heading in candidate_headings:
            if np.dot(heading, goal_vec) <= 0.0:
                continue
            next_pos = self.current_position + heading * cfg.step_size
            if self.is_safe_segment(self.current_position, next_pos):
                # Prefer progress toward the goal, with a small continuity
                # preference to avoid abrupt changes between safe steps.
                score = 2.0 * np.dot(heading, goal_vec) + 0.25 * np.dot(
                    heading, self.current_heading
                )
                safe_candidates.append((score, heading, next_pos))

        if not safe_candidates:
            self.state = NavigationState.BLOCKED
            return True

        _, chosen_heading, next_pos = max(safe_candidates, key=lambda item: item[0])
        self.current_heading = chosen_heading
        self.current_position = self.conform_to_surface(next_pos)
        return False

    # ------------------------------------------------------------------
    # Slopes, classes, features, navigate, save
    # ------------------------------------------------------------------
    def calculate_surface_normal(
        self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray
    ) -> np.ndarray:
        e1 = v2 - v1
        e2 = v3 - v1
        n = np.cross(e1, e2)
        norm = np.linalg.norm(n)
        if norm > 0:
            n = n / norm
        if n[2] < 0:
            n = -n
        return n

    def calculate_slopes(
        self, normal: np.ndarray, heading: np.ndarray
    ) -> Tuple[float, float]:
        vertical = np.array([0.0, 0.0, 1.0])
        slope = np.arccos(np.clip(np.dot(normal, vertical), -1, 1))
        slope = np.degrees(slope)

        h3 = np.array([heading[0], heading[1], 0.0])
        perp = np.array([-h3[1], h3[0], 0.0])
        cs_comp = np.dot(normal, perp)
        cross = np.degrees(np.arcsin(np.clip(cs_comp, -1, 1)))
        return slope, cross

    def get_surface_class_at_position(self, position: np.ndarray) -> int:
        d, idx = self.kdtree.query(position, k=1)
        if isinstance(d, np.ndarray):
            d = d[0] if len(d) > 0 else np.inf
            idx = idx[0] if len(idx) > 0 else -1
        if d < 0.2 and idx >= 0:
            cls = self.point_cloud["classifications"][idx]
            if hasattr(cls, "item"):
                return int(cls.item())
            elif hasattr(cls, "__iter__") and not isinstance(cls, str):
                return int(list(cls)[0])
            else:
                return int(cls)
        return 0

    def extract_terrain_features(self) -> TerrainFeatures:
        v1, v2, v3 = self.create_triangle_vertices(
            self.current_position, self.current_heading
        )
        centroid = (v1 + v2 + v3) / 3.0

        step_length = 0.0
        if self.path_points:
            step_length = np.linalg.norm(centroid - self.path_points[-1])

        obs_dists = self.calculate_obstacle_distances(centroid)
        min_obs = min(obs_dists.values())

        path_width = self.calculate_path_width(centroid, self.current_heading)
        offset = self.calculate_centerline_offset(centroid, self.current_heading)

        # For open_space_* columns, we can reuse fan‑based or single‑ray.
        # Here we use fan‑based values for consistency.
        left_w, right_w = self._fan_side_clearances(centroid, self.current_heading)

        normal = self.calculate_surface_normal(v1, v2, v3)
        slope, cross = self.calculate_slopes(normal, self.current_heading)

        surf_class = self.get_surface_class_at_position(centroid)
        dist_goal = np.linalg.norm(self.goal_position - centroid)

        return TerrainFeatures(
            step_id=self.step_counter,
            vertex1=v1,
            vertex2=v2,
            vertex3=v3,
            centroid=centroid,
            step_length=step_length,
            open_space_width_left=left_w,
            open_space_width_right=right_w,
            surface_normal=normal,
            slope_angle=slope,
            cross_slope_angle=cross,
            surface_class=surf_class,
            heading_direction=self.current_heading,
            distance_to_goal=dist_goal,
            navigation_state=self.state.name,
            distance_to_nearest_obstacle=min_obs,
            obstacle_distances=obs_dists,
            path_width=path_width,
            centerline_offset=offset,
        )

    def navigate(self, max_steps: int = 1000) -> bool:
        print("Starting sidewalk centerline navigation...")
        for step in range(max_steps):
            self.step_counter = step

            if self.planned_route is not None and self.route_index < len(self.planned_route):
                delta = self.planned_route[self.route_index] - self.current_position
                delta[2] = 0.0
                self.current_heading = delta / (np.linalg.norm(delta) + 1e-6)

            features = self.extract_terrain_features()
            self.terrain_features.append(features)
            self.path_points.append(features.centroid.copy())

            print(
                f"Step {step:4d} | "
                f"dist_goal={features.distance_to_goal:.2f}m | "
                f"state={features.navigation_state:18s} | "
                f"path_width={features.path_width:.3f}m | "
                f"offset={features.centerline_offset:.3f}m | "
                f"min_obs={features.distance_to_nearest_obstacle:.2f}m"
            )

            if self.centerline_navigation_step():
                if self.state == NavigationState.GOAL_REACHED:
                    print(f"Goal reached in {step + 1} steps!")
                    return True
                print(f"Navigation blocked after {step + 1} steps: no safe ground-supported step")
                return False

        print(f"Navigation terminated after {max_steps} steps")
        return False

    def save_results(self, output_file: str):
        print(f"Saving results to {output_file}...")
        rows = []
        for f in self.terrain_features:
            row = {
                "step_id": f.step_id,
                "centroid_x": f.centroid[0],
                "centroid_y": f.centroid[1],
                "centroid_z": f.centroid[2],
                "step_length": f.step_length,
                "open_space_width_left": f.open_space_width_left,
                "open_space_width_right": f.open_space_width_right,
                "surface_normal_x": f.surface_normal[0],
                "surface_normal_y": f.surface_normal[1],
                "surface_normal_z": f.surface_normal[2],
                "slope_angle": f.slope_angle,
                "cross_slope_angle": f.cross_slope_angle,
                "surface_class": f.surface_class,
                "heading_x": f.heading_direction[0],
                "heading_y": f.heading_direction[1],
                "heading_z": f.heading_direction[2],
                "distance_to_goal": f.distance_to_goal,
                "navigation_state": f.navigation_state,
                "distance_to_nearest_obstacle": f.distance_to_nearest_obstacle,
                "path_width": f.path_width,
                "centerline_offset": f.centerline_offset,
            }
            for k, v in f.obstacle_distances.items():
                row[f"obstacle_{k}"] = v
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"Results saved with {len(rows)} steps")
        print(
            f"Total path length: "
            f"{sum(f.step_length for f in self.terrain_features):.2f}m"
        )
