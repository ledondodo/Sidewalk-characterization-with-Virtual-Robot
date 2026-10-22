
# Characterizing Pedestrian Network from Segmented 3D Point Clouds for Accessibility Assessment: A Virtual 
Robotic Approach


A Python implementation of a **virtual robotic approach** for automated characterization
of pedestrian network accessibility from semantically segmented 3D LiDAR point clouds.

Developed as part of the research described in:

> **Ahmadi, A., Mostafavi, M.A., Morales, E., & Sabo, N. (2026).**
> Characterizing Pedestrian Network from Segmented 3D Point Clouds for Accessibility
> Assessment: A Virtual Robotic Approach.
> *Sensors, 26*. https://doi.org/10.3390/s26072172

---

## Overview

The method navigates a classified 3D LiDAR point cloud using a **triangular virtual robot**
whose side length matches a standard wheelchair footprint. At each step, it records
terrain and navigation features aligned with the **Measure of Environmental Accessibility (MEA)**
framework and **ADA / IBC** accessibility standards.

Key features:
- Loads classified `.las` point clouds with semantic labels
- Separates navigable surfaces (sidewalks, ramps, curb cuts) from obstacles
- Estimates left and right clearances using **fan-based ray casting** (±15°, 5 rays/side)
- Tracks the **sidewalk centerline** with exponential heading smoothing
- Adapts to 3D terrain by conforming robot vertices to the local surface
- Exports per-step accessibility parameters to CSV for analysis

### Measured Parameters

| Parameter | MEA Feature | ADA/IBC Threshold |
|---|---|---|
| `path_width` | Path width | ≥ 0.91 m (single), ≥ 1.52 m (passing) |
| `slope_angle` | Running slope | ≤ 5% (2.86°) |
| `cross_slope_angle` | Cross slope | ≤ 2% (1.15°) |
| `surface_class` | Surface type | — |
| `distance_to_nearest_obstacle` | Obstacle clearance | — |
| `centerline_offset` | Lateral deviation from centre | — |

> The algorithm records **continuous raw measurements**, not binary pass/fail values,
> enabling personalized accessibility assessment for individual user profiles.

---

## Repository Structure

```text
sidewalk-centerline-navigation/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
│
├── src/
│   └── centerline_navigation.py    ← core NavigationConfig, TerrainFeatures, CenterlineNavigator
│
├── examples/
│   └── run_example.py              ← example script: configure, run, save results
│
├── data/
│   └── README.md                   ← dataset format, class codes, preparation guide
│
├── results/
│   └── README.md                   ← output CSV column descriptions and MEA mapping
│
└── figures/
    └── example_path.png            ← trajectory visualization (optional)
```

---

## Requirements

- Python 3.10 or newer

| Package | Minimum version |
|---|---|
| numpy | 1.24 |
| laspy | 2.4 |
| pandas | 2.0 |
| scipy | 1.10 |

---

## Installation

```bash
uv sync
```

---

## Usage

```python
from src.centerline_navigation import NavigationConfig, CenterlineNavigator

config = NavigationConfig(
    triangle_side_length=0.8,
    step_size=0.3,
    sensor_range=5.0,
    goal_tolerance=0.2,
    navigable_classes=[1111.0, 1112.0, 1113.0, 1114.0, 1115.0, 1116.0, 1810.0],
    min_passage_width=0.6,
    desired_side_distance=0.5,
    min_side_distance=0.45,
    ray_coarse_step=0.05,
    ray_refine_iters=6,
    width_fan_half_angle_deg=15.0,
    width_fan_rays=5,
)

navigator = CenterlineNavigator(config)
navigator.load_point_cloud("data/your_point_cloud.las")
navigator.initialize_robot(
    start_point=[193328.856628, 5102483.658905, 143.8510],
    end_point=[193311.138123, 5102503.181488, 144.1012],
)
navigator.navigate(max_steps=2000)
navigator.save_results("results/run_output.csv")
```

Or run the ready-made example script:

```bash
uv run python examples/run_example.py
```

The example uses `data/example.las` when present, deriving a start and goal
from its navigable points. Otherwise it looks for
`SlamProject-35-03_classified_with_modelclass.laz` in the neighboring
`Data Samples/LiDAR Recordings/classified_litept-l-mls-5cm/` folder. To use
another recording:

```bash
uv run python examples/run_example.py --input /path/to/recording.laz
```

For very large recordings, limit the randomly sampled points to reduce memory
use:

```bash
uv run python examples/run_example.py --input /path/to/recording.laz --max-points 4000000
```

The supplied `*_with_modelclass.laz` recordings use `ModelClass=1` for ground.
The navigator also recognizes the paper dataset's `Label` field and standard
LAS `classification` (class 2 = ground). For other class maps, set
`NavigationConfig.navigable_classes` explicitly.

---

## Input Data

The algorithm expects a `.las` file with:
- `x`, `y`, `z` coordinates in a projected CRS (meters)
- A `Label` attribute (float) with semantic class codes

The **Victoriaville, Quebec** dataset used in the paper was collected by
[Groupe Trifide](https://www.groupetrifide.com/) and annotated in CloudCompare
into 35 semantic classes. Navigable classes include sidewalks (1111), paths (1112),
parking lots (1113), bare surfaces (1114), curb cuts (1115), and crosswalks (1116).

See [`data/README.md`](data/README.md) for full details.

---

## Output

Each run produces a CSV file with one row per navigation step (~0.30 m).
Key columns: `centroid_x/y/z`, `path_width`, `slope_angle`, `cross_slope_angle`,
`surface_class`, `navigation_state`, `centerline_offset`, and 36 directional
obstacle distances.

See [`results/README.md`](results/README.md) for all column descriptions.

### Validation Accuracy (from the paper)

| Parameter | MAE | NRMSE |
|---|---|---|
| Path width | 0.027 m | **2.71%** |
| Running slope | 0.776° | 66.56% |
| Cross slope | 1.315° | 64.89% |

---

## Method Summary

1. Load and classify point cloud into navigable and obstacle points.
2. Initialize robot position and heading toward the goal.
3. At each step, estimate local path width and centerline offset using fan-based rays.
4. Select a heading that balances goal alignment and centerline tracking.
5. Move one step, conforming robot vertices to the local surface height.
6. Record terrain and navigation features.
7. Export all steps to CSV for accessibility assessment and route planning.

---

## Funding

This work was supported by:
- Canada Research Chairs Program
- Natural Sciences and Engineering Research Council of Canada (NSERC)
- iAccess project — Ministère de l'Économie, de l'Innovation et de l'Énergie du Québec (MEIE)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ahmadi2026sidewalk,
  title   = {Characterizing Pedestrian Network from Segmented 3D Point Clouds
             for Accessibility Assessment: A Virtual Robotic Approach},
  author  = {Ahmadi, Ali and Mostafavi, Mir Abolfazl and Morales, Ernesto and Sabo, Nouri},
  journal = {Sensors},
  volume  = {26},
  year    = {2026},
  doi     = {doi.org/10.3390/s26072172},
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata,
or click **Cite this repository** on the GitHub page.

---

## License

MIT License — see [`LICENSE`](LICENSE).

## Contact

**Ali Ahmadi** · ali.ahmadi.2@ulaval.ca
Center for Research in Geospatial Data and Intelligence (CRDIG)
Département des sciences géomatiques, Université Laval, Quebec City, Canada
"""
