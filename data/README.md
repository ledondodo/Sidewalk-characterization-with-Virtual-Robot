# Input data

Large point clouds are not stored in this repository. The sample runner uses
the recordings in the neighboring `Data Samples/LiDAR Recordings/` folder by
default, so there is no need to copy them into this directory. To use a file
elsewhere, pass its path with `--input`.

## Supported class dimensions

The loader detects these point-cloud dimensions:

| Dimension | Default navigable values |
|---|---|
| `Label` | 1111–1116 and 1810 (paper dataset) |
| `ModelClass` | 1 (ground in the supplied samples) |
| `classification` | 2 (LAS standard ground class) |

For a different label map, set `NavigationConfig.navigable_classes` to the
values representing the walkable surface.

The Victoriaville dataset used in the paper is available from [Zenodo](https://doi.org/10.5281/zenodo.19354886).
