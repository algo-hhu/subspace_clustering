# Connected Subspace Clustering

This repository accompanies the paper **_Connected Subspace Clustering: Hardness, a Scalable Heuristic, and an
Application to Sea Level Geodesy_**.
It implements the *Connected Subspace Clustering* (Conn-Subspace) algorithm (Alg. 1 in the paper), the
initial-clustering methods, and the connectivity subroutines (Algs. 2 and 3), and applies them to global
sea level anomaly (SLA) data from satellite altimetry.

> **Scope.** This code reproduces *our* methods: the initial clusterings, the four connectivity strategies,
> the Conn-Subspace pipeline, the preprocessing (spherical Gaussian + temporal filtering), and the
> sea-level analysis plots. The external baselines compared against in the paper — SSC-OMP, EnSC (App. D.3),
> and the HSI methods EGCSC/EKGCSC (App. D.4) — are **not** part of this repository; they were run with the
> authors' original code and are not reproduced here.

---

## How to Run

1. install python (3.12)
2. install pip
3. Install [Poetry](https://python-poetry.org/).
4. Install dependencies:  ```poetry install```
5. Run the application:

   ```bash
   poetry run python -m src.main
   ```

   All input and output paths are anchored to the project root, so the program can be launched from any
   working directory. (Equivalently: `cd src && poetry run python main.py`.)

Preprocessing the full-resolution dataset and filtering it is memory- and time-intensive. All expensive
intermediate results are **cached on disk**, so re-running the program skips any step whose output already
exists (filtered field, coarsened field, initial clusterings, completed subspace-clustering runs).

---

## Configuration

All parameters can be configured in the `src/settings/settings.py` file. The four settings classes are plain
`pydantic` `BaseModel`s — values come **only** from the file (there is no environment-variable override).
Each setting also applies to the steps following after it. If the method for initial clustering is set to
`agglomerative_clustering` with `spatio_temporal_distance_function`, then this is the input clustering for the
subspace clustering.

### `GlobalSettings`

| Setting                                | Description                                                                                                                                                                                                                                                                  |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `output_path`                          | Path where all output files will be saved. Defaults to `<repo>/output` (anchored to the project root, independent of the working directory).                                                                                                                                |
| `data_path`                            | Path where input data and intermediate modifications are saved. Defaults to `<repo-parent>/data`.                                                                                                                                                                           |
| `sea_level_anomaly_data_download_path` | Path to the folder containing the NetCDF files downloaded from the [Copernicus Marine Service](https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/). The program expects **all NetCDF files to be in a single folder**. See *Data* below.            |
| `variable`                             | Name of the variable in the NetCDF files (default: `"sla"`).                                                                                                                                                                                                                |
| `resolution`                           | Coarsening for the grid (original resolution is 0.25° × 0.25°; e.g. `2` means 2° × 2°). The paper's experiments use `2`.                                                                                                                                                     |
| `filtering_sla`                        | Set to `True` to apply spatial and temporal filtering (recommended, and required when using `spatio_temporal_distance_function`).                                                                                                                                           |
| `half_width`                           | Half-width of the Gaussian kernel used in spatial filtering (in kilometers; e.g. `500`).                                                                                                                                                                                    |
| `random_seed`                          | Seed for all randomness (NumPy/`random` global RNGs, scikit-learn k-means, and PCA's randomized solver), making runs reproducible. Default `42`.                                                                                                                            |
| `filtered_data_path`                   | *(Derived, read-only.)* Computed from `output_path` and `half_width`; it is no longer a settable field. The filtered field is cached here.                                                                                                                                  |

### `InitialClusteringSettings`

| Setting              | Description                                                                                                                                                                                    |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `method`             | Clustering method to use. Options: <br> - `agglomerative_clustering` <br> - `agglomerative_connected_clustering` <br> - `k_means_clustering_with_connectivity` <br> - `wards_method_connected` |
| `distance_function`  | Distance metric used. Options: <br> - `euclidean` <br> - `spatio_temporal_distance_function` <br> *Note: k-means and Ward's method always use Euclidean distance.*                             |
| `number_of_clusters` | List of cluster counts to save from the iterative process (e.g. `[25, 20, 15, 12, 10, 8]`).                                                                                                   |

### `SubspaceClusteringSettings`

| Setting                   | Description                                                                          |
| ------------------------- | ----------------------------------------------------------------------------------- |
| `apply_weights`           | Whether to weight grid points by cosine of latitude to adjust for Earth's curvature. |
| `do_subspace_clustering`  | Set to `True` to run Conn-Subspace with the **IterMerge**, **PostMerge** and **SmoothMerge** connectivity strategies. |
| `number_of_clusters`      | Number of clusters `k` the subspace clustering operates on (e.g. `25`). The initial clustering with this many clusters is used as the starting point. |
| `number_of_components`    | List of subspace dimensions `m'` to evaluate (e.g. `[5, 10, 15, 30]`).               |
| `integrated_connectivity` | Set to `True` to additionally run the **IntegratedConn** strategy (Alg. 3).         |

### `EvaluationSettings`

| Setting              | Description                                                            |
| -------------------- | -------------------------------------------------------------------- |
| `do_evaluation`      | Set to `True` to evaluate the resulting clusters (per-cluster EOFs/PCs). |
| `number_of_clusters` | Which clustering (by number of clusters) to evaluate.                 |

---

## Data

The experiments use the CMEMS *Global Ocean Gridded L4 Sea Surface Heights And Derived Variables Reprocessed
(1993–ongoing)* product (`SEALEVEL_GLO_PHY_L4_MY_008_047`), monthly SLA on a 0.25° × 0.25° grid
([Copernicus Marine Service](https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/)).

1. Download the monthly NetCDF files into **one** folder.
2. Point `sea_level_anomaly_data_download_path` at that folder.
3. On the first run, the program reads and concatenates all files, converts longitudes from `[0°, 360°]` to
   `[-180°, 180°]`, and caches the merged dataset as `<data_path>/sea_level_anomaly_data.nc`. Subsequent runs
   reuse this cache.

After removing land, coastlines and high latitudes, 532,783 ocean grid points with complete time series remain
(see paper Sec. 4.1).

---

## Mapping paper method names to settings

The paper refers to methods by short names (Sec. 4.2). To reproduce a given row of the result tables, set the
corresponding settings.

### Initial clustering methods

| Paper name        | `method`                                | `distance_function`                 |
| ----------------- | --------------------------------------- | ----------------------------------- |
| **Agglo-ST**      | `agglomerative_clustering`              | `spatio_temporal_distance_function` |
| **Conn-Agglo-Euc**| `agglomerative_connected_clustering`    | `euclidean`                         |
| **Conn-Agglo-ST** | `agglomerative_connected_clustering`    | `spatio_temporal_distance_function` |
| **Conn-KMeans++** | `k_means_clustering_with_connectivity`  | *(Euclidean — always)*              |
| **Conn-Ward**     | `wards_method_connected`                | *(Ward objective — always Euclidean)* |

### Connectivity strategies (produced during Conn-Subspace)

| Paper name        | How to enable                          | Output subdirectory                       |
| ----------------- | -------------------------------------- | ----------------------------------------- |
| **IterMerge**     | `do_subspace_clustering = True`        | `establish_connectivity_every_iteration/` |
| **PostMerge**     | `do_subspace_clustering = True`        | `establish_connectivity_once/`            |
| **SmoothMerge**   | `do_subspace_clustering = True`        | `filter_every_round_connectivity_once/`   |
| **IntegratedConn**| `integrated_connectivity = True`       | `integrated_connectivity/`                |

A single run with `do_subspace_clustering = True` produces **IterMerge, PostMerge and SmoothMerge together**,
for every subspace dimension `m'` in `number_of_components`.

---

## Output structure

```
output/
├── results_<k>_clusters.csv                      # LaTeX-ready summary of distances to subspaces (per run)
├── resolutions/                                  # cached coarsened SLA fields
├── spherical_gaussian_filtering/                 # cached spatially filtered SLA field
└── filter_<half_width>/                          # or  no_filtering/   when filtering_sla = False
    └── <resolution>_degree_grid/
        └── <method>_<distance_function>/         # e.g. agglomerative_clustering_spatio_temporal_distance_function
            ├── clustering_<k>.nc                  # initial clusterings, one per k in number_of_clusters
            └── subspace_clustering_<k>/           # k = SubspaceClusteringSettings.number_of_clusters
                ├── establish_connectivity_every_iteration/   # IterMerge
                │   └── components_<m'>/
                │       ├── clustering_<k>.nc                  # final connected clustering
                │       ├── final_results_<m'>.txt            # best/total iterations, start & best cost
                │       └── sum_distances_to_subspaces.pkl    # cost trajectory
                ├── establish_connectivity_once/              # PostMerge
                ├── filter_every_round_connectivity_once/     # SmoothMerge
                └── integrated_connectivity/                  # IntegratedConn
```

The per-run `results_<k>_clusters.csv` collects the start and best "distance to subspace" values
(the subspace clustering cost) for each strategy and `m'`, which are the numbers reported in the paper's tables.

---

## Reproducing the paper's experiments

The main evaluation (Sec. 4.5–4.6, Tables 5–12) sweeps **5 initial clusterings × 4 cluster counts
`k ∈ {8, 15, 20, 25}` × 4 subspace dimensions `m' ∈ {5, 10, 15, 30}` × {filtered, unfiltered} = 160 experiments**,
each compared across the four connectivity strategies.

Because one run already covers all `m'` (a list) and all four connectivity strategies, the full sweep is
**40 runs**: for each of the 5 method configurations, for `filtering_sla ∈ {True, False}`, for
`k ∈ {8, 15, 20, 25}`, set:

- `InitialClusteringSettings.method` / `distance_function` per the mapping table above,
- `InitialClusteringSettings.number_of_clusters` includes the chosen `k` (the default `[25, 20, 15, 12, 10, 8]` covers all four),
- `GlobalSettings.filtering_sla` = `True` / `False`,
- `SubspaceClusteringSettings.number_of_clusters` = `k`,
- `SubspaceClusteringSettings.number_of_components` = `[5, 10, 15, 30]`,
- `SubspaceClusteringSettings.do_subspace_clustering` = `True` and `integrated_connectivity` = `True`,

then run the program. Collect the costs from the `results_<k>_clusters.csv` files and the
`final_results_<m'>.txt` / `sum_distances_to_subspaces.pkl` files in each output subdirectory.

- **Tables 5–12** (distances from points to subspaces): the `initial clustering` / `IterMerge` / `PostMerge` /
  `SmoothMerge` / `IntegratedConn` columns correspond to the start cost and the four output subdirectories.
- **Fig. 3** (subspace vs. k-means cost across initializations) and **Fig. 4** (Conn-Subspace vs. Conn-Ward):
  derived from the costs above across `k`.
- **Fig. 9** (cost trajectories per strategy): from `sum_distances_to_subspaces.pkl`.
- **Per-cluster EOFs/PCs and ENSO comparison** (Figs. 5, 8, Sec. 4.7): set `EvaluationSettings.do_evaluation = True`
  and `number_of_clusters` to the clustering you want to analyse.

> **Note on reproducibility.** Randomness is seeded via `GlobalSettings.random_seed` (default `42`) — this
> covers k-means and PCA's randomized solver — so a given seed produces the same results across runs. Change
> the seed to assess sensitivity to initialization.

---

## Preprocessing pipeline

Implemented in `src/preprocessing/` and `src/helper.py`, matching paper App. D.1 / B.2:

1. **Read & merge** all NetCDF files over the time dimension; convert longitude `[0°,360°] → [-180°,180°]` and sort.
2. **Spatial filtering** (if `filtering_sla`): a parallelized, curvature-aware **spherical Gaussian filter**
   using Haversine (great-circle) distances, with half-width `half_width` km (σ = h/1.178, cutoff radius 3σ),
   robust to missing values. Cached under `output/spherical_gaussian_filtering/`.
3. **Temporal filtering**: a 15-month centered rolling mean (low-pass boxcar; ≈ 90 % gain at a 24-month period).
4. **Coarsening**: interpolation to a `resolution`° grid. Cached under `output/resolutions/`.
5. **Latitude weighting** (subspace stage, if `apply_weights`): multiply SLA by `cos(latitude)`.

---

## Reproducing Results from Thompson et al. (2014)

To replicate the methodology from *Thompson et al., 2014* (the **Agglo-ST** baseline), the following steps were
applied based on information in the paper and reasonable assumptions where details were missing:

1. **Read satellite altimeter data**
   - Use SLA data (e.g., from AVISO or Copernicus).

2. **Adjust longitude range**
   - Convert longitudes from [0°, 360°] to [-180°, 180°], as expected in many spatial libraries.

3. **Transform Coordinate Reference System (CRS)**
   - Transform data to a geocentric CRS in meters (EPSG:4978).
     *Note: The original paper does not specify the CRS.*

4. **Apply spatial filtering**
   - Use a symmetric Gaussian filter with a half-width of **500 km** to smooth spatial noise.

5. **Apply temporal filtering**
   - Each time series is smoothed with a low-pass convolution filter that retains **90% of the amplitude at a 24-month period**.
   - This step emphasizes interannual and longer-term variability.

6. **Interpolate to coarser grid**
   - Resample data to a **5° × 5°** regular grid.
     *Note: Interpolation method not specified in the paper — we used **linear interpolation**.*

7. **Define custom distance function between grid points**

   The dissimilarity between two grid points \( x_i \) and \( x_j \) is defined as:

   \[
   D(x_i, x_j) = 1 - \exp\left(-\frac{d(x_i, x_j)}{2a^2}\right) \cdot r(x_i, x_j)
   \]

   where:
   - \( d(x_i, x_j) \) is the **Euclidean distance** between the two grid points.
   - \( r(x_i, x_j) \) is the **temporal correlation coefficient** (assumed to be Pearson's \( r \)).
   - \( a \) is a scaling constant such that:

     \[
     \exp\left(-\frac{3000^2}{2a^2}\right) = 0.5 \quad \Rightarrow \quad a \approx 2550.76\ \text{km}
     \]

   - Pearson correlation coefficient is defined as:

     \[
     r = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i - \bar{x})^2 \cdot \sum_i (y_i - \bar{y})^2}}
     \]

     *Note: The paper does not specify whether Pearson or another correlation method is used.*

8. **Compute distance matrix**
   - Calculate pairwise distances between all grid points using the function above.

9. **Perform hierarchical clustering**
   - Use **average linkage** (i.e., the distance between clusters is the average of all pairwise distances between their members).
   - At each iteration, merge the two clusters with the smallest average linkage distance and recalculate distances to all other clusters.

---

## Tests

```bash
cd test
PYTHONPATH=.. poetry run python -m unittest discover -s . -p "test_*.py"
```

This is the same invocation used in CI (`.github/workflows/ci.yml`).

---

## Dependencies

All dependencies are managed with Poetry. See `pyproject.toml` for full details. Main dependencies include:

- `xarray` + `netcdf4` for NetCDF file handling
- `scikit-learn` for clustering (agglomerative, Ward, k-means++) and PCA
- `numpy`, `scipy`, `pandas` for numerical processing
- `matplotlib`, `cartopy`, `geopandas` for plotting on maps
- `joblib` for parallelized filtering
- `pydantic` for the settings models
- `loguru` for logging

---

## Workflow overview

1. **Data preprocessing** — read & merge, spatial + temporal filtering, coarsen to the target resolution.
2. **Initial clustering** — one of the five methods (see mapping table), producing `clustering_<k>.nc`.
3. **Subspace clustering** — Conn-Subspace (Alg. 1) with the selected connectivity strategies.
4. **Evaluation** *(optional)* — per-cluster EOF/PC analysis and ENSO comparison.
