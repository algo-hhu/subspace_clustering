# User Guide

## How to Run

1. install python
2. install pip
3. Install [Poetry](https://python-poetry.org/).
4. Install dependencies:  ```poetry install```
5. Run the application ```poetry run python main.py```

# Configuration
All parameters can be configured in the settings.py file.
Each setting also applies to the steps following after it. If the method for initial clustering is set to `agglomerative_clustering` with `spatio_temporal_distance_function`, then this is the input clustering for the subspace clustering.
| Setting                                | Description                                                                                                                                                                                                                                                                                                                                                         |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `output_path`                          | Path where all output files will be saved (e.g., `"../output"`).                                                                                                                                                                                                                                                                                                    |
| `data_path`                            | Path where input data and intermediate modifications are saved (e.g., `"../data"`).                                                                                                                                                                                                                                                                                 |
| `sea_level_anomaly_data_download_path` | Path to the folder containing NetCDF files downloaded from the [Copernicus Marine Service](https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/files?subdataset=cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1M-m_202411&path=SEALEVEL_GLO_PHY_L4_MY_008_047%2F`). The program expects **all NetCDF files to be in a single folder**. *(e.g. "../data/SEALEVEL_GLO_PHY_L4_MY_008_047")*|
| `variable`                             | Name of the variable in the NetCDF files (default: `"sla"`).                                                                                                                                                                                                                                                                                                        |
| `resolution`                           | Coarsening for the grid (original resolution is 0.25° × 0.25°; e.g., `2` means 2° × 2°).                                                                                                                                                                                                                                                                 |
| `filtering_sla`                        | Set to `True` to apply spatial and temporal filtering (recommended when using `spatio_temporal_distance_function`).                                                                                                                                                                                                                                                 |
| `half_width`                           | Half-width of the Gaussian kernel used in filtering (in kilometers; e.g., `500`).                                                                                                                                                                                                                                                                                   |
| `filtered_data_path`                   | Path where the filtered data will be saved. You can usually leave this as-is.                                                                                                                                                                                                                                                                                       |

| Setting              | Description                                                                                                                                                                                    |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `method`             | Clustering method to use. Options: <br> - `agglomerative_clustering` <br> - `agglomerative_connected_clustering` <br> - `k_means_clustering_with_connectivity` <br> - `wards_method_connected` |
| `distance_function`  | Distance metric used. Options: <br> - `euclidean` <br> - `spatio_temporal_distance_function` <br> *Note: k-means and Ward's method always use Euclidean distance.*                             |
| `number_of_clusters` | List of cluster counts to save from the iterative process (e.g., `[25, 20, 15, 12, 10, 8]`).                                                                                                   |


| Setting                   | Description                                                                          |
| ------------------------- | ------------------------------------------------------------------------------------ |
| `apply_weights`           | Whether to weight grid points by cosine of latitude to adjust for Earth's curvature. |
| `do_subspace_clustering`  | Set to `True` to apply subspace clustering as an optimization step.                  |
| `number_of_clusters`      | Number of clusters to use in subspace clustering (e.g., `15`).                       |
| `number_of_components`    | List of dimensions to use when calculating subspaces (e.g., `[5, 10, 15, 30]`).      |
| `integrated_connectivity` | Set to `True` to use integrated connectivity in subspace clustering.                 |


| Setting              | Description                                           |
| -------------------- | ----------------------------------------------------- |
| `do_evaluation`      | Set to `True` to evaluate the resulting clusters.     |
| `number_of_clusters` | Which clustering (by number of clusters) to evaluate. |



## Dependencies

All dependencies are managed with Poetry. See `pyproject.toml` for full details.

Main dependencies include:
- `xarray` for NetCDF file handling
- `scikit-learn` for clustering
- `numpy`, `pandas`, `matplotlib` for data processing and plotting
- `pydantic` and `pydantic-settings` for configuration



# Workflow:
## Data preprocessing: 
### Reading data 
### Filtering data
### Interpolating to coarser grid
## Input clustering
## Subspace clustering



## Reproducing Results from Thompson et al. (2014)

To replicate the methodology from *Thompson et al., 2014*, the following steps were applied based on information in the paper and reasonable assumptions where details were missing:

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





## Subspace Clustering