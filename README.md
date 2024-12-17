# User Guide

## Run the code

- install docker
- install python
- install pip
- Open a terminal and run docker-compose up
- install poetry
- run `poetry install`
- run `prisma generate`
- run `prisma db push`
- run `poetry run`

# Workflow:

## Reproducing Thompson et al. 2014

- Read satellite altimeter data
- Change longitude to a range between -180 and 180 (it is in 0 to 360 if aviso dataset is used)
- Transform CRS to geocentric CRS in meters (EPSG:4978)  (*CRS not specified in the paper*)
- filter spatially with a symmetric Gaussian filter of half-width 500 km
- interpolate to 5 degree grid (*method not specified in the paper - chose linear*)
- Apply a convolution low-pass filter passing 90% of the amplitude at 24 months to each time series.  (To emphasize
  inter annual and longer variability)
- implement distance function between two grid points x_i and x_j - D(x_i, x_j) = 1 - exp(- d(x_i, x_j)/2a^2) r(x_i,
  x_j)
    - d is Euclidean distance, r is temporal correlation coefficient, a is constant such that the value of the
      exponential
      is 0.5, when d=3000 km
    - assumption: r is Pearson's correlation coefficient (*not specified in the paper*)
    - r = (sum(x_i - x_mean)(y_i - y_mean)) / sqrt(sum(x_i - x_mean)^2 sum(y_i - y_mean)^2) (range from -1 to 1)
- calculate distances between each pair of grid points
- hierarchical clustering
    - average linkage: distance between two clusters is average distance between data points in the first cluster and
      data points in the second cluster
    - at each stage combine the two clusters with the smallest average linkage distance

## Alternative way of finding start-clusters?

## Different distance function?

- Can we enforce spatial proximity differently?
- We have a grid, what can we do with this?

## Subspace Clustering