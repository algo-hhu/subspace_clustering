import numpy as np
from tqdm import tqdm


def find_neighbors(sea_level_anomaly_data, nan_mask, lat_lon_to_grid_point_id):
    """
    Find neighbors for each grid point in the latitudes and longitudes arrays.

    :param sea_level_anomaly_data:
    :param lat_lon_to_grid_point_id:
    :param nan_mask: Mask indicating NaN values in the data array.
    :return:
    """
    # Initialize variables

    lat_range = sea_level_anomaly_data["latitude"].shape[0]
    latitudes = sea_level_anomaly_data.latitude.values
    long_range = sea_level_anomaly_data["longitude"].shape[0]
    longitudes = sea_level_anomaly_data.longitude.values
    data = sea_level_anomaly_data["sla"].values
    lat_range = len(latitudes)
    long_range = len(longitudes)

    first_longitude, last_longitude, lat_for_first_longitude, lat_for_last_longitude = find_first_last_longitude(
        lat_range, long_range, nan_mask)
    iteratively_find_neighbors(data, latitudes, longitudes, lat_range, long_range, first_longitude, last_longitude,
                               nan_mask, lat_lon_to_grid_point_id)


def iteratively_find_neighbors(data, latitudes, longitudes, lat_range, long_range, first_longitude, last_longitude,
                               nan_mask, lat_lon_to_grid_point_id):
    """

    :return:
    """
    neighbors = {}  # {grid_point_id: {neighbor_grid_point1, neighbor_grid_point2, ...}}
    # iterate through latitudes and longitudes and find neighbors for each grid point
    for i in tqdm(range(lat_range)):
        for j in (range(long_range)):
            if nan_mask[i, j]:  # points without valid data can be skipped
                continue
            neighbors[
                lat_lon_to_grid_point_id[latitudes[i], longitudes[j]]] = set()  # set of neighbors for each grid point
            # direct neighbors
            neighbor_positions = [
                (i - 1, j),  # North
                (i + 1, j),  # South
                (i, (j - 1) % long_range),  # West (wraps around)
                (i, (j + 1) % long_range),  # East (wraps around)
            ]
            # check if the grid point is on the edge of the grid, because of the interpolation there might be nan values at the edges
            if j == last_longitude:
                neighbor_positions.extend([(i, first_longitude), (i - 1, first_longitude), (i + 1, first_longitude)])
            if j == first_longitude:
                neighbor_positions.extend([(i, last_longitude), (i - 1, last_longitude), (i + 1, last_longitude)])
            # diagonal neighbors
            neighbor_positions.extend([
                ((i - 1), (j - 1) % long_range),  # Northwest
                ((i - 1), (j + 1) % long_range),  # Northeast
                ((i + 1), (j - 1) % long_range),  # Southwest
                ((i + 1), (j + 1) % long_range),  # Southeast
            ])
            # Handle out-of-bounds positions
            neighbor_positions_without_out_of_bounds = [
                (pos[0], pos[1]) if 0 <= pos[0] < lat_range else None
                for pos in neighbor_positions
            ]
            valid_neighbor_positions = [(pos[0], pos[1]) for pos in neighbor_positions_without_out_of_bounds if
                                        not nan_mask[pos[0], pos[1]]]
            # add valid neighbors to the set of neighbors
            grid_point1 = lat_lon_to_grid_point_id[latitudes[i], longitudes[j]]
            for pos in valid_neighbor_positions:
                if pos is not None and not nan_mask[pos[0], pos[1]]:
                    grid_point2 = lat_lon_to_grid_point_id[latitudes[pos[0]], longitudes[pos[1]]]
                    neighbors[grid_point1].add(grid_point2)

    neighbors = ensure_bidirectional_neighbors(neighbors)
    return neighbors


def find_first_last_longitude(lat_range, long_range, nan_mask):
    """
    Find the first and last longitude that has data
    :param lat_range:
    :param long_range:
    :param nan_mask:
    :return:
    """
    # find first and last longitude that has data
    first_longitude = np.inf
    lat_for_first_longitude = np.inf
    last_longitude = 0
    lat_for_last_longitude = 0
    for i in range(long_range):
        for j in range(lat_range):
            if not nan_mask[j, i]:
                if i < first_longitude:
                    first_longitude = i
                    lat_for_first_longitude = j
                continue
    for i in reversed(range(long_range)):
        for j in range(lat_range):
            if not nan_mask[j, i]:
                if i > last_longitude:
                    last_longitude = i
                    lat_for_last_longitude = j
                continue
    return first_longitude, last_longitude, lat_for_first_longitude, lat_for_last_longitude


def ensure_bidirectional_neighbors(neighbors: {}):
    """
    Ensure that neighbor-relationships are bidirectional
    :param neighbors:
    :return:
    """
    for key, value in neighbors.items():
        for neighbor in value:
            if key not in neighbors[neighbor]:
                neighbors[neighbor].append(key)
    return neighbors
