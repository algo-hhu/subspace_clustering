import math

import numpy as np


def spatio_temporal_distance_function(lat1: float, long1: float, timeseries1: [float], lat2: float, long2: float,
                                      timeseries2: [float]):
    """
    Calculate the distance function between two points D(x_i, x_j) = 1 - exp(- d(x_i, x_j)/2a^2) r(x_i, x_j)
    :param timeseries2:
    :param long2:
    :param lat2:
    :param timeseries1:
    :param long1:
    :param lat1:
    :return:
    """
    a = math.sqrt(- (1500 / (math.log(0.5))))
    earth_radius = 6371  # km
    lat1, lat2, long1, long2 = map(np.radians, [lat1, lat2, long1, long2])
    delta_phi = lat2 - lat1
    delta_lambda = long2 - long1
    haversine_distance = 2 * earth_radius * np.arcsin(
        np.sqrt(np.sin(delta_phi / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lambda / 2) ** 2)
    )

    # Pearsons correlation coefficient
    r = np.corrcoef(timeseries1, timeseries2)[0, 1]

    # calculate difference
    difference = 1 - np.exp(-haversine_distance / (2 * a ** 2)) * r
    return difference


def test_distance_function(lat1, long1, time_series1, lat2, long2, time_series2):
    return abs(sum(time_series1) - sum(time_series2))


def euclidean_distance(lat1: float, long1: float, timeseries1: [float], lat2: float, long2: float,
                       timeseries2: [float]):
    """

    :param lat1:
    :param long1:
    :param timeseries1:
    :param lat2:
    :param long2:
    :param timeseries2:
    :return:
    """
    distance = np.linalg.norm(np.array(timeseries1) - np.array(timeseries2))
    return distance


def distance_for_wards_method(timeseries1: [float], timeseries2: [float]):
    """

    :param timeseries1:
    :param timeseries2:
    :return:
    """
    distance = np.sum((np.array(timeseries1) - np.array(timeseries2)) ** 2)
    return distance


def subspace_timeseries_distance_calculation(all_distances, current_time_series, mean, subspace):
    """
    Calculate the distance of the current time series to the subspace
    :param all_distances:
    :param current_time_series:
    :param mean:
    :param subspace:
    :return:
    """
    distance = 0
    current_time_series_for_cluster = current_time_series - mean
    # project current time series onto subspace
    projection = subspace.T @ (subspace @ current_time_series_for_cluster)
    # use squared Euclidean distance
    residual = current_time_series_for_cluster - projection
    distance = np.sum(residual ** 2)
    all_distances.append(distance)
    # otherwise could use the norm
    # distance = np.linalg.norm(current_time_series_for_cluster - x_proj)
    # if distance is less than the previous ones, update the minimum
    return distance
