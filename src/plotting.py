import os

import xarray as xr
from cartopy import crs as ccrs
from matplotlib import pyplot as plt


def plot_sla_for_point_in_time(sea_level_anomaly_data: xr.Dataset, out_dir: str, feature, name: str):
    """
    Plot sea level anomaly for one point in time
    :param feature:
    :param name:
    :param out_dir:
    :param sea_level_anomaly_data:
    :return:
    """
    # plot data for one point in time
    data = sea_level_anomaly_data[feature].isel(time=0)
    fig = plt.figure(figsize=(50, 25))
    ax = plt.axes(projection=ccrs.PlateCarree())
    data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='jet', add_colorbar=True)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.savefig(os.path.join(out_dir, f"{name}.svg"), dpi=500)
    plt.close(fig)
