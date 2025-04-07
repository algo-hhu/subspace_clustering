import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import math
import numpy as np
from tqdm import tqdm
import random

def distance_function(lat1, long1, timeseries1, lat2, long2, timeseries2):
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


run = 0

while run < 5:

    ax = plt.axes(projection = ccrs.PlateCarree())
    ax.coastlines()
    ax.set_global()

    #Hardcoded center coordinates
    #center_lons = [156, -133, -67 , -35, 147, -2]
    #center_lats = [25, 10, 29, 60, -55, -43]
    cluster_colors = ['red','yellow','green','cyan','blue','purple']


    path = "../Data/Sea_Level_Data/sea_level_anomaly_data.nc"
    data = xr.open_dataset(path)
    sea_level = data['sla']
    #sea_level.isel(time = 0).plot(ax=ax, transform=ccrs.PlateCarree(), cmap='jet', add_colorbar=True)


    valid_points = ~sea_level.isnull().any(dim='time')
    print(valid_points)

    #filtered_sea_level = sea_level.where(valid_points,drop=True)

    all_lats = sea_level['latitude'].values
    all_lons = sea_level['longitude'].values

    k = 6
    i = 0

    centers = []
    center_lons =[]
    center_lats = []

    while i < k:
        lon_id = random.sample(range(len(all_lons)),1)[0]
        lat_id = random.sample(range(len(all_lats)),1)[0]

        print(lon_id, lat_id)

        if valid_points[lat_id][lon_id]:
            centers.append(sea_level.sel(latitude = all_lats[lat_id], longitude = all_lons[lon_id]).values[:48]) 
            center_lons.append(all_lons[lon_id])
            center_lats.append(all_lats[lat_id])
            i = i + 1
        

    #get nearest entities in data array for hardcoded centers

    #for i in tqdm(range(len(center_lats))):

    #    tmp = sea_level.sel(latitude = center_lats[i], longitude = center_lons[i], method = 'nearest')
    #    centers.append(tmp.values)
    #    center_lons[i] = float(tmp['longitude'])
    #    center_lats[i] = float(tmp['latitude'])
        
    #plt.scatter(center_lons,center_lats,color = 'red' , marker = 'x')
    #plt.show()

    clustering_variable = np.zeros((sea_level.latitude.size, sea_level.longitude.size))
    clustering_var_da = xr.DataArray(data=clustering_variable[:, :],
                                    dims=['latitude', 'longitude'],
                                    coords={ 'latitude': sea_level.latitude, 'longitude': sea_level.longitude })

    #compute clustering
    for i,lat in tqdm(enumerate(all_lats)):
        for j,lon in enumerate(all_lons):
            if valid_points[i][j]:
                time_series = sea_level.loc[:,lat, lon].values[:48]
                min_dist = math.inf
                #find closest center
                for k in range(len(centers)):
                    c = centers[k]
                    #tmp = math.dist(time_series,c)
                    tmp = math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(time_series, c)))
                    #tmp = distance_function(lat,lon,time_series,center_lats[k],center_lons[k],c)
                    if tmp < min_dist:
                        min_dist = tmp
                        assignment = k
                clustering_var_da[i,j] = assignment
            else:
                clustering_var_da[i,j] = 'nan'


    clustering_var_da.plot()

    plt.scatter(center_lons,center_lats, facecolor = 'red' , marker = 's', edgecolor = 'black')
    plt.savefig(fname = "Euclidean" + str(run + 1))
    run = run + 1




