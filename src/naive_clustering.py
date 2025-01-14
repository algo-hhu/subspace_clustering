import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import math
import numpy as np
from tqdm import tqdm

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


ax = plt.axes(projection = ccrs.PlateCarree())
ax.coastlines()
ax.set_global()

#Hardcoded center coordinates
center_lons = [156, -133, -67 , -35, 147, -2]
center_lats = [25, 10, 29, 60, -55, -43]
colors = ['red','yellow','green','cyan','blue','purple']


path = "../Data/Sea_Level_Data/sea_level_anomaly_data.nc"
data = xr.open_dataset(path)
sea_level = data['sla']
#sea_level.isel(time = 0).plot(ax=ax, transform=ccrs.PlateCarree(), cmap='jet', add_colorbar=True)


valid_points = ~sea_level.isnull().any(dim='time')
print(valid_points)
#filtered_sea_level = sea_level.where(valid_points,drop=True)

all_lats = sea_level['latitude'].values
all_lons = sea_level['longitude'].values
#get nearest entities for hardcoded centers

centers = []
for i in tqdm(range(len(center_lats))):

    tmp = sea_level.sel(latitude = center_lats[i], longitude = center_lons[i], method = 'nearest')
    centers.append(tmp.values)
    center_lons[i] = float(tmp['longitude'])
    center_lats[i] = float(tmp['latitude'])
    
print(centers)

assignment = []   
lats = []
lons = []


clustering_variable = np.zeros((sea_level.latitude.size, sea_level.longitude.size))
clustering_var_da = xr.DataArray(data=clustering_variable[:, :],
                                dims=['latitude', 'longitude'],
                                coords={ 'latitude': sea_level.latitude, 'longitude': sea_level.longitude })

#compute clustering
for i,lat in tqdm(enumerate(all_lats[:300])):
    for j,lon in enumerate(all_lons):
        if valid_points[i][j]:
            time_series = sea_level.loc[:,lat, lon].values
            min_dist = math.inf
            #find closest center
            for k in range(len(centers)):
                c = centers[k]
                tmp = distance_function(lat,lon,time_series,center_lats[k],center_lons[k],c)
                if tmp < min_dist:
                    tmp = min_dist
                    assignment = k
                clustering_var_da[i,j] = assignment
        else:
            clustering_var_da[i,j] = 'nan'
clustering_var_da.plot()
plt.show()



