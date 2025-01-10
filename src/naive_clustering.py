import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import math
import numpy as np

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

plt.scatter(center_lons, center_lats, color = colors, marker = 'x')
path = "../Data/Copernicus/sea_level_anomaly_data.nc"
data = xr.open_dataset(path)

all_lats = data['latitude'].values
all_lons = data['longitude'].values

#print(data['sla'].loc[:,all_lats[100], all_lons[100]].values)
#get nearest entities for hardcoded centers

centers = []
for i in range(len(center_lats)):

    centers.append(data['sla'].sel(latitude = center_lats[i], longitude = center_lons[i], method = 'nearest'))
    center_lons[i] = float(centers[i]['longitude'])
    center_lats[i] = float(centers[i]['latitude'])
 

assignment = [ [0]*len(all_lats) for i  in range(len(all_lons))]

time_series = data['sla'].loc[:,all_lats[0], all_lons[0]].values

print(distance_function(all_lats[0],all_lons[0],time_series,center_lats[0],center_lons[0],centers[0].values))
#for i,lat in enumerate(all_lats):
#    for j,lon in enumerate(all_lons):
#        time_series = data['sla'].loc[:,lat, lon].values
#        min_dist = math.inf
#        for k in range(len(centers)):
#            c = centers[k].values
#            tmp = distance_function(lat,lon,time_series,)

       #print(distance_function(lat,lon))
#       print("test")



#print(centers[0])
#print(centers[0].values)


