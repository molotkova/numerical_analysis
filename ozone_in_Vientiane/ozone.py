from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import json

city = "Vientiane"
lat = 17.58
lon = 102.35

data = Dataset('MSR-2.nc')
lats = np.array(data.variables['latitude'][:])
lons = np.array(data.variables['longitude'][:])
time = np.array(data.variables['time'][:])

lat_ind = np.searchsorted(lats, lat)
lon_ind = np.searchsorted(lons, lon)

plt.figure()
plt.title('Average concentration of ozone in Vientiane')
x_all = np.arange(480)
x_jan = np.arange(0, time.size, 12)
x_jul = np.arange(6, time.size, 12)
y_all = np.array(data.variables['Average_O3_column'][x_all, lat_ind, lon_ind])
y_jan = np.array(data.variables['Average_O3_column'][x_jan, lat_ind, lon_ind])
y_jul = np.array(data.variables['Average_O3_column'][x_jul, lat_ind, lon_ind])
plt.plot(x_all, y_all, label = 'All period from 01/1979 to 01/2019')
plt.plot(x_jan, y_jan, label = 'January of each year')
plt.plot(x_jul, y_jul, label = 'July of each year')
plt.axis([0, 500, 220, 300])
plt.xlabel('time (months)')
plt.ylabel('concentration (Dobson units)')
plt.grid()
plt.legend(loc='upper left')
plt.show()
plt.savefig('ozon.png')

min_all = np.amin(y_all)
max_all = np.amax(y_all)
mean_all = round(np.mean(y_all), 1)

min_jan = np.amin(y_jan)
max_jan = np.amax(y_jan)
mean_jan = round(np.mean(y_jan), 1)

min_jul = np.amin(y_jul)
max_jul = np.amax(y_jul)
mean_jul = round(np.mean(y_jul), 1)

d = {
    'city': 'Vientiane',
    'coordinates': [lat, lon],
    'jan': {
        'min': float(min_jan),
        'max': float(max_jan),
        'mean': float(mean_jan)},
    'jul': {'min': float(min_jul),
            'max': float(max_jul),
            'mean': float(mean_jul)},
    'all': {'min': float(min_all),
            'max': float(max_all),
            'mean': float(mean_all)}
    }
with open('ozon.json', 'w') as f:
    json.dump(d, f)
with open('ozon.json', 'r') as f:
    print(f.read())
