import matplotlib
import numpy
from pylab import *
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from pandas import *

filepath1 = '/Users/georgeberry/Documents/info6850/observation/results/monthly_redone_full_1.csv'
filepath2 = '/Users/georgeberry/Documents/info6850/observation/results/monthly_redone_full_2.csv'

#want users and locations to be our two index columns

## read in

df1 = read_csv(filepath1, index_col = [0,1]) #usecols = [0,1,2,3,4,5,7,8,9,10])

lat1 = []
long1 = []

for name, group in df1.groupby(level=0):
    long1.append(group.iloc[0]['user centroid long'])
    lat1.append(group.iloc[0]['user centroid lat'])


df2 = read_csv(filepath2, index_col = [0,1]) #usecols = [0,1,2,3,4,5,7,8,9,10])

lat2 = []
long2 = []

#by user
for name, group in df2.groupby(level=0):
    long2.append(group.iloc[0]['user centroid long'])
    lat2.append(group.iloc[0]['user centroid lat'])

##plot

fig = plt.figure(figsize=(24,18))

map = Basemap(projection='robin', lat_0=0, lon_0=-100,
              resolution='l', area_thresh=1000.0)

map.drawcoastlines()
map.drawcountries()
map.fillcontinents(color = 'gray')
map.drawmapboundary()
map.drawmeridians(np.arange(0, 360, 30))
map.drawparallels(np.arange(-90, 90, 30))

x,y = map(lat1, long1)
map.plot(x,y,'bo', markersize=4)

z,a = map(lat2, long2)
map.plot(z,a,'ro', markersize=4)

plt.show()

