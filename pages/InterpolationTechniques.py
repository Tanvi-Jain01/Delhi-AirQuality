import xarray as xr
import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from shapely.geometry import Point
import pandas as pd
import numpy as np
#!pip install cupy
import matplotlib.pyplot as plt
#from cuml.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
import streamlit as st
from metpy.cbook import get_test_data
from metpy.interpolate import (interpolate_to_grid, remove_nan_observations,
                               remove_repeat_coordinates)
from metpy.plots import add_metpy_logo
from scipy.interpolate import LinearNDInterpolator,NearestNDInterpolator

st.title("Geo Spatial Interpolation")

st.markdown("---")
st.title("Linear Interpolation")
st.markdown("---")

st.write("Linear interpolation is a commonly used technique in geospatial interpolation to estimate missing or unobserved data points within a spatial dataset. Linear interpolation assumes a linear relationship between neighboring observed data points and uses this relationship to predict values at unsampled locations.")
st.write("In geospatial interpolation, linear interpolation utilizes the spatial coordinates (longitude and latitude) and the observed values of the target variable to estimate values at unsampled locations along a straight line between neighboring points. The method calculates the weightage or proportion assigned to each neighboring point based on their distance to the target location. These weightages are used to determine the interpolated value, where closer neighbors have higher influence. ")
st.write("Linear interpolation assumes a smooth and gradual change in the variable across space, resulting in a continuous surface representation. While linear interpolation is relatively simple and computationally efficient, it may not capture complex spatial patterns or abrupt changes in the target variable as effectively as other interpolation methods. Nevertheless, linear interpolation provides a basic approach for geospatial interpolation, serving as a useful baseline for comparison with more advanced techniques.")

st.title("Nearest Interpolation")
st.markdown("---")
st.write("NearestNDInterpolator is a widely used technique in geospatial interpolation. This method assigns the value of the nearest observed data point to the unsampled location, effectively applying the 'nearest neighbor' concept.")
st.write("In geospatial interpolation, NearestNDInterpolator utilizes the spatial coordinates (longitude and latitude) of both observed and unsampled locations to identify the closest observed point. The method assigns the value of this nearest observed point to the unsampled location, resulting in a discrete representation of the target variable.")
st.write("NearestNDInterpolator is particularly useful when spatial patterns exhibit abrupt changes or discontinuities, as it preserves the original values without interpolation between neighboring points. However, it may not capture fine-grained variations in the target variable as effectively as other interpolation methods. Nevertheless, NearestNDInterpolator provides a straightforward and computationally efficient approach for geospatial interpolation, particularly when preserving the exact values of the observed data points is critical.")
st.markdown("---")


import wget

# Download dataset file from GitHub
dataset_url = "https://github.com/patel-zeel/delhi_aq/raw/main/data/delhi_cpcb_2022.nc"
dataset_file = wget.download(dataset_url)

# Read the NetCDF file
ds = xr.open_dataset(dataset_file)

# Load the NetCDF file into an xarray dataset
#ds = xr.open_dataset(r'C:\Users\Harshit Jain\Desktop\delhiaq\delhi_cpcb_2022.nc')
#print(type(ds))

df = ds.to_dataframe().reset_index()
#print(df)
#-----------------------------------------------------------------------

#PREPROCESSING
df['Date'] = pd.to_datetime(df['time'])
#print(df['Date'].dtype)

df['Date'] = df['Date'].dt.date
#print(df['Date'])

df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
#print(df['Date'])

df = df[df['Date'].dt.year != 2023]
#df.shape, df


#print(df['PM2.5'].isna().sum())
dropped=df=df.dropna(subset=["PM2.5"])
#df.shape

#-------------------------------------------------------------------------

#Extracting necessary columns
daily_mean=pd.DataFrame()

#daily_mean= df.groupby(['station', 'Date','latitude','longitude'])['PM2.5'].mean().reset_index()

daily_mean = df.groupby(['station', 'Date', 'latitude', 'longitude']).mean().reset_index()
#print(daily_mean.columns)

#daily_mean=daily_mean[['station', 'Date', 'latitude', 'longitude','WS','WD','AT','RF','TOT-RF','PM2.5']]
type(daily_mean)
#print(daily_mean)

df=daily_mean


#------------------------------------------------------------------------------


#Checking groups
df['date_index']=df['Date']
#print(df.columns)
df.set_index('date_index')



#----------------------------------------------------------------------------------

unique=df[['station','latitude','longitude']].drop_duplicates()
#print(unique)
#print(len(unique))
#type(unique)

lat = unique['latitude']
lon = unique['longitude']

geometry = [Point(x, y) for x, y in zip(lon, lat)]
stationgeo=gpd.GeoDataFrame(unique,geometry=geometry)
#print(stationgeo)
#type(stationgeo)


#-------------------------------------------------------------------------------------------------------

#gdf_shape = (r'C:\Users\Harshit Jain\Desktop\delhiaq\Delhi\Districts.shp')
#gdf_shape = gpd.read_file(gdf_shape)

gdf_shape = (r'Districts.shp')
gdf_shape = gpd.read_file(gdf_shape)
#--------------------------------------------------------------------------------------------------------------

gdf_data = gpd.GeoDataFrame(unique, geometry=geometry)

# Set the CRS of the GeoDataFrame to match the shapefile
gdf_data.crs = gdf_shape.crs   #try directly stationgeo here


#-------------------------------------------------------------------------------------------
df = df.fillna(method='ffill')
df = df.fillna(method='bfill')


#-------------------------------------------------------------------------------------------

all_stations = df.station.unique()
train_station, test_station = train_test_split(all_stations, test_size=0.2, random_state=42)
X_train = df[df.station.isin(train_station)]
X_test = df[df.station.isin(test_station)]

#X_train.station.unique().shape, X_test.shape, X_test.columns    


#----------------------------------------------------------------------------------------------------
#df.set_index("time_")
X_train = X_train[['Date','latitude','longitude','PM2.5']]
X_test = X_test[['Date','latitude','longitude','PM2.5']]

#----------------------------------------------------------------------------------------------




#from scipy.interpolate import interpolate_to_grid
#delhi_shapefile = gdf_shape = (r'C:\Users\Harshit Jain\Desktop\delhiaq\Delhi\Districts.shp')
#delhi_shapefile = gpd.read_file(delhi_shapefile)
gdf_shape = (r'Districts.shp') 
delhi_shapefile= gpd.read_file(gdf_shape)   

#--------------------------------------------------------------------------------------------------





st.sidebar.title("Interpolation Techniques")

import streamlit as st
from metpy.cbook import get_test_data
from metpy.interpolate import interpolate_to_grid,natural_neighbor_to_grid
from metpy.plots import add_metpy_logo


#--------------------------------------------------------------------------------------------------

# Create a selectbox for interpolation type

selected_date = st.sidebar.date_input('Select Date', value=pd.to_datetime('2022-08-23'))

#st.write(selected_date)

selected_date = pd.to_datetime(selected_date)


interp_type = st.sidebar.selectbox('Interpolation Type', ['linear', 'nearest', 'rbf', 'natural_neighbor', 'barnes', 'cressman'])
st.write(interp_type)
# Create a slider for horizontal resolution
hres = st.sidebar.slider('Horizontal Resolution', min_value=10000, max_value=100000, step=10000, value=50000)

# Create a slider for minimum neighbors
minimum_neighbors = st.sidebar.slider('Minimum Neighbors', min_value=1, max_value=39, value=3)

# Create a slider for gamma (applicable for barnes interpolation)
gamma = st.sidebar.slider('Gamma', min_value=0.0, max_value=1.0, value=0.25)

# Create a slider for kappa_star (applicable for barnes interpolation)
kappa_star = st.sidebar.slider('Kappa Star', min_value=0.0, max_value=10.0, value=5.052)

# Create a slider for search radius
search_radius = st.sidebar.slider('Search Radius', min_value=0.0, max_value=10000.0, value=100.0)

# Create a selectbox for rbf function
rbf_func = st.sidebar.selectbox('Rbf Function', ['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'])

# Create a slider for rbf smooth
rbf_smooth = st.sidebar.slider('Rbf Smooth', min_value=0.0, max_value=1.0, value=0.0)

lon = st.sidebar.slider('Longitude', min_value=10, max_value=500, value=50, step=5)
lat = st.sidebar.slider('Latitude', min_value=10, max_value=500, value=50, step=5)


#--------------------------------------------------------------------------------------------------


train = df[['Date','latitude','longitude','PM2.5']]
train_filtered = train[train['Date'] == selected_date]


# Convert train_filtered coordinates and variable to numpy arrays
xp = train_filtered['longitude'].to_numpy()
yp = train_filtered['latitude'].to_numpy()
variable = train_filtered['PM2.5'].to_numpy()

print('shfzdtdthzedtjseztjs',xp,yp,variable)


# Generate the grid of points
x = np.linspace(stationgeo['longitude'].min()-0.5, stationgeo['longitude'].max()+0.5, num=300)
y = np.linspace(stationgeo['latitude'].min()-0.5 , stationgeo['latitude'].max()+0.5, num=300)

# Create the grid coordinates using meshgrid
gx, gy = np.meshgrid(x, y)


if (interp_type=='linear'):
    interp=LinearNDInterpolator(list(zip(xp,yp)),variable)
    img=interp(gx,gy)


elif interp_type=='nearest':
    interp=NearestNDInterpolator(list(zip(xp,yp)),variable)
    img=interp(gx,gy)

elif interp_type=='natural_neighbor':
    img = natural_neighbor_to_grid(xp, yp, variable, gx, gy)

else:
# Call the interpolate_to_grid function with the provided parameters
    gx,gy,img = interpolate_to_grid(xp,yp,variable, interp_type=interp_type, hres=hres, minimum_neighbors=minimum_neighbors,
                             gamma=gamma, kappa_star=kappa_star, search_radius=search_radius,
                             rbf_func=rbf_func, rbf_smooth=rbf_smooth)

# Print the result
#st.write('Interpolated Grid:', img)





#--------------------------------------------------------------------------------------------------
print('gggggggggggggggggggggggggg',gx,gy,img)
print('image',img)
# Create a masked array to handle NaN values
#img_2d = img.reshape(gx.shape)
#img = np.ma.masked_where(np.isnan(img), img)


print('image',img)


# Create the figure and plot
fig, ax = plt.subplots(figsize=(7, 7))
shapefile_extent = delhi_shapefile.total_bounds

# Set the plot extent to match the shapefile
plt.xlim(shapefile_extent[0], shapefile_extent[2])
plt.ylim(shapefile_extent[1], shapefile_extent[3])

# Plot the interpolated values as a contour plot
contour = ax.contourf(gx, gy, img, cmap='coolwarm', levels=50)

# Add the shapefile to the plot
delhi_shapefile.plot(ax=ax, edgecolor='black', facecolor='none')
gdf_data.plot(ax=ax, color='black', markersize=20, label='Air Stations')
# Add a colorbar
#plt.colorbar(contour,ax=ax)
ax.legend()
plt.colorbar(contour, label='PM2.5',shrink=0.7,format='%.3f')

# Customize the plot appearance
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geospatial Interpolation')

# Show the plot
plt.show()
st.pyplot(fig)


 #-------------------------------------------------------------------------------------------



































































