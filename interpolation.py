import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from scipy.interpolate import NearestNDInterpolator
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.tree import DecisionTreeRegressor
# import netcdf4 
#import dask.distributed
import dask.array as da
from shapely.geometry import Point


# URL of the NetCDF file on GitHub
#nc_url = 'https://github.com/patel-zeel/delhi_aq/raw/main/data/delhi_cpcb_2022.nc'


import wget

# Download dataset file from GitHub
dataset_url = "https://github.com/patel-zeel/delhi_aq/raw/main/data/delhi_cpcb_2022.nc"
dataset_file = wget.download(dataset_url)


# Read the NetCDF file
ds = xr.open_dataset(dataset_file)

# Display the dataset




# Create a DataFrame
df = ds.to_dataframe().reset_index()
#print(df)

df['Date'] = pd.to_datetime(df['time'])
#print(df['Date'].dtype)

df['Date'] = df['Date'].dt.date
#print(df['Date'])
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
#print(df['Date'])


daily_mean=pd.DataFrame()

#daily_mean= df.groupby(['station', 'Date','latitude','longitude'])['PM2.5'].mean().reset_index()

daily_mean = df.groupby(['station', 'Date', 'latitude', 'longitude']).mean().reset_index()
print(daily_mean.columns)

daily_mean=daily_mean[['station', 'Date', 'latitude', 'longitude','WS','WD','AT','RF','TOT-RF','PM2.5']]
type(daily_mean)
#print(daily_mean)


df = df[df['Date'].dt.year != 2023]
df.shape, df

dropped=df=df.dropna(subset=["PM2.5"])

df['date_index']=df['Date']
print(df.columns)
df.set_index('date_index')


daily_mean=pd.DataFrame()

#daily_mean= df.groupby(['station', 'Date','latitude','longitude'])['PM2.5'].mean().reset_index()

daily_mean = df.groupby(['station', 'Date', 'latitude', 'longitude']).mean().reset_index()
print(daily_mean.columns)

daily_mean=daily_mean[['station', 'Date', 'latitude', 'longitude','WS','WD','AT','RF','TOT-RF','PM2.5']]
type(daily_mean)
#print(daily_mean)

df=daily_mean

unique=df[['station','latitude','longitude']].drop_duplicates()
#print(unique)
#print(len(unique))
#type(unique)

import geopandas as gpd
import matplotlib.pyplot as plt

lat = df['latitude'].drop_duplicates()
lon = df['longitude'].drop_duplicates()


geometry = [Point(x, y) for x, y in zip(lon, lat)]
stationgeo=gpd.GeoDataFrame(unique,geometry=geometry)
#print(stationgeo)
#type(stationgeo)


fig, ax = plt.subplots(figsize=(8, 8))

# Plot the GeoDataFrame
stationgeo.plot(ax=ax,color='red')

# Show the plot
plt.show()


import geopandas as gpd
import matplotlib.pyplot as plt

# Read the shapefile
shapefile_path = 'https://github.com/Tanvi-Jain01/Delhi-AirQuality/blob/main/Districts.shp'
gdf_shape = gpd.read_file(shapefile_path)

# Convert DataFrame to GeoDataFrame
#gdf_data = gpd.GeoDataFrame(unique, geometry=gpd.points_from_xy(unique.longitude, unique.latitude))

gdf_data = gpd.GeoDataFrame(unique, geometry=geometry)

# Set the CRS of the GeoDataFrame to match the shapefile
gdf_data.crs = gdf_shape.crs

# Plot the shapefile
fig, ax = plt.subplots(figsize=(10, 10))
gdf_shape.boundary.plot(ax=ax,edgecolor='black')

# Plot the data points on top of the shapefile
gdf_data.plot(ax=ax, color='red', markersize=20, label='Air Stations')
ax.legend()
plt.title('Delhi Air Stations')
# Show the plot
plt.show()
st.pyplot(fig)


all_stations = df.station.unique()
train_station, test_station = train_test_split(all_stations, test_size=0.2, random_state=42)
X_train = df[df.station.isin(train_station)]
X_test = df[df.station.isin(test_station)]


df = df.fillna(method='ffill')
df = df.fillna(method='bfill')




X_train = X_train[['Date','latitude','longitude','WS', 'WD', 'AT', 'RF','TOT-RF','PM2.5']]
X_test = X_test[['Date','latitude','longitude','WS', 'WD', 'AT', 'RF','TOT-RF','PM2.5']]








df['Date'] = pd.to_datetime(df['time'])
print(df)
print(df.describe())



import plotly.express as px
fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", hover_name=df["station"],
                        zoom=8, height=500, color_discrete_sequence=['blue'])
fig.update_layout(mapbox_style="open-street-map", title="Delhi Air Stations")
#fig.show()
st.plotly_chart(fig)





from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

# Function to fit the model
def fit_model(x):
    X, y = x.iloc[:, 1:-1], x.iloc[:, -1]
    model = gs.fit(X, y)
    return model

# Add Streamlit sidebar widgets for parameter selection
selected_date = st.sidebar.date_input("Select Date")
k = st.sidebar.slider('k', min_value=1, max_value=15, value=5)
distance_metric = st.sidebar.selectbox('Distance Metric', ['euclidean', 'manhattan'])
algorithm = st.sidebar.radio('Algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])

# Create a button to run the algorithm
run_button = st.sidebar.button("Run Algorithm")

if run_button:
    # Train the model with selected parameters
    gs = KNeighborsRegressor(n_neighbors=k, metric=distance_metric, algorithm=algorithm)
    
    model_list = []
    train_time_df = X_train.groupby("datetime")
    model = train_time_df.apply(fit_model)
    model_list.extend(model)
    
    predn_list = []
    test_time_df = X_test.groupby("datetime")
    
    # Predict using the trained model for each time step
    for i, j in enumerate(test_time_df.groups.keys()):
        group_a = test_time_df.get_group(j)
        predns = model_list[i].predict(group_a.iloc[:, 1:-1])
        predn_list.append(predns)
    
    # Calculate RMSE
    predict = pd.concat([pd.DataFrame(predn) for predn in predn_list], ignore_index=True)
    test_time_df = predict.rename(columns={0: "pred_y"})
    test_time_df["true_y"] = X_test["PM2.5"].reset_index(drop=True)
    
    # Compute RMSE for each time step
    rmse = mean_squared_error(test_time_df["true_y"], np.concatenate(predn_list)) ** 0.5
    
    # Display RMSE value
    st.write(f"RMSE for k={k}: {rmse}")
    
    
    
    
    
   
  

