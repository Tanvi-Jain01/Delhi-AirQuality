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

st.title("Geo Spatial Interpolation")

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
print(df['Date'].dtype)

df['Date'] = df['Date'].dt.date
#print(df['Date'])

df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
print(df['Date'])

df = df[df['Date'].dt.year != 2023]
#df.shape, df


print(df['PM2.5'].isna().sum())
dropped=df=df.dropna(subset=["PM2.5"])
#df.shape

#-------------------------------------------------------------------------

#Extracting necessary columns
daily_mean=pd.DataFrame()

#daily_mean= df.groupby(['station', 'Date','latitude','longitude'])['PM2.5'].mean().reset_index()

daily_mean = df.groupby(['station', 'Date', 'latitude', 'longitude']).mean().reset_index()
print(daily_mean.columns)

#daily_mean=daily_mean[['station', 'Date', 'latitude', 'longitude','WS','WD','AT','RF','TOT-RF','PM2.5']]
type(daily_mean)
print(daily_mean)

df=daily_mean


#------------------------------------------------------------------------------


#Checking groups
df['date_index']=df['Date']
print(df.columns)
df.set_index('date_index')


#----------------------------------------------------------------------------------

unique=df[['station','latitude','longitude']].drop_duplicates()
print(unique)
print(len(unique))
type(unique)

lat = unique['latitude']
lon = unique['longitude']

geometry = [Point(x, y) for x, y in zip(lon, lat)]
stationgeo=gpd.GeoDataFrame(unique,geometry=geometry)
print(stationgeo)
type(stationgeo)


#-------------------------------------------------------------------------------------------------------

#gdf_shape = (r'C:\Users\Harshit Jain\Desktop\delhiaq\Delhi\Districts.shp')
#gdf_shape = gpd.read_file(gdf_shape)
#shapefile_path = 'https://github.com/Tanvi-Jain01/Delhi-AirQuality/blob/main/Districts.shp'
#gdf_shape = gpd.read_file(shapefile_path)

import geopandas as gpd

#dataset_url = "https://github.com/Tanvi-Jain01/Delhi-AirQuality/blob/main/Districts.shp"
#gdf_shape = wget.download(dataset_url)
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


#-----------------------------------------------------------
#df.set_index("time_")
X_train = X_train[['Date','latitude','longitude','PM2.5']]
X_test = X_test[['Date','latitude','longitude','PM2.5']]
#-----------------------------------------------------------


st.sidebar.title("K-Nearest Neighbour")
selected_date = st.sidebar.date_input('Select Date', value=pd.to_datetime('2022-08-23'))

# Slider for k value
k = st.sidebar.slider('Choose K', min_value=1, max_value=39, value=14)

# Selection for weights
weights = st.sidebar.selectbox('Weights', ['uniform', 'distance'])

# Selection for distance metric
distance_metric = st.sidebar.selectbox('Distance Metric', ['euclidean', 'manhattan', 'minkowski'])

# Selection for algorithm   
algorithm = st.sidebar.selectbox('Algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])




if st.sidebar.button('Run Algorithm'):
# Initialize lists to store RMSE and k values
    
    selected_date = pd.to_datetime(selected_date)
    def fit_model(x):
        X, y = x.iloc[:,1:-1], x.iloc[:,-1]
        model = gs.fit(X, y)
        return model

    # Loop over the k values
    #for k in k:
    #gs = KNeighborsRegressor(n_neighbors=k, weights='uniform')
    gs = KNeighborsRegressor(n_neighbors=k, algorithm=algorithm, weights=weights, metric=distance_metric, n_jobs=-1)
    model_list = []
    #train_time_df = X_train.groupby(selected_date)
    train_time_df = X_train.groupby('Date')
    # Train the model for each time step
    model = train_time_df.apply(fit_model)
    model_list.extend(model)
    


    ###TESTING RMSE
    rmse_values = []
    predn_list = []
    #test_time_df = X_test.groupby(selected_date)
    test_time_df = X_test.groupby('Date')
        
        # Predict using the  trained model for each time step

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
        
        # Store RMSE and k values
    rmse_values.append(rmse)
    


    print(np.concatenate(predn_list))
    st.write('Testing RMSE',rmse_values)
 #-------------------------------------------------------------------------------------------
  #  flat_list = np.array(predn_list).flatten()

    train = df[['Date','latitude','longitude','PM2.5']]

 #-------------------------------------------------------------------------------------------


 # Load the shapefile
   # delhi_shapefile = gpd.read_file(r'C:\Users\Harshit Jain\Desktop\delhiaq\Delhi\Districts.shp')
    #delhi_shapefile = 'https://github.com/Tanvi-Jain01/Delhi-AirQuality/blob/main/Districts.shp'
    gdf_shape = (r'Districts.shp') 
    delhi_shapefile= gpd.read_file(gdf_shape)   


 # Generate the grid of points
    x = np.linspace(stationgeo['longitude'].min() - 0.5, stationgeo['longitude'].max() + 0.5, num=25)
    y = np.linspace(stationgeo['latitude'].min() - 0.5, stationgeo['latitude'].max() + 0.5, num=25)

    lon_grid, lat_grid = np.meshgrid(x, y)

 # Create a GeoDataFrame from the grid points
    grid_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon_grid.flatten(), lat_grid.flatten()))




    testing=pd.DataFrame()  
    testing['longitude'] = pd.DataFrame(lon_grid.flatten(), columns=['longitude'])
    testing['latitude']= pd.DataFrame(lat_grid.flatten(), columns=['latitude'])
    testing.loc[:, 'Date'] = pd.to_datetime('2022-08-23')
    testing.loc[:, 'PM2.5'] = 0

    testing=testing[['Date','latitude','longitude','PM2.5']]


 #print(testing)



    def fit_model(x):
        X, y = x.iloc[:,1:-1], x.iloc[:,-1]
        model = gs.fit(X, y)
        return model

# Loop over the k values

    gs = KNeighborsRegressor(n_neighbors=k, algorithm=algorithm, weights=weights, metric=distance_metric, n_jobs=-1)
    model_list = []
    train_time_df = train.groupby("Date")
    
 # Train the model for each time step
    model = train_time_df.apply(fit_model)
    model_list.extend(model)
    

    testing.loc[:, 'Date'] = pd.to_datetime(selected_date)

    testing.loc[:, 'PM2.5'] = 0
    testing=testing[['Date','latitude','longitude','PM2.5']]
    #st.dataframe(testing)
    predn_list = []
    test_time_df = testing.groupby("Date")
    
 # Predict using the trained model for each time step
    for i, j in enumerate(test_time_df.groups.keys()):
        group_a = test_time_df.get_group(j) 
        predns = model_list[i].predict(group_a.iloc[:, 1:-1])
        predn_list.append(predns)

    flat_list = np.array(predn_list).flatten()

    predictions = pd.DataFrame(list(zip(flat_list)),columns =['PM2.5'])


    testing['PM2.5']=predictions['PM2.5']




 #-----------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(35, 10))
    shapefile_extent = delhi_shapefile.total_bounds

 # Set the plot extent to match the shapefile
    plt.xlim(shapefile_extent[0], shapefile_extent[2])
    plt.ylim(shapefile_extent[1], shapefile_extent[3])




 # Plot the PM2.5 values as a contour plot
    contour = ax.contourf(lon_grid,lat_grid, testing['PM2.5'].values.reshape(lon_grid.shape), cmap='coolwarm', levels=200)

 # Add the shapefile to the plot
    delhi_shapefile.plot(ax=ax, edgecolor='black', facecolor='none')
    #gdf_shape.plot(ax=ax, edgecolor='black', facecolor='none')
 # Plot the grid points
 #grid_points.plot(ax=ax, marker='o', color='grey', markersize=10, label='Grid Points')
    gdf_data.plot(ax=ax, color='black', markersize=20, label='Delhi Air Stations')


 # Add a colorbar
    plt.colorbar(contour, label='PM2.5',shrink=0.7)

# Customize the plot appearance
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Spatial Interpolation')
    plt.legend()

 # Show the plot
 #plt.show()
    st.pyplot(fig)
 #-------------------------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Predictions")
    st.markdown("---")

    st.dataframe(testing)





 #-------------------------------------------------------------------------------------------
    st.markdown("---")
   # st.subheader("Dynamic Presentation")
    st.markdown("---")
    import plotly.express as px

    fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", hover_name="station",
                        zoom=9, height=500,color='PM2.5')
    fig.update_layout(mapbox_style="open-street-map")
 #fig.show()
    #st.plotly_chart(fig)


