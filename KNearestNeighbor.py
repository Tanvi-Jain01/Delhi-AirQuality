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

#st.markdown("---")
         
st.write("Geospatial interpolation is a technique used to estimate and fill in missing or unobserved data points in a spatial dataset, specifically in the context of the Delhi dataset. The Delhi dataset consists of georeferenced measurements of air quality variables, such as PM2.5, collected from various monitoring stations across the Delhi region. Interpolation methods are employed to predict the air quality values at unsampled locations based on the observed data from neighboring stations. These methods utilize the spatial coordinates (longitude and latitude)including Date and performing Spatial Interpolation of the air quality measurements at the observed stations to create a continuous surface representation of the variable of interest.")

st.subheader("K-Nearest Neighbor")
st.write("This algorithm calculates the distances between the unsampled location and the available data points, and then selects the k nearest neighbors. The values of these neighbors are used to estimate the missing value by taking into account their distances and potentially applying weights based on their proximity.")
st.write("kNN interpolation is simple to implement and can handle irregularly distributed data points. However, it is sensitive to the choice of k, the number of neighbors, and the distance metric used. Additionally, kNN interpolation may not capture complex spatial patterns or account for spatial dependencies. Nevertheless, kNN-based geospatial interpolation offers a flexible and intuitive approach for filling in missing values and generating continuous spatial representations of the target variable.")

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


#------------------------------------------------------------------------------
#df.set_index("time_")
X_train = X_train[['Date','latitude','longitude','PM2.5']]
X_test = X_test[['Date','latitude','longitude','PM2.5']]

#---------------------------------------------------------------------------


st.sidebar.title("K-Nearest Neighbour")

selected_date = st.sidebar.date_input('Select Date', value=pd.to_datetime('2022-08-23'))

# Slider for k value
k = st.sidebar.slider('Choose K for Training & Testing', min_value=1, max_value=31, value=14)

# Selection for weights
weights = st.sidebar.selectbox('Weights', ['uniform', 'distance'])

# Selection for distance metric
distance_metric = st.sidebar.selectbox('Distance Metric', ['euclidean', 'manhattan', 'minkowski'])

# Selection for algorithm   
algorithm = st.sidebar.selectbox('Algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])


#st.write(selected_date)

#-------------------------------------------------------------------------------------------
#if st.sidebar.button('Run Algorithm'):
# Initialize lists to store RMSE and k values
    
selected_date = pd.to_datetime(selected_date)
def fit_model(x):
    X, y = x.iloc[:,1:-1], x.iloc[:,-1]
    model = gs.fit(X, y)
    return model



# Loop over the k values
#for k in k:
gs = KNeighborsRegressor(n_neighbors=k, algorithm=algorithm, weights=weights, metric=distance_metric, n_jobs=-1)
model_list = []
train_time_df = X_train[X_train['Date'] ==selected_date].groupby('Date')
#train_time_df = X_train.groupby('Date')

# Train the model for each time ste
model = train_time_df.apply(fit_model)
model_list.extend(model)


#-------------------------------------------------------------------------------------------

###TRAINING RMSE
st.markdown("---")
st.subheader("Training")
st.markdown("---")

rmse_values = []
predn_list = []

for i,j in enumerate(train_time_df.groups.keys()):
    X_train_i = train_time_df.get_group(j)
    y_train_i = X_train_i.iloc[:, -1]
    y_train_pred_i = model_list[i].predict(X_train_i.iloc[:, 1:-1])
    predn_list.append(y_train_pred_i)


rmse_i = mean_squared_error(y_train_i, y_train_pred_i, squared=False)   
print(np.concatenate(predn_list))
rmse_values.append(rmse_i)

st.write('Training RMSE',rmse_values)

train_time_df=pd.DataFrame()
train_time_df['true_y']=y_train_i
train_time_df['pred_y']=y_train_pred_i
train_time_df.reset_index(drop=True,inplace=True)
st.write(train_time_df.T)

#k_list.append(k)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(train_time_df.index, train_time_df['true_y'], label='True Y')
ax.plot(train_time_df.index, train_time_df['pred_y'], label='Pred Y')
ax.set_xlabel('Station')
ax.set_ylabel('Value')
ax.set_title('True Y vs Pred Y')
ax.legend()

# Display the plot
st.pyplot(fig)



# fig, ax = plt.subplots(figsize=(7, 7))
# scatter=ax.scatter(X_train_i['longitude'], X_train_i['latitude'], c= train_time_df['true_y'], cmap='coolwarm', alpha=0.8,s=50)
# cbar = plt.colorbar(scatter, ax=ax)

# # Set labels and title
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
# ax.set_title('Bubble Plot of PM2.5')

# # Display the plot
# st.pyplot(fig)

#-------------------------------------------------------------------------------------------



###TESTING RMSE
st.markdown("---")
st.subheader("Testing")
st.markdown("---")

rmse_values = []
predn_list = []

#test_time_df = X_test.groupby(selected_date)
test_time_df = X_test[X_test['Date'] == selected_date].groupby('Date')
#st.write(test_time_df)

#test_time_df = X_test.groupby('Date')
    # Predict using the  trained model for each time step

for i, j in enumerate(test_time_df.groups.keys()):
        group_a = test_time_df.get_group(j)
        #st.write(j)
        predns = model_list[i].predict(group_a.iloc[:, 1:-1])
        predn_list.append(predns)

# Calculate RMSE
predict = pd.concat([pd.DataFrame(predn) for predn in predn_list], ignore_index=True)
test_time_df = predict.rename(columns={0: "pred_y"})
#test_time_df["true_y"] = X_test["PM2.5"].reset_index(drop=True)
test_time_df["true_y"] = X_test[X_test['Date'] == selected_date]["PM2.5"].values


st.write(test_time_df.T)
# Compute RMSE for each time step

rmse = mean_squared_error(test_time_df["true_y"], np.concatenate(predn_list)) ** 0.5

# Store RMSE and k values
rmse_values.append(rmse)  
 #k_list.append(k)
st.write('Testing RMSE',rmse_values)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(test_time_df.index, test_time_df["true_y"], label='True Y')
ax.plot(test_time_df.index, test_time_df["pred_y"], label='Pred Y')
ax.legend()
ax.set_xlabel('Station')
ax.set_ylabel('PM2.5')
ax.set_title('True Y vs Pred Y')

st.pyplot(fig)
#print(np.concatenate(predn_list))
#     fig, ax = plt.subplots(figsize=(5, 5))

#     #size = test_time_df["true_y"] * 10
#     ax.scatter(group_a['longitude'], group_a['latitude'], c=test_time_df["true_y"],cmap='coolwarm', alpha=0.5,s=50)
#     cbar = plt.colorbar(scatter, ax=ax)
# # Set labels and title
#     ax.set_xlabel('Longitude')   
#     ax.set_ylabel('Latitude')
#     ax.set_title('Bubble Plot of PM2.5')

# # Display the plot
#     st.pyplot(fig)




#-------------------------------------------------------------------------------------------
flat_list = np.array(predn_list).flatten()
#st.write(flat_list)


#-------------------------------------------------------------------------------------------
train = df[['Date','latitude','longitude','PM2.5']]

st.markdown("---")

k_final = st.sidebar.slider('Choose K for Interpolation', min_value=1, max_value=39, value=14)
lon = st.sidebar.slider('Longitude', min_value=10, max_value=50, value=25, step=5)
lat = st.sidebar.slider('Latitude', min_value=10, max_value=50, value=25, step=5)

 # Load the shapefile
#delhi_shapefile = gpd.read_file(r'C:\Users\Harshit Jain\Desktop\delhiaq\Delhi\Districts.shp')
gdf_shape = (r'Districts.shp') 
delhi_shapefile= gpd.read_file(gdf_shape)   

 # Generate the grid of points
x = np.linspace(stationgeo['longitude'].min() - 0.5, stationgeo['longitude'].max() + 0.5, num=lon)
y = np.linspace(stationgeo['latitude'].min() - 0.5, stationgeo['latitude'].max() + 0.5, num=lat)

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

gs = KNeighborsRegressor(n_neighbors=k_final, algorithm=algorithm, weights=weights, metric=distance_metric, n_jobs=-1)
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
st.subheader("Interpolation")
st.markdown("---")
st.pyplot(fig)

 #-------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("Predictions")
st.markdown("---")

st.dataframe(testing)





 #-------------------------------------------------------------------------------------------
# st.markdown("---")
# st.subheader("Dynamic Presentation")
# st.markdown("---")
import plotly.express as px

fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", hover_name="station",
                        zoom=9, height=500,color='PM2.5')
fig.update_layout(mapbox_style="open-street-map")
 #fig.show()
#st.plotly_chart(fig)



#AERMOD
#DISPERSION  
#RECEPTOR MODEL
