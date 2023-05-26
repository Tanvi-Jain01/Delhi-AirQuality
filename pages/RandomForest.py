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
from sklearn.linear_model import LinearRegression

st.title("Geo Spatial Interpolation")
st.markdown("---")


st.title("Random Forest")

st.write("RF is an ensemble learning method that combines multiple decision trees to make predictions. In the context of geospatial interpolation, RF utilizes the spatial coordinates (longitude and latitude) and the observed data from stations to predict the values at unsampled locations.")
st.write("The RF model is trained on the available data, with each tree learning different spatial patterns and relationships between the spatial coordinates and the target variable. The trained RF model can then be applied to interpolate the target variable at unsampled locations by considering the spatial characteristics captured by the ensemble of trees.")
st.write("RF interpolation offers several advantages, including the ability to handle complex non-linear relationships, capture spatial dependencies, and provide variable importance measures. It is particularly effective when there are complex spatial patterns and interactions among the variables. However, RF interpolation requires careful tuning of parameters and consideration of potential overfitting. Overall, RF-based geospatial interpolation provides a robust and flexible approach for estimating missing values and generating continuous spatial surfaces of the target variable.")
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


#------------------------------------------------------------------------------
#df.set_index("time_")
X_train = X_train[['Date','latitude','longitude','PM2.5']]
X_test = X_test[['Date','latitude','longitude','PM2.5']]

#---------------------------------------------------------------------------
st.sidebar.title("Random Forest")

selected_date = st.sidebar.date_input('Select Date', value=pd.to_datetime('2022-08-23'))


n_estimators = st.sidebar.slider('n_estimators', min_value=1, max_value=500, value=100, step=10)
criterion = st.sidebar.selectbox('criterion', ('squared_error', 'absolute_error', 'friedman_mse', 'poisson'), index=0)
max_depth = st.sidebar.selectbox('max_depth', (None, 5, 10, 15, 20,25,30), index=0)
max_leaf_nodes = st.sidebar.slider('max_leaf_nodes', min_value=2, max_value=20, value=None)


#st.write(selected_date)

#-------------------------------------------------------------------------------------------
#if st.sidebar.button('Run Algorithm'):
selected_date = pd.to_datetime(selected_date)

def fit_model(x):
        X, y = x.iloc[:, 1:-1], x.iloc[:, -1]
        model = gs.fit(X, y)
        return model

gs = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes,random_state=42)
model_list = []
train_time_df = X_train[X_train['Date'] == selected_date].groupby('Date')

# Train the model for each time step
model = train_time_df.apply(fit_model)
model_list.extend(model)

#-------------------------------------------------------------------------------------------

###TRAINING RMSE

st.subheader("Training")
st.markdown("---")

rmse_values = []
predn_list = []

for i, j in enumerate(train_time_df.groups.keys()):
    X_train_i = train_time_df.get_group(j)
    y_train_i = X_train_i.iloc[:, -1]
    y_train_pred_i = model_list[i].predict(X_train_i.iloc[:, 1:-1])
    predn_list.append(y_train_pred_i)

rmse_i = mean_squared_error(y_train_i, y_train_pred_i, squared=False)
rmse_values.append(rmse_i)

st.write('Training RMSE', rmse_values)

train_time_df = pd.DataFrame()
train_time_df['true_y'] = y_train_i
train_time_df['pred_y'] = y_train_pred_i
train_time_df.reset_index(drop=True, inplace=True)
st.write(train_time_df.T)


fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(train_time_df.index, train_time_df['true_y'], label='True Y')
ax.plot(train_time_df.index, train_time_df['pred_y'], label='Pred Y')
ax.set_xlabel('Stations')
ax.set_ylabel('Value')
ax.set_title('True Y vs Pred Y')
ax.legend()

# Display the plot
st.pyplot(fig)

# fig, ax = plt.subplots(figsize=(7, 7))
# scatter = ax.scatter(X_train_i['longitude'], X_train_i['latitude'], c=train_time_df['true_y'], cmap='coolwarm', alpha=0.8, s=50)
# cbar = plt.colorbar(scatter, ax=ax)

# # Set labels and title
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
# ax.set_title('Bubble Plot of PM2.5')

# # Display the plot
# st.pyplot(fig)

#-------------------------------------------------------------------------------------------

###TESTING RMSE
st.subheader("Testing")
st.markdown("---")

rmse_values = []
predn_list = []

test_time_df = X_test[X_test['Date'] == selected_date].groupby('Date')

# Predict using the trained model for each time step
for i, j in enumerate(test_time_df.groups.keys()):
    group_a = test_time_df.get_group(j)
    predns = model_list[i].predict(group_a.iloc[:, 1:-1])
    predn_list.append(predns)

# Calculate RMSE
predict = pd.concat([pd.DataFrame(predn) for predn in predn_list], ignore_index=True)
test_time_df = predict.rename(columns={0: "pred_y"})
test_time_df["true_y"] = X_test[X_test['Date'] == selected_date]["PM2.5"].values

#test_time_df=test_time_df["true_y","pred_y"]
st.write(test_time_df.T)

# Compute RMSE for each time step
rmse = mean_squared_error(test_time_df["true_y"], np.concatenate(predn_list)) ** 0.5

# Store RMSE and k values
rmse_values.append(rmse)
st.write('Testing RMSE', rmse_values)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(test_time_df.index, test_time_df["true_y"], label='True Y')
ax.plot(test_time_df.index, test_time_df["pred_y"], label='Pred Y')
ax.legend()
ax.set_xlabel('Index')
ax.set_ylabel('PM2.5')
ax.set_title('True Y vs Pred Y')
st.pyplot(fig)  

# fig, ax = plt.subplots(figsize=(5, 5))

# ax.scatter(group_a['longitude'], group_a['latitude'], c=test_time_df["true_y"], cmap='coolwarm', alpha=0.5, s=50)
# cbar = plt.colorbar(scatter, ax=ax)

# # Set labels and title
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
# ax.set_title('Bubble Plot of PM2.5')

# # Display the plot
# st.pyplot(fig)

flat_list = np.array(predn_list).flatten()

#-------------------------------------------------------------------------------------------
train = df[['Date', 'latitude', 'longitude', 'PM2.5']]


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

testing = pd.DataFrame()
testing['longitude'] = pd.DataFrame(lon_grid.flatten(), columns=['longitude'])
testing['latitude'] = pd.DataFrame(lat_grid.flatten(), columns=['latitude'])
testing.loc[:, 'Date'] = pd.to_datetime('2022-08-23')
testing.loc[:, 'PM2.5'] = 0

testing = testing[['Date', 'latitude', 'longitude', 'PM2.5']]

def fit_model(x):
    X, y = x.iloc[:, 1:-1], x.iloc[:, -1]
    model = gs.fit(X, y)
    return model


gs = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes,random_state=42)
model_list = []
train_time_df = train.groupby("Date")   

# Train the model for each time step
model = train_time_df.apply(fit_model)
model_list.extend(model)

testing.loc[:, 'Date'] = pd.to_datetime(selected_date)
testing.loc[:, 'PM2.5'] = 0
testing = testing[['Date', 'latitude', 'longitude', 'PM2.5']]

predn_list = []
test_time_df = testing.groupby("Date")

# Predict using the trained model for each time step
for i, j in enumerate(test_time_df.groups.keys()):
    group_a = test_time_df.get_group(j)
    predns = model_list[i].predict(group_a.iloc[:, 1:-1])
    predn_list.append(predns)

#st.write(predn_list)
flat_list = np.array(predn_list).flatten()
predictions = pd.DataFrame(list(zip(flat_list)), columns=['PM2.5'])
#st.write(predictions)
testing['PM2.5'] = predictions['PM2.5']

fig, ax = plt.subplots(figsize=(35, 10))
shapefile_extent = delhi_shapefile.total_bounds

# Set the plot extent to match the shapefile
plt.xlim(shapefile_extent[0], shapefile_extent[2])
plt.ylim(shapefile_extent[1], shapefile_extent[3])

# Plot the PM2.5 values as a contour plot
contour = ax.contourf(lon_grid, lat_grid, testing['PM2.5'].values.reshape(lon_grid.shape), cmap='coolwarm', levels=200)

# Add the shapefile to the plot
delhi_shapefile.plot(ax=ax, edgecolor='black', facecolor='none')

# Plot the grid points
# grid_points.plot(ax=ax, marker='o', color='grey', markersize=10, label='Grid Points')
gdf_data.plot(ax=ax, color='black', markersize=20, label='Delhi Air Stations')

# Add a colorbar
plt.colorbar(contour, label='PM2.5', shrink=0.7)

# Customize the plot appearance
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Spatial Interpolation')
plt.legend()

# Show the plot
# plt.show()
st.pyplot(fig)

st.markdown("---")
st.subheader("Predictions")
st.markdown("---")

st.dataframe(testing)































