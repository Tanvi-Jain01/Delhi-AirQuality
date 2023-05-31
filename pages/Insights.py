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
#import pygwalker as pyg

st.title("Geo Spatial Interpolation")

st.markdown("---")

import wget
# Load the NetCDF file into an xarray dataset
#ds = xr.open_dataset(r'C:\Users\Harshit Jain\Desktop\delhiaq\delhi_cpcb_2022.nc')
#print(type(ds))


dataset_url = "https://github.com/patel-zeel/delhi_aq/raw/main/data/delhi_cpcb_2022.nc"
dataset_file = wget.download(dataset_url)

# Read the NetCDF file
ds = xr.open_dataset(dataset_file)
#print(df)

df = ds.to_dataframe().reset_index()
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

gdf_shape = (r'C:\Users\Harshit Jain\Desktop\delhiaq\Delhi\Districts.shp')
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
st.sidebar.title("Linear Regression")

selected_date = st.sidebar.date_input('Select Date', value=pd.to_datetime('2022-08-23'))

#st.write(selected_date)

#-------------------------------------------------------------------------------------------
#if st.sidebar.button('Run Algorithm'):
lr_selected = st.sidebar.checkbox('Linear Regression')
rf_selected = st.sidebar.checkbox('Random Forest')
dt_selected = st.sidebar.checkbox('Decision Tree')
knn_selected = st.sidebar.checkbox('K-Nearest Neighbor')

selected_date = pd.to_datetime(selected_date)
#-------------------------------------------------------------------------------------------


# Extract the features and target variable for the selected date from the training data

X_train_date = X_train[X_train['Date'] == selected_date]
X_train_date = X_train_date.drop('PM2.5', axis=1)
X_train_date['Date'] = X_train_date['Date'].astype(np.int64)

y_train_date = X_train[X_train['Date'] == selected_date]['PM2.5'].values
#y_train_date = X_train['PM2.5']

# Define the parameter grid for grid search
param_grid = {'n_neighbors': [1,2,3,4,5,6,7,8, 9,10,11,12,13,14,15,16,17,18,19,20]}  # Specify the range of K values to try

# Create the KNN regressor
knn = KNeighborsRegressor()

# Perform grid search to find the best K value
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_date, y_train_date)

# Retrieve the best K value and the corresponding mean squared error
best_k = grid_search.best_params_['n_neighbors']
best_mse = -grid_search.best_score_

# Print the best K value and the corresponding mean squared error
#st.write('Best K:', best_k)
#st.write('Best Mean Squared Error:', best_mse)




#-------------------------------------------------------------------------------------------

def fit_model(x,model_name):
    X, y = x.iloc[:, 1:-1], x.iloc[:, -1]
    models = {
        'lr': LinearRegression().fit(X, y),
        'rf': RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=52).fit(X, y),
        'dt': DecisionTreeRegressor().fit(X, y),
        'knn': KNeighborsRegressor(n_neighbors=best_k, metric='euclidean', n_jobs=-1).fit(X, y)
    }
    return models

model_listlr = []
model_listrf = []
model_listdt = []
model_listknn = []
train_time_df = X_train[X_train['Date'] == selected_date].groupby('Date')

for model_name in ['lr', 'rf', 'dt', 'knn']:
    models = train_time_df.apply(fit_model, model_name=model_name)
    if model_name == 'lr':
        model_listlr.extend(models)
    elif model_name == 'rf':
        model_listrf.extend(models)
    elif model_name == 'dt':
        model_listdt.extend(models)
    elif model_name == 'knn':
        model_listknn.extend(models)

# -------------------------------------------------------------------------------------------

### TRAINING RMSE

st.subheader("Training")
st.markdown("---")

rmse_values = []

predn_listlr = []
predn_listrf = []
predn_listdt = []
predn_listknn = []

fig, ax = plt.subplots(figsize=(8, 6))
#fig.set_facecolor('black')
ax.set_facecolor('black')
train_time_df_i=pd.DataFrame()


for i, j in enumerate(train_time_df.groups.keys()):
    X_train_i = train_time_df.get_group(j)
    y_train_i = X_train_i.iloc[:, -1]
    train_time_df_i['true_y']=y_train_i
    train_time_df_i.reset_index(drop=True, inplace=True)
     
    ax.plot(train_time_df_i.index, train_time_df_i['true_y'], label='True Y')

    if lr_selected:
        y_train_pred_lr = model_listlr[i]['lr'].predict(X_train_i.iloc[:, 1:-1])
        predn_listlr.append(y_train_pred_lr)


        train_time_df_i['pred_lr']=np.concatenate(predn_listlr)

        rmse_lr = mean_squared_error(y_train_i, y_train_pred_lr, squared=False)
        ax.plot(train_time_df_i.index, train_time_df_i['pred_lr'], label='Linear Regression')
        st.write('Training RMSE Linear Regression:', rmse_lr)


    if rf_selected:

        y_train_pred_rf = model_listrf[i]['rf'].predict(X_train_i.iloc[:, 1:-1])
        predn_listrf.append(y_train_pred_rf)

        train_time_df_i['pred_rf']=np.concatenate(predn_listrf)
    
        rmse_rf = mean_squared_error(y_train_i, y_train_pred_rf, squared=False)
        ax.plot(train_time_df_i.index, train_time_df_i['pred_rf'], label='Random Forest')
        st.write('Training RMSE Random Forest:', rmse_rf)


    
    if dt_selected:

        y_train_pred_dt = model_listdt[i]['dt'].predict(X_train_i.iloc[:, 1:-1])
        predn_listdt.append(y_train_pred_dt)


        train_time_df_i['pred_dt']=np.concatenate(predn_listdt)
        rmse_dt = mean_squared_error(y_train_i, y_train_pred_dt, squared=False)

        ax.plot(train_time_df_i.index, train_time_df_i['pred_dt'], label='Decision Tree')
        st.write('Training RMSE Decision Tree:', rmse_dt)

    
    if knn_selected:

        y_train_pred_knn = model_listknn[i]['knn'].predict(X_train_i.iloc[:, 1:-1])
        predn_listknn.append(y_train_pred_knn)

        train_time_df_i['pred_knn']=np.concatenate(predn_listknn)

        rmse_knn = mean_squared_error(y_train_i, y_train_pred_knn, squared=False)
        ax.plot(train_time_df_i.index, train_time_df_i['pred_knn'], label='K-Nearest Neighbor')
        st.write('Training RMSE K-Nearest Neighbor:', rmse_knn,'Best K:', best_k)



st.write(train_time_df_i.T)



# -------------------------------------------------------------------------------------------


# Set the axis labels and title
ax.set_xlabel('Station', color='white')
ax.set_ylabel('Value', color='white')
ax.set_title('True Y vs Pred Y', color='white')
ax.legend()

# Display the plot
st.pyplot(fig)


# -------------------------------------------------------------------------------------------



st.subheader("Testing")
st.markdown("---")

rmse_test_dt = []
rmse_test_rf = []
rmse_test_lr = []
rmse_test_knn = []


predn_list_lr = []
predn_list_knn = []
predn_list_dt = []
predn_list_rf = []



test_time_=pd.DataFrame()
test_time_ = X_test[X_test['Date'] == selected_date].reset_index(drop=True, inplace=True)
     
#st.write(test_time_['PM2.5'])

test_time_df = X_test[X_test['Date'] == selected_date].groupby('Date')
#st.write(test_time_df)




fig, ax = plt.subplots(figsize=(8, 6))
ax.set_facecolor('black')

ax.plot(test_time_.index, test_time_['PM2.5'], label='True Y')

# Predict using the trained model for each time step
for i, j in enumerate(test_time_df.groups.keys()):
    group_a = test_time_df.get_group(j)


    if lr_selected:
        predns = model_listlr[i]['lr'].predict(group_a.iloc[:, 1:-1])
        predn_list_lr.append(predns)

        test_time_['pred_lr'] =  np.concatenate(predn_list_lr)
        
        rmse = mean_squared_error(test_time_['PM2.5'], np.concatenate(predn_list_lr)) ** 0.5
        st.write('Testing RMSE Linear Regression', rmse)

        ax.plot(test_time_.index, test_time_['pred_lr'], label='Linear Regression')



    if rf_selected:
        predns = model_listrf[i]['rf'].predict(group_a.iloc[:, 1:-1])
        predn_list_rf.append(predns)

        test_time_['pred_rf'] =  np.concatenate(predn_list_rf)

        rmse = mean_squared_error(test_time_['PM2.5'], np.concatenate(predn_list_rf)) ** 0.5
        st.write('Testing RMSE Random Forest', rmse)

        ax.plot(test_time_.index, test_time_['pred_rf'], label='Random Forest')


    if dt_selected:
        predns = model_listdt[i]['dt'].predict(group_a.iloc[:, 1:-1])
        predn_list_dt.append(predns)

        test_time_['pred_dt'] =  np.concatenate(predn_list_dt)

        rmse = mean_squared_error(test_time_['PM2.5'], np.concatenate(predn_list_dt)) ** 0.5
        st.write('Testing RMSE Decision Tree', rmse)

        ax.plot(test_time_.index, test_time_['pred_dt'], label='Decision Tree')


    if knn_selected:
        predns = model_listknn[i]['knn'].predict(group_a.iloc[:, 1:-1])
        predn_list_knn.append(predns)

        test_time_['pred_knn'] =  np.concatenate(predn_list_knn)
        
        rmse = mean_squared_error(test_time_['PM2.5'], np.concatenate(predn_list_knn)) ** 0.5
        st.write('Testing RMSE KNN', rmse)

        ax.plot(test_time_.index, test_time_['pred_knn'], label='K Nearest Neighbor')


#test_time_['pred_lr']=predn_list_lr

# -------------------------------------------------------------------------------------------
test_time_ = test_time_.drop(['Date', 'latitude', 'longitude'], axis=1)
st.write(test_time_.T)


ax.set_xlabel('Station', color='white')
ax.set_ylabel('Value', color='white')
ax.set_title('True Y vs Pred Y', color='white')
ax.legend()
st.pyplot(fig)









# -------------------------------------------------------------------------------------------


"""
pyg.walk(train_time_df_i, env='Streamlit')
"""




































































































































































































































































































































































