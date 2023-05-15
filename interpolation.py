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



# URL of the NetCDF file on GitHub
#nc_url = 'https://github.com/patel-zeel/delhi_aq/raw/main/data/delhi_cpcb_2022.nc'


import wget

# Download dataset file from GitHub
dataset_url = "https://github.com/patel-zeel/delhi_aq/raw/main/data/delhi_cpcb_2022.nc"
dataset_file = wget.download(dataset_url)


# Read the NetCDF file
ds = xr.open_dataset(dataset_file)

# Display the dataset
#st.write(type(ds))


df = ds.to_dataframe().reset_index().set_index("time")

df["datetime"] = df.index
#df = df["2022-01-01": "2022-12-31"]


#print(type(df))
st.write("Dataset of Air Quality - Delhi")

uniquestations = df.station.unique()
st.write('There are',len(uniquestations),'Unique station')

dropped=df=df.dropna(subset=["PM2.5"])

uniquestations = df.station.unique()
st.write('After Removing NAN values from Dataset we have',len(uniquestations),'unique stations of Delhi')


st.dataframe(df)

st.write('After Interpolation')
df = df.ffill().bfill()
st.dataframe(df)


import plotly.express as px
fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", hover_name=df["station"],
                        zoom=8, height=500, color_discrete_sequence=['blue'])
fig.update_layout(mapbox_style="open-street-map", title="Delhi Air Stations")
#fig.show()
st.plotly_chart(fig)

df['time_']=df['datetime']


#TrainTest Split
all_stations = df.station.unique()
train_station, test_station = train_test_split(all_stations, test_size=0.2, random_state=42)
X_train = df[df.station.isin(train_station)]
X_test = df[df.station.isin(test_station)]

#X_train.station.unique().shape, X_test.shape


df.set_index("time_")
X_train = X_train[['datetime','latitude','longitude','PM2.5']]
X_test = X_test[['datetime','latitude','longitude','PM2.5']]



import subprocess
package_url = "https://pypi.nvidia.com/cuml/cuml_cu11-<version>.tar.gz"
package_file = wget.download(package_url)
subprocess.call(["pip", "install", package_file])


from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

# Load your data (replace with your own data loading code)
# X_train, X_test, y_train, y_test = load_data()

# Function to fit the model
def fit_model(x):
    X, y = x.iloc[:, 1:-1], x.iloc[:, -1]
    model = gs.fit(X, y)
    return model

# Add Streamlit sidebar widgets for parameter selection
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

  
  
st.write("Hello")
