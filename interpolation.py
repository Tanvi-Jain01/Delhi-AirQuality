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




from cuml.neighbors import KNeighborsRegressor
from time import time
from sklearn.model_selection import GridSearchCV
# from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd


gs= KNeighborsRegressor(n_neighbors=12)


# Train a model for each time step
init = time()
model_list=[]
def fit_model(x,model_list):    
    X, y = x.iloc[:,1:-1],x.iloc[:,-1]
    # print(X)
    model = gs.fit(X, y)
    #model = gs.fit(X, y).best_estimator_
    model_list.append(model)
    return model

# Group train data by datetime and apply fit_model function
train_time_df = X_train.groupby("datetime")
model = train_time_df.apply(fit_model,model_list=model_list)
print(train_time_df)
print(f"Training finished in {time() - init:.2f} seconds")


# Predict for each time step
def predict_model(x,model):
    # model = x.iloc[0]  # Get the model from the groupby result
    return model.predict(x[:,1:])

# Group test data by datetime and apply predict_model function

test_time_df = X_test.groupby("datetime")
predn_list=[]
for i,j in enumerate(test_time_df.groups.keys()):
  # test_time_df.dtype
  group_a = test_time_df.get_group(j)
  print(group_a)
  predns = model_list[i].predict(group_a.iloc[:,1:-1])
  predn_list.append(predns)

  
 predict = pd.concat([pd.DataFrame(predn) for predn in predn_list], ignore_index=True)
test_time_df = predict.rename(columns={0: "pred_y"})
test_time_df["true_y"] = X_test["PM2.5"].reset_index(drop=True)



# Compute RMSE for each time step
test_time_df["RMSE"] = np.sqrt(mean_squared_error(test_time_df["true_y"], test_time_df["pred_y"]))

test_time_df["RMSE"].plot(figsize=(20, 7), grid=True)
print(test_time_df)  
  
  
st.write("Hello")
