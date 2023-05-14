import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import NearestNDInterpolator
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
st.write(ds)


df = ds.to_dataframe().reset_index().set_index("time")
df = df["2022-01-01": "2022-12-31"]
print(type(df))
df["datetime"] = df.index



st.write("Hello")
