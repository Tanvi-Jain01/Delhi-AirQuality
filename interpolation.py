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
import matplotlib.pyplot as plt
# import netcdf4 
import dask.distributed
import dask.array as da



# URL of the NetCDF file on GitHub
nc_url = 'https://github.com/patel-zeel/delhi_aq/raw/main/data/delhi_cpcb_2022.nc'

# Read the NetCDF file
ds = xr.open_dataset(nc_url)

# Display the dataset
st.write(ds)






st.write("Hello")
