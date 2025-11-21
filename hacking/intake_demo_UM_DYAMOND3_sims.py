#!/usr/bin/env python
# coding: utf-8

import math as maths
import cartopy.crs as ccrs
import intake
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import easygems.healpix as egh
import healpy

def get_nn_lon_lat_index(nside, lons, lats):
    lons2, lats2 = np.meshgrid(lons, lats)
    return xr.DataArray(
        healpy.ang2pix(nside, lons2, lats2, nest=True, lonlat=True),
        coords=[("lat", lats), ("lon", lons)],
    )
my_zoom=8
idx = get_nn_lon_lat_index(
    2**my_zoom, np.linspace(-180, 180, 1800), np.linspace(-15, 15, 150)
)

cat = intake.open_catalog('https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml')['online']
sim = cat['icon_d3hp003']

sim(zoom=my_zoom).urlpath
ds3_icon = sim(zoom=my_zoom, time='PT3H').to_dask()
#ds3_icon

rlut_lon_lat_icon = ds3_icon.rlut.isel(cell=idx)
rlut_lon_lat_icon_6h=rlut_lon_lat_icon.isel(time=slice(None,None,2))

rlut_lon_lat_icon_6h.to_netcdf("/home/users/plvidale/hackathon/icon_d3hp003_rlut_6h.nc")

