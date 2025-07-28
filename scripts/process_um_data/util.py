from functools import partial
import asyncio
import random
import subprocess as sp

import botocore.exceptions
import iris
from iris.experimental.stratify import relevel
from loguru import logger
import numpy as np
import stratify

# import xarray as xr
# async def async_retry_open_zarr(url, max_retries=20):
#     retries = 0
#     while retries < max_retries:
#         try:
#             ds = xr.open_zarr(url)
#             logger.debug(f'Successfully opened {url}')
#             return ds
#         except Exception as e:
#             # This has started (30/4/2025) raising exceptions
#             # It has previously been fine.
#             logger.warning(f'Failed to open {url}')
#             logger.warning(e)
#             retries += 1
#             # Sleep 10s, then 20s... with 5s jitter.
#             timeout = 1 * retries + random.uniform(-0.5, 0.5)
#             logger.warning(f'sleeping for {timeout} s')
#             await asyncio.sleep(timeout)
#     raise Exception(f'failed to open {url} after {retries} retries')


async def async_da_to_zarr_with_retries(da, store, region, max_retries=5):
    # This got complicated quite quickly. I was getting exceptions intermittently with da.to_zarr. Handling these
    # exceptions wasn't working because it was being thrown from async code, so I needed to write my own async func
    # so I could call await asyncio.sleep(...). This is how you call it. ChatGPT helped with the async stuff.
    retries = 0
    success = False
    while retries < max_retries:
        try:
            da.to_zarr(store, region=region)
            success = True
            logger.debug(f'{da.name} successfully written to zarr')
            break
        except (botocore.exceptions.ClientError, OSError, PermissionError, FileNotFoundError) as e:
            # This has started (26/4/2025) raising exceptions, one as an inner, one as outer.
            # Not sure which exception is responsible/the one to catch.
            # It has previously been fine.
            logger.warning(f'Failed to write {da.name} to zarr store {store}')
            logger.warning(e)
            retries += 1
            # Sleep 10s, then 20s... with 5s jitter.
            timeout = 10 * retries + random.uniform(-5, 5)
            logger.debug(f'sleeping for {timeout} s')
            await asyncio.sleep(timeout)
    if not success:
        raise Exception(f'failed to write {da.name} to zarr store {store} after {retries} retries')


def model_level_to_pressure(cube, p, z, enforce_greater_than_zero=True):
    logger.debug(f're-level model level to pressure for {cube.name()}')
    cube = cube[-p.shape[0]:]
    assert (p.coord('time').points == cube.coord('time').points).all()

    # Direction of pressure_levels must match that of air_pressure/p.
    # This runs, but it also inverts the 3D fields! Fix by inverting output.
    pressure_levels = z.coord('pressure').points[::-1] * 100  # convert from hPa to Pa.
    interpolator = partial(stratify.interpolate,
                           interpolation=stratify.INTERPOLATE_LINEAR,
                           extrapolation=stratify.EXTRAPOLATE_LINEAR,
                           rising=False)
    new_cube_data = np.zeros((cube.shape[0], len(pressure_levels), cube.shape[2], cube.shape[3]))
    for i in range(cube.shape[0]):
        logger.trace(i)
        regridded_cube = relevel(cube[i], p[i], pressure_levels, interpolator=interpolator)
        # logger.trace(f'regridded_cube.data.sum() {regridded_cube.data.sum()}')
        # Fix 3D fields so that they are the right way round - invert output.
        new_cube_data[i] = regridded_cube.data[::-1]

    if enforce_greater_than_zero:
        # Some values are ending up as negatives (why? Perhaps due to linear extrap. outside domain?)
        # Enfore greater than zero if so (these are all for mass_ fields - must by >= 0).
        new_cube_data[new_cube_data < 0] = 0

    coords = [(cube.coord('time'), 0), (z.coord('pressure'), 1), (cube.coord('latitude'), 2),
              (cube.coord('longitude'), 3)]
    new_cube = iris.cube.Cube(new_cube_data,
                              long_name=cube.name(),
                              units=cube.units,
                              dim_coords_and_dims=coords,
                              attributes=cube.attributes)
    logger.trace(new_cube)
    return new_cube


def sysrun(cmd):
    return sp.run(cmd, check=True, shell=True, stdout=sp.PIPE, stderr=sp.PIPE, encoding='utf8')



