"""Includes all the code for coarsening from one healpix level to the next lower."""
import asyncio

import numpy as np
import xarray as xr
from loguru import logger

from util import async_da_to_zarr_with_retries

# TODO: docstrings.


def nan_weight(arr, axis):
    return ((~np.isnan(arr)).sum(axis=axis) / 4).astype(np.float32)


def map_regional_to_global(_da, src_zoom, dim):
    logger.debug(_da.name)
    coords = {
        n: c for n, c in _da.coords.items()
        if n != 'cell'
    }
    coords['cell'] = np.arange(12 * 4 ** src_zoom)

    if _da.name == 'weights':
        dims = ['cell']
        shape = [len(coords['cell'])]
    else:
        if dim == '2d':
            dims = ['time', 'cell']
        elif dim == '3d':
            dims = ['time', 'pressure', 'cell']
        # dims = [d for d in coords if d != 'crs']
        shape = list(_da.shape[:-1]) + [len(coords['cell'])]
    # Would be nice, but it's actually quite a bit slower?
    # dummies = dask.array.zeros(shape, dtype=np.float32)
    # da_new = xr.DataArray(dummies, dims=dims, name=_da.name, coords=coords)
    da_new = xr.DataArray(np.full(shape, np.nan, np.float32), dims=dims, name=_da.name, coords=coords)
    # Dangerously wrong! Will silently ignore this.
    # da_new.isel(cell=_da.cell).values = _da.values

    # This on the other hand works perfectly, because .loc allows assignment.
    da_new.loc[dict(cell=_da.cell)] = _da
    return da_new


def map_global_to_regional(_da, src_ds_region, tgt_ds_store, dim):
    logger.debug(_da.name)
    coords = {
        n: c for n, c in _da.coords.items()
        if n != 'cell'
    }
    coords['cell'] = tgt_ds_store.cell
    if _da.name == 'weights':
        dims = ['cell']
        shape = [len(coords['cell'])]
    else:
        if dim == '2d':
            dims = ['time', 'cell']
        elif dim == '3d':
            dims = ['time', 'pressure', 'cell']
        da = list(src_ds_region.data_vars.values())[0]
        shape = list(da.shape[:-1]) + [len(coords['cell'])]

    da_new = xr.DataArray(np.full(shape, np.nan, np.float32), dims=dims, name=_da.name, coords=coords)
    da_new.values = _da.isel(cell=tgt_ds_store.cell.values)
    return da_new


def calc_tgt_weights(src_ds, tgt_ds, tgt_zoom, dim):
    # It is enough to calc the weights for one field, and store.
    if dim == '2d':
        src_da = list(src_ds.data_vars.values())[0].isel(time=0)
    elif dim == '3d':
        src_da = list(src_ds.data_vars.values())[0].isel(time=0, pressure=0)
    weights = src_da.coarsen(cell=4).reduce(nan_weight).compute()
    cells = np.arange(12 * 4 ** tgt_zoom)
    coords = {'cell': cells}
    # Use a global weights DataArray to convert between the src domain and the tgt domain.
    weights_global = xr.DataArray(np.full(12 * 4 ** tgt_zoom, np.nan, np.float32), dims=['cell'], coords=coords)
    # I'm found it difficult to figure out how to use cells to map into weights_global
    # But this works.
    tgt_cell_from_src = ((src_ds.cell.values.reshape(-1, 4).mean(axis=-1) - 1.5) / 4).astype(int)
    weights_global.loc[dict(cell=tgt_cell_from_src)] = weights.values
    tgt_weights = weights_global.isel(cell=tgt_ds.cell)
    logger.debug(float(np.isnan(tgt_weights).sum().values))
    return tgt_weights


def coarsen_healpix_zarr_region(src_ds, tgt_store, tgt_zoom, dim, start_idx, end_idx, chunks, regional=False):
    time_slice = slice(start_idx, end_idx)
    src_zoom = tgt_zoom + 1
    logger.info(f'Coarsen to {tgt_zoom}, time_slice={time_slice}')

    tgt_chunks = chunks[tgt_zoom]

    src_ds_region = src_ds.isel(time=time_slice)
    if regional:
        if dim == '3d' and 'weights' in src_ds.data_vars.keys():
            logger.debug('drop weights')
            src_ds_region = src_ds_region.drop_vars('weights')
        src_ds_region = src_ds_region.map(map_regional_to_global, src_zoom=src_zoom, dim=dim)

    # To compute or not to compute...
    # Computing would force the download and loading into mem, but might stop contention for the resource?
    # tgt_ds = src_ds_region.coarsen(cell=4).mean()
    logger.debug('compute tgt_ds')
    tgt_ds = src_ds_region.coarsen(cell=4).mean().compute()

    logger.info('modify metadata')
    if dim == '2d':
        preferred_chunks = {'time': tgt_chunks[0], 'cell': tgt_chunks[1]}
    else:
        preferred_chunks = {'time': tgt_chunks[0], 'pressure': tgt_chunks[1], 'cell': tgt_chunks[2]}
    for da in tgt_ds.data_vars.values():
        da.attrs['healpix_zoom'] = tgt_zoom
        da.encoding['chunks'] = tgt_chunks
        da.encoding['preferred_chunks'] = preferred_chunks

    if regional:
        zarr_chunks = {'time': chunks[tgt_zoom][0], 'cell': -1}
        tgt_ds_store = xr.open_zarr(tgt_store, chunks=zarr_chunks)
        tgt_ds = tgt_ds.map(map_global_to_regional, src_ds_region=src_ds_region, tgt_ds_store=tgt_ds_store, dim=dim)
        if (src_ds_region.time[0] == src_ds.time[0]) and 'weights' not in src_ds.data_vars:
            tgt_weights = calc_tgt_weights(src_ds, tgt_ds, tgt_zoom, dim)
            tgt_ds['weights'] = tgt_weights
            logger.debug(float(np.isnan(tgt_ds.weights).sum().values))

    tgt_ds = tgt_ds.drop_vars('crs')

    if dim == '2d':
        region = {'time': time_slice, 'cell': slice(None)}
    elif dim == '3d':
        region = {'time': time_slice, 'pressure': slice(None), 'cell': slice(None)}
    else:
        raise ValueError(f'dim {dim} is not supported')
    for da in tgt_ds.data_vars.values():
        if np.isnan(da.values).all():
            logger.error(f'ERROR: ALL VALUES ARE NAN FOR {da.name}')
        else:
            logger.debug('values not all nan')
        if da.name in ['orog', 'sftlf']:
            continue
        logger.debug(f'  writing {da.name}')
        if da.name == 'weights':
            if src_ds_region.time[0] == src_ds.time[0]:
                if dim == '2d':
                    region = {'cell': slice(None)}
                    asyncio.run(
                        async_da_to_zarr_with_retries(da.chunk({'cell': preferred_chunks['cell']}), tgt_store, region))
                elif dim == '3d':
                    # TODO: still causes error due to dims mismatch.
                    logger.warning('Not writing weights for 3D data')
                    # region = {'pressure': slice(None), 'cell': slice(None)}
                    # asyncio.run(
                    #     async_da_to_zarr_with_retries(da.chunk({'cell': preferred_chunks['cell']}), tgt_store, region))
            continue
        asyncio.run(
            async_da_to_zarr_with_retries(da.chunk({'cell': preferred_chunks['cell']}), tgt_store, region))
