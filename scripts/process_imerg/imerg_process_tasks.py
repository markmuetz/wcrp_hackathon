import asyncio
import datetime as dt
import json
from timeit import default_timer as timer
from io import StringIO
from pathlib import Path

import dask
import dask.array
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from loguru import logger

import sys
sys.path.insert(0, '../process_um_data')
from um_latlon_pp_to_healpix_nc import UMLatLon2HealpixRegridder, gen_weights, get_limited_healpix
from healpix_coarsen import async_da_to_zarr_with_retries

from imerg_config import vn

chunks2d = {
    9: (1, 12 * 4**9),
    8: (1, 12 * 4**8),
    7: (4, 12 * 4**7),
    6: (4**2, 12 * 4**6),
    5: (4**3, 12 * 4**5),
    4: (4**4, 12 * 4**4),
    3: (4**5, 12 * 4**3),
    2: (4**6, 12 * 4**2),
    1: (4**7, 12 * 4),
    0: (4**8, 12),
}

s3cfg = dict([l.split(' = ') for l in Path('/home/users/mmuetz/.s3cfg').read_text().split('\n') if l])

def fix_coords(ds, lat_dim="lat", lon_dim="lon"):
    # Find where longitude crosses from negative to positive (approx. where lon=0)
    lon_0_index = (ds[lon_dim] < 0).sum().item()

    # Create indexers for the roll
    lon_indices = np.roll(np.arange(ds.sizes[lon_dim]), -lon_0_index)

    # Roll dataset and convert longitudes to 0-360 range
    ds = ds.isel({lon_dim: lon_indices})
    lon360 = xr.where(ds[lon_dim] < 0, ds[lon_dim] + 360, ds[lon_dim])
    ds = ds.assign_coords({lon_dim: lon360})

    # Ensure latitude and longitude are in ascending order if needed
    if np.all(np.diff(ds[lat_dim].values) < 0):
        ds = ds.isel({lat_dim: slice(None, None, -1)})
    if np.all(np.diff(ds[lon_dim].values) < 0):
        ds = ds.isel({lon_dim: slice(None, None, -1)})

    return ds


def da_to_zarr(da, zarr_store_url_tpl, zoom, nan_checks=False):
    from loguru import logger
    name = da.name
    logger.info(f'{name} to zarr for zoom level {zoom}')

    # zarr_store_name = group['zarr_store']
    url = zarr_store_url_tpl.format(zoom=zoom)
    jasmin_s3 = s3fs.S3FileSystem(
        anon=False,
        secret=s3cfg['secret_key'],
        key=s3cfg['access_key'],
        client_kwargs={'endpoint_url': 'http://hackathon-o.s3.jc.rl.ac.uk'}
    )

    zarr_store = s3fs.S3Map(
        root=url,
        s3=jasmin_s3, check=False)

    # Open the zarr store so I can get time coord to match with da's time coord.
    ds_tpl = xr.open_zarr(zarr_store)

    # Match source (da) to target (zarr_store) times.
    # Get an index into zarr store to allow me to write block of da's data.
    timename = 'time'
    source_times_to_match = pd.DatetimeIndex(da[timename].values)
    target_times_to_match = pd.DatetimeIndex(ds_tpl[timename].values)

    idx = np.argmin(np.abs(source_times_to_match[0] - target_times_to_match))
    assert (np.abs(source_times_to_match - target_times_to_match[idx: idx + len(source_times_to_match)]) < pd.Timedelta(
        minutes=5)).all(), 'source times do not match target times (thresh = 5 mins)'

    logger.debug(
        f'writing {name} to zarr store {url} (idx={idx}, time={source_times_to_match[0]})')
    if nan_checks:
        if np.isnan(da.values).all():
            logger.error(da)
            raise Exception(f'da {da.name} is full of NaNs')

    region = {timename: slice(idx, idx + len(da[timename])), 'cell': slice(None)}
    # Handle errors if they arise (started happening on 26/4/25).
    asyncio.run(async_da_to_zarr_with_retries(da, zarr_store, region))
    return name


def da_to_healpix(da, zoom, regional_chunks=None):
    from loguru import logger

    lonname = 'lon'
    latname = 'lat'
    weights_path = Path('/gws/nopw/j04/hrcm/mmuetz/weights/') / weights_filename(da, zoom, lonname, latname,
                                                                                 True, True)
    logger.trace(f'  - using weights: {weights_path}')
    regridder = UMLatLon2HealpixRegridder(method='easygems_delaunay', zoom_level=zoom, weights_path=weights_path,
                                          add_cyclic=True, regional=True, regional_chunks=regional_chunks)

    da_hp = regridder.regrid(da, lonname, latname)

    return da_hp


def weights_filename(da, zoom, lonname, latname, add_cyclic, regional):
    lon0, lonN = da[lonname].values[[0, -1]]
    lat0, latN = da[latname].values[[0, -1]]
    lonstr = f'({lon0.item():.3f},{lonN.item():.3f},{len(da[lonname])})'
    latstr = f'({lat0.item():.3f},{latN.item():.3f},{len(da[latname])})'

    return f'regrid_weights.hpz{zoom}.cyclic_lon={add_cyclic}.regional={regional}.lon={lonstr}.lat={latstr}.nc'


class ImergProcessTasks:
    def __init__(self):
        self.debug_log = StringIO()
        logger.add(self.debug_log)
        deploy = 'dev'
        self.zarr_store_url_tpl = f's3://obs-data/{deploy}/{vn}/IR_IMERG_combined/IR_IMERG_combined_V07B.hp_z{{zoom}}.zarr'
        self.max_zoom = 9

    def create_empty_zarr_stores(self, task):
        inpath = task['inpath']
        zarr_time = pd.date_range(task['first_date'], pd.Timestamp(task['last_date']) + pd.Timedelta(hours=1), freq='1h')

        jasmin_s3 = s3fs.S3FileSystem(
            anon=False,
            secret=s3cfg['secret_key'],
            key=s3cfg['access_key'],
            client_kwargs={'endpoint_url': 'http://hackathon-o.s3.jc.rl.ac.uk'}
        )

        metadata = {
            'regional': True,
            'geospatial_lat_min': -60,
            'geospatial_lat_max': 60,
            'geospatial_lon_min': 0,
            'geospatial_lon_max': 360,
        }

        ds = xr.open_dataset(inpath).pipe(fix_coords).isel(time=slice(None, None, 2))
        for zoom in range(self.max_zoom + 1)[::-1]:
            chunks = chunks2d[zoom]
            # Multipls of 4**n chosen as these will align well with healpix grids.
            # Aim for 1-10MB per chunk, bearing in mind that this is saved with 4-byte float32s.
            logger.trace(f'chunks={chunks}')
            # weights_path = Path('/gws/nopw/j04/hrcm/mmuetz/weights/') / weights_filename(da, zoom, lonname, latname,
            #                                                                              add_cyclic, regional)
            ds_tpl = xr.Dataset()
            for name, da in ds.data_vars.items():
                da_tpl = self.create_dataarray_template(da, chunks, zoom, zarr_time)
                ds_tpl[name] = da_tpl

                if zoom != self.max_zoom:
                    dummies = dask.array.zeros(da_tpl.shape, dtype=np.float32, chunks=chunks)
                    ds_tpl[f'{da.name}_weights'] = xr.DataArray(dummies, name=f'{da.name}_weights', coords=da_tpl.coords)

            # This means that the data will be plottable (by easygems.healpix.healpix_show at least) even when it
            # is regional.
            crs = xr.DataArray(
                name="crs",
                attrs={
                    "grid_mapping_name": "healpix",
                    "healpix_nside": 2 ** zoom,
                    "healpix_order": "nest",
                },
            )
            ds_tpl = ds_tpl.assign_coords(crs=crs)
            ds_tpl.attrs.update(ds.attrs)
            ds_tpl.attrs.update(metadata)

            logger.info(f'Saving IMERG zoom={zoom}')
            store_url = self.zarr_store_url_tpl.format(zoom=zoom)
            zarr_store = s3fs.S3Map(
                root=store_url,
                s3=jasmin_s3, check=False)
            logger.debug(store_url)
            logger.debug(ds_tpl)
            ds_tpl.to_zarr(zarr_store, mode='a', compute=False)

    def create_dataarray_template(self, da, chunks, zoom, zarr_time):
        lonname = 'lon'
        latname = 'lat'

        weights_path = Path('/gws/nopw/j04/hrcm/mmuetz/weights/') / weights_filename(da, zoom, lonname, latname,
                                                                                     True, True)
        if not weights_path.exists():
            logger.info(f'No weights for {da.name}, generating')
            # chunks[-1] selects the spatial chunk.
            gen_weights(da, zoom, lonname, latname, add_cyclic=True, regional=True,
                        regional_chunks=chunks[-1], weights_path=weights_path)

        minlon, maxlon = da[lonname].values[[0, -1]]
        minlat, maxlat = da[latname].values[[0, -1]]
        extent = [minlon, maxlon, minlat, maxlat]
        _, _, ichunk = get_limited_healpix(extent, zoom=zoom, chunksize=chunks[-1])
        cells = ichunk
        dims = ['time', 'cell']
        coords = {'time': zarr_time, 'cell': cells}
        shape = (len(zarr_time), len(cells))

        dummies = dask.array.zeros(shape, dtype=np.float32, chunks=chunks)
        da_tpl = xr.DataArray(dummies, dims=dims, coords=coords, name=da.name, attrs=da.attrs)
        # da_tpl.attrs['UM_name'] = da.name
        # da_tpl.attrs['long_name'] = long_name
        # da_tpl.attrs['grid_mapping'] = 'healpix_nested'
        # da_tpl.attrs['healpix_zoom'] = zoom
        return da_tpl

    def regrid(self, task):
        inpaths = task['inpaths']
        logger.debug(f'Opening # paths: {len(inpaths)}')
        ds = xr.open_mfdataset(inpaths).pipe(fix_coords).isel(time=slice(None, None, 2))

        zoom = self.max_zoom
        chunks = chunks2d[zoom]

        for name, da in ds.data_vars.items():
            logger.info(f'Regridding {name}')
            da_hp = da_to_healpix(da, zoom, regional_chunks=chunks[-1])
            da_to_zarr(da_hp, self.zarr_store_url_tpl, zoom)

    # def coarsen_healpix_region(self, task):
    #     dim = task['dim']
    #     tgt_zoom = task['tgt_zoom']
    #     src_zoom = tgt_zoom + 1
    #
    #     freqs = {
    #         '2d': 'PT1H',
    #         '3d': 'PT3H',
    #     }
    #     rel_url_tpl = self.config['zarr_store_url_tpl'][5:]  # chop off 's3://'
    #     freq = freqs[dim]
    #     urls = {
    #         z: rel_url_tpl.format(freq=freq, zoom=z)
    #         for z in range(11)
    #     }
    #     src_store = s3fs.S3Map(root=urls[src_zoom], s3=jasmin_s3, check=False)
    #     tgt_store = s3fs.S3Map(root=urls[tgt_zoom], s3=jasmin_s3, check=False)
    #
    #     chunks = self.config['groups'][dim]['chunks']
    #     zarr_chunks = {'time': chunks[tgt_zoom][0], 'cell': -1}
    #     src_ds = xr.open_zarr(src_store, chunks=zarr_chunks)
    #     regional = self.config['regional']
    #
    #     for subtask in task['tgt_times']:
    #         start_idx = subtask['start_idx']
    #         end_idx = subtask['end_idx']
    #         logger.debug((start_idx, end_idx))
    #         coarsen_healpix_zarr_region(src_ds, tgt_store, tgt_zoom, dim, start_idx, end_idx, chunks, regional)
    #
    #     logger.info('completed')


def slurm_run(tasks, array_index):
    start = timer()
    task = tasks[array_index]
    # logger.debug(task)
    proc = ImergProcessTasks()
    if task['task_type'] == 'regrid':
        proc.regrid(task)
    elif task['task_type'] == 'create_empty_zarr_stores':
        proc.create_empty_zarr_stores(task)
    # elif task['task_type'] == 'coarsen':
    #     proc.coarsen_healpix_region(task)
    else:
        raise Exception(f'unknown task type {task["task_type"]}')

    end = timer()
    logger.info(f'Completed in: {end - start:.2f}s')
    if task.get('donepath', ''):
        Path(task['donepath']).write_text(proc.debug_log.getvalue())

    return proc


if __name__ == '__main__':
    logger.remove()
    custom_fmt = ("<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                  "<level>{level: <8}</level> | "
                  "<blue>[{process.id}]</blue><cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    logger.add(sys.stderr, level="TRACE", format=custom_fmt, colorize=not sys.argv[1].startswith('slurm'))
    filepath = Path(__file__)
    logger.debug('{} last edited: {:%Y-%m-%d %H:%M:%S}'.format(filepath.name,
                                                               dt.datetime.fromtimestamp(filepath.stat().st_mtime)))

    logger.debug(sys.argv)

    if sys.argv[1] == 'slurm':
        tasks_path = sys.argv[2]
        logger.info(tasks_path)
        with Path(tasks_path).open('r') as f:
            tasks = json.load(f)

        slurm_run(tasks, int(sys.argv[3]))
    else:
        raise Exception(f'Unknown command {sys.argv[1]}')
