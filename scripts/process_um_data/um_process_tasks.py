import asyncio
import datetime as dt
import json
import sys
from timeit import default_timer as timer
from collections import defaultdict
from functools import partial
from io import StringIO
from pathlib import Path

import dask
import dask.array
from dask.distributed import LocalCluster
import iris
import iris.exceptions
import numpy as np
import pandas as pd
import s3fs
import stratify
import xarray as xr
from iris.experimental.stratify import relevel
from loguru import logger

from processing_config import processing_config, shared_metadata, cube_cell_method_is_not_empty
from healpix_coarsen import coarsen_healpix_zarr_region, async_da_to_zarr_with_retries
from latlon_to_healpix import LatLon2HealpixRegridder, gen_weights, get_limited_healpix

s3cfg = dict([l.split(' = ') for l in Path('/home/users/mmuetz/.s3cfg').read_text().split('\n') if l])
iris.FUTURE.date_microseconds = True


def model_level_to_pressure(cube, p, z):
    from loguru import logger
    logger.debug(f're-level model level to pressure for {cube.name()}')
    # p = cubes.extract_cube('air_pressure')
    cube = cube[-p.shape[0]:]
    assert (p.coord('time').points == cube.coord('time').points).all()

    # z = cubes.extract_cube('geopotential_height')
    # Direction of pressure_levels must match that of air_pressure/p.
    # TODO: This runs, but it also inverts the 3D fields! Fix!!
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
        new_cube_data[i] = regridded_cube.data

    coords = [(cube.coord('time'), 0), (z.coord('pressure'), 1), (cube.coord('latitude'), 2),
              (cube.coord('longitude'), 3)]
    new_cube = iris.cube.Cube(new_cube_data,
                              long_name=cube.name(),
                              units=cube.units,
                              dim_coords_and_dims=coords,
                              attributes=cube.attributes)
    logger.trace(new_cube)
    return new_cube


def da_to_zarr(da, zarr_store_url_tpl, group_name, group, zoom, regional, nan_checks=False):
    from loguru import logger
    name = da.name
    logger.info(f'{name} to zarr for zoom level {zoom}')

    zarr_store_name = group['zarr_store']
    url = zarr_store_url_tpl.format(freq=zarr_store_name, zoom=zoom)
    jasmin_s3 = s3fs.S3FileSystem(
        anon=False,
        secret=s3cfg['secret_key'],
        key=s3cfg['access_key'],
        client_kwargs={'endpoint_url': 'http://hackathon-o.s3.jc.rl.ac.uk'}
    )

    zarr_store = s3fs.S3Map(
        root=url,
        s3=jasmin_s3, check=False)

    half_time = find_halfpast_time(da)
    # Open the zarr store so I can get time coord to match with da's time coord.
    ds_tpl = xr.open_zarr(zarr_store)

    # Match source (da) to target (zarr_store) times.
    # Get an index into zarr store to allow me to write block of da's data.
    timename = [c for c in da.coords if c.startswith('time')][0]
    if timename == half_time:
        # This da is stored on the half hour because it's an hourly mean.
        # Need to add on half an hour here and then align with ds_tpl times.
        times_halfpast = pd.DatetimeIndex(da[timename].values)
        source_times_to_match = times_halfpast + pd.Timedelta(minutes=30)
    else:
        source_times_to_match = pd.DatetimeIndex(da[timename].values)
    zarr_time_name = 'time'
    target_times_to_match = pd.DatetimeIndex(ds_tpl[zarr_time_name].values)

    da = da.rename(**{timename: zarr_time_name})
    idx = np.argmin(np.abs(source_times_to_match[0] - target_times_to_match))
    assert (np.abs(source_times_to_match - target_times_to_match[idx: idx + len(source_times_to_match)]) < pd.Timedelta(
        minutes=5)).all(), 'source times do not match target times (thresh = 5 mins)'

    logger.debug(
        f'writing {name} to zarr store {url} (idx={idx}, time={source_times_to_match[0]})')
    if nan_checks:
        if np.isnan(da.values).all():
            logger.error(da)
            raise Exception(f'da {da.name} is full of NaNs')
        if not regional and np.isnan(da.values).any():
            logger.warning(f'da {da.name} contains NaNs')

    if group_name.startswith('2d'):
        region = {zarr_time_name: slice(idx, idx + len(da[zarr_time_name])), 'cell': slice(None)}
    elif group_name.startswith('3d'):
        region = {zarr_time_name: slice(idx, idx + len(da[zarr_time_name])), 'pressure': slice(None),
                  'cell': slice(None)}
    else:
        raise ValueError(f'group name {group_name} not recognized')
    # Handle errors if they arise (started happening on 26/4/25).
    asyncio.run(async_da_to_zarr_with_retries(da, zarr_store, region))
    return name


def cube_to_da(cube):
    from loguru import logger
    logger.debug(f'convert {cube.name()} from iris to xarray')
    da = xr.DataArray.from_iris(cube)
    # For some cubes (ones with names like m01s30i461 the da gets a name like filled-XXXXXX...
    # Make sure it's got the actual cube name so I can rename it later.
    return da.rename(cube.name())


def da_to_healpix(da, zoom, regrid_method, name_map, drop_vars, add_cyclic=True, regional=False, regional_chunks=None):
    from loguru import logger
    um_name = da.name
    long_name, short_name = name_map[um_name]

    lonname = [c for c in da.coords if c.startswith('longitude')][0]
    latname = [c for c in da.coords if c.startswith('latitude')][0]
    if regrid_method == 'easygems_delaunay':
        # TODO: hardcoded path
        weights_path = Path('/gws/nopw/j04/hrcm/mmuetz/weights/') / weights_filename(da, zoom, lonname, latname,
                                                                                     add_cyclic, regional)
        logger.trace(f'  - using weights: {weights_path}')
        regridder = LatLon2HealpixRegridder(weights_path=weights_path, method='easygems_delaunay', zoom_level=zoom,
                                            add_cyclic=add_cyclic, regional=regional, regional_chunks=regional_chunks)
    else:
        regridder = LatLon2HealpixRegridder(weights_path=None, method=regrid_method, zoom_level=zoom,
                                            add_cyclic=add_cyclic, regional=regional, regional_chunks=regional_chunks)

    # These have to be dropped before you cyclic pad *some* data arrays, or you will get a coord mismatch.
    drop_vars_exists = list(set(drop_vars) & set(k for k in da.coords.keys()))
    logger.debug(f'dropping {drop_vars_exists}')
    da = da.drop_vars(drop_vars_exists)

    da_hp = regridder.regrid(da, lonname, latname)
    da_hp = da_hp.rename(short_name)
    da_hp.attrs['UM_name'] = um_name
    da_hp.attrs['long_name'] = long_name
    da_hp.attrs['grid_mapping'] = 'healpix_nested'
    da_hp.attrs['healpix_zoom'] = zoom

    return da_hp


def weights_filename(da, zoom, lonname, latname, add_cyclic, regional):
    lon0, lonN = da[lonname].values[[0, -1]]
    lat0, latN = da[latname].values[[0, -1]]
    lonstr = f'({lon0.item():.3f},{lonN.item():.3f},{len(da[lonname])})'
    latstr = f'({lat0.item():.3f},{latN.item():.3f},{len(da[latname])})'

    return f'regrid_weights.hpz{zoom}.cyclic_lon={add_cyclic}.regional={regional}.lon={lonstr}.lat={latstr}.nc'


def find_halfpast_time(ds):
    times = {name: pd.DatetimeIndex(ds[name].values)
             for name in ds.coords if name.startswith('time')}
    for name, time in times.items():
        if ((time.second == 0) & (time.minute == 30)).all():
            return name
    return None


def get_regional_bounds(da):
    if 'latitude' in da.coords and 'longitude' in da.coords:
        bounds = {
            'lower_left_lat': float(round(da.latitude.values[0], 3)),
            'lower_left_lon': float(round(da.longitude.values[0] % 360, 3)),
            'upper_right_lat': float(round(da.latitude.values[-1], 3)),
            'upper_right_lon': float(round(da.longitude.values[-1] % 360, 3)),
        }
        return bounds
    else:
        return None


class UMProcessTasks:
    def __init__(self, config):
        self.config = config
        self.drop_vars = config['drop_vars']
        self.groups = config['groups']

        self.debug_log = StringIO()
        logger.add(self.debug_log)

    def create_empty_zarr_stores(self, task):
        inpaths = task['inpaths']
        zarr_store_url_tpl = self.config['zarr_store_url_tpl']

        cubes = iris.load(inpaths)
        logger.trace(cubes)

        regional = self.config.get('regional', False)
        metadata = {
            **{
                'bounds': None,
                'latitiude_convention': '[-90, 90]',
                'longitude_convention': '[0, 360]',
                'regional': regional,
            },
            **self.config['metadata'],
            **shared_metadata,
        }
        if not regional:
            metadata['bounds'] = {
                'lower_left_lat': -90,
                'lower_left_lon': 0,
                'upper_right_lat': 90,
                'upper_right_lon': 360,
            }

        grouped_da = defaultdict(list)
        for group_name, group in self.config['groups'].items():
            name_map = group['name_map']
            logger.info(f'Creating {group_name}')
            group_cubes = cubes.extract(group['constraint'])
            logger.info(f'Found {len(group_cubes)} cubes for {group_name}')

            for name in name_map.keys():
                constraint = group['extra_constraints'].get(name, name)
                try:
                    cube = group_cubes.extract_cube(constraint)
                except iris.exceptions.ConstraintMismatchError as e:
                    logger.warning(f'cube {name} not present')
                    continue

                # Want to be able to pass extra kwargs to from_iris but can't...
                # da = xr.DataArray.from_iris(cube, decode_timedelta=True)
                da = xr.DataArray.from_iris(cube).rename(cube.name())
                grouped_da[group_name].append(da)

        for zoom in range(self.config['max_zoom'] + 1)[::-1]:
            jasmin_s3 = s3fs.S3FileSystem(
                anon=False,
                secret=s3cfg['secret_key'],
                key=s3cfg['access_key'],
                client_kwargs={'endpoint_url': 'http://hackathon-o.s3.jc.rl.ac.uk'}
            )

            npix = 12 * 4 ** zoom
            ds_tpls = defaultdict(xr.Dataset)

            for group_name, group in self.config['groups'].items():
                time = group['time']
                zarr_time_name = 'time'
                zarr_time = time
                zarr_store_name = group['zarr_store']

                chunks = group['chunks'][zoom]
                # Multipls of 4**n chosen as these will align well with healpix grids.
                # Aim for 1-10MB per chunk, bearing in mind that this is saved with 4-byte float32s.
                logger.trace(f'chunks={chunks}')

                for da in grouped_da[group_name]:
                    da_tpl, short_name = self.create_dataarray_template(group, da, chunks, zoom, npix,
                                                                        zarr_time, zarr_time_name)
                    if regional:
                        if metadata['bounds'] is None:
                            metadata['bounds'] = get_regional_bounds(da)
                            logger.debug('bounds={}'.format(metadata['bounds']))
                    ds_tpls[zarr_store_name][short_name] = da_tpl

                if regional and zoom != self.config['max_zoom']:
                    coords = {n: c for n, c in da_tpl.coords.items() if not n == 'time'}
                    dummies = dask.array.zeros(da_tpl.shape[1:], dtype=np.float32, chunks=chunks[1:])
                    ds_tpls[zarr_store_name]['weights'] = xr.DataArray(dummies, name='weights', coords=coords)

            for zarr_store_name, ds_tpl in ds_tpls.items():
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
                ds_tpl.attrs.update(metadata)

                logger.info(f'Saving {task["config_key"]} zoom={zoom}')
                store_url = zarr_store_url_tpl.format(freq=zarr_store_name, zoom=zoom)

                zarr_store = s3fs.S3Map(
                    root=store_url,
                    s3=jasmin_s3, check=False)
                logger.debug(store_url)
                logger.debug(ds_tpl)
                ds_tpl.to_zarr(zarr_store, mode='a', compute=False)

    def create_dataarray_template(self, group, da, chunks, zoom, npix, zarr_time, zarr_time_name):
        long_name, short_name = group['name_map'][da.name]
        logger.info(f'- creating da for {da.name} -> {short_name}')

        timename = [c for c in da.coords if c.startswith('time')][0]
        lonname = [c for c in da.coords if c.startswith('longitude')][0]
        latname = [c for c in da.coords if c.startswith('latitude')][0]
        add_cyclic = self.config.get('add_cyclic', True)
        regional = self.config.get('regional', False)

        logger.trace((zoom, self.config['max_zoom']))
        if zoom == self.config['max_zoom']:
            # TODO: hardcoded path
            weights_path = Path('/gws/nopw/j04/hrcm/mmuetz/weights/') / weights_filename(da, zoom, lonname, latname,
                                                                                         add_cyclic, regional)
            if not weights_path.exists():
                logger.info(f'No weights for {da.name}, generating')
                # chunks[-1] selects the spatial chunk.
                gen_weights(da, weights_path=weights_path, zoom=zoom, lonname=lonname, latname=latname,
                            add_cyclic=add_cyclic, regional=regional, regional_chunks=chunks[-1])
        if regional:
            minlon, maxlon = da[lonname].values[[0, -1]]
            minlat, maxlat = da[latname].values[[0, -1]]
            extent = [minlon, maxlon, minlat, maxlat]
            _, _, ichunk = get_limited_healpix(extent, zoom=zoom, chunksize=chunks[-1])
            cells = ichunk
        else:
            cells = np.arange(npix)
        if da.ndim == 3:
            dims = ['time', 'cell']
            coords = {zarr_time_name: zarr_time, 'cell': cells}
            shape = (len(zarr_time), len(cells))
        elif da.ndim == 4:
            dims = ['time', 'pressure', 'cell']
            pressure_levels = [1, 5, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 750,
                               800, 850, 875, 900, 925, 950, 975, 1000]
            coords = {zarr_time_name: zarr_time,
                      'pressure': (['pressure'], pressure_levels, {'units': 'hPa'}),
                      'cell': cells}
            shape = (len(zarr_time), len(pressure_levels), len(cells))
        else:
            raise Exception('ndim must be 3 or 4')
        dummies = dask.array.zeros(shape, dtype=np.float32, chunks=chunks)
        da.attrs.update(group['extra_attrs'].get(da.name, {}))
        da_tpl = xr.DataArray(dummies, dims=dims, coords=coords, name=short_name, attrs=da.attrs)
        da_tpl = da_tpl.rename(**{timename: zarr_time_name})
        da_tpl.attrs['UM_name'] = da.name
        da_tpl.attrs['long_name'] = long_name
        da_tpl.attrs['grid_mapping'] = 'healpix_nested'
        da_tpl.attrs['healpix_zoom'] = zoom
        return da_tpl, short_name

    def regrid(self, task):
        # TODO: proc_extra_vars: only load nec.
        # Speed up for CoMA9_inst.
        inpaths = task['inpaths']
        # inpaths = task['inpaths'][1:2]

        logger.info('loading cubes')
        logger.trace(inpaths)
        cubes = iris.load(inpaths)

        add_cyclic = self.config.get('add_cyclic', True)
        regional = self.config.get('regional', False)

        # TODO: proc_extra_vars: maybe not avail.
        p = cubes.extract_cube('air_pressure')
        z = cubes.extract_cube('geopotential_height')

        for group_name, group in self.groups.items():
            logger.info(f'processing group {group_name}')
            group_constraint = group['constraint']
            name_map = group['name_map']
            chunks = group['chunks'][self.config['max_zoom']]
            group_cubes = cubes.extract(group_constraint)

            for name in name_map.keys():
                logger.info(f'Regridding {name}')
                constraint = group['extra_constraints'].get(name, name)
                try:
                    # TODO: try to handle in config rather than with logic.
                    if self.config.get('name', '').startswith('glm.n1280_GAL9_nest') and name == 'stratiform_rainfall_flux':
                        cube = group_cubes.extract_cube(constraint)
                        constraint2 = (iris.Constraint(name='convective_rainfall_flux') & iris.Constraint(
                            cube_func=cube_cell_method_is_not_empty))
                        cube2 = group_cubes.extract_cube(constraint2)
                        cube.data += cube2.data
                    else:
                        cube = group_cubes.extract_cube(constraint)
                except iris.exceptions.ConstraintMismatchError as e:
                    logger.debug(f'cube {name} not present')
                    continue
                cubes.remove(cube)
                if 'extra_processing' in group and name in group['extra_processing']:
                    fn = group['extra_processing'][name]
                    logger.debug(f'applying {fn} to {name}')
                    cube = fn(cube)

                if group.get('interpolate_model_levels_to_pressure', False):
                    cube = model_level_to_pressure(cube, p, z)

                da = cube_to_da(cube)
                zoom = self.config['max_zoom']
                da_hp = da_to_healpix(da, zoom, self.config['regrid_method'], group['name_map'], self.drop_vars,
                                      add_cyclic,
                                      regional, regional_chunks=chunks[-1])
                da_to_zarr(da_hp, self.config['zarr_store_url_tpl'], group_name, group, zoom, self.config['regional'])

    def coarsen_healpix_region(self, task):
        dim = task['dim']
        tgt_zoom = task['tgt_zoom']
        src_zoom = tgt_zoom + 1

        freqs = {
            '2d': 'PT1H',
            '3d': 'PT3H',
        }
        rel_url_tpl = self.config['zarr_store_url_tpl'][5:]  # chop off 's3://'
        freq = freqs[dim]
        urls = {
            z: rel_url_tpl.format(freq=freq, zoom=z)
            for z in range(11)
        }
        jasmin_s3 = s3fs.S3FileSystem(
            anon=False,
            secret=s3cfg['secret_key'],
            key=s3cfg['access_key'],
            client_kwargs={'endpoint_url': 'http://hackathon-o.s3.jc.rl.ac.uk'}
        )

        src_store = s3fs.S3Map(root=urls[src_zoom], s3=jasmin_s3, check=False)
        tgt_store = s3fs.S3Map(root=urls[tgt_zoom], s3=jasmin_s3, check=False)

        chunks = self.config['groups'][dim]['chunks']
        zarr_chunks = {'time': chunks[tgt_zoom][0], 'cell': -1}
        src_ds = xr.open_zarr(src_store, chunks=zarr_chunks)
        regional = self.config['regional']

        cluster = LocalCluster()  # Fully-featured local Dask cluster
        client = cluster.get_client()
        logger.debug(cluster)
        logger.debug(client)

        for subtask in task['tgt_times']:
            subtask_log = StringIO()
            logger_id = logger.add(subtask_log)

            start_idx = subtask['start_idx']
            end_idx = subtask['end_idx']
            donepath = Path(subtask['donepath'])
            logger.debug((start_idx, end_idx))
            coarsen_healpix_zarr_region(src_ds, tgt_store, tgt_zoom, dim, start_idx, end_idx, chunks, regional)
            donepath.parent.mkdir(parents=True, exist_ok=True)
            logger.trace(f'completed subtask {subtask}')
            logger.info(f'writing donepath: {donepath}')
            donepath.write_text(subtask_log.getvalue())
            logger.remove(logger_id)

        logger.info('completed')


def slurm_run(tasks, array_index):
    start = timer()
    task = tasks[array_index]
    logger.debug(task)
    proc = UMProcessTasks(processing_config[task['config_key']])
    if task['task_type'] == 'regrid':
        proc.regrid(task)
    elif task['task_type'] == 'create_empty_zarr_stores':
        proc.create_empty_zarr_stores(task)
    elif task['task_type'] == 'coarsen':
        proc.coarsen_healpix_region(task)
    else:
        raise Exception(f'unknown task type {task["task_type"]}')

    end = timer()
    logger.info(f'Completed in: {end - start:.2f}s')
    if task.get('donepath', ''):
        Path(task['donepath']).write_text(proc.debug_log.getvalue())

    return proc


def add_orog_land_sea(config_key):
    config = processing_config[config_key]
    basedir = config['basedir']
    max_zoom = config['max_zoom']
    add_cyclic = config.get('add_cyclic', True)
    regional = config.get('regional', False)
    zarr_store_url_tpl = config['zarr_store_url_tpl']

    cubes = iris.load(basedir / f'field.pp/apa.pp/{config_key}.apa_20200120T00.pp')
    land = xr.DataArray.from_iris(cubes.extract_cube('land_binary_mask'))
    orog = xr.DataArray.from_iris(cubes.extract_cube('surface_altitude'))

    # TODO: hardcoded path
    weights_path = (Path('/gws/nopw/j04/hrcm/mmuetz/weights/') /
                    weights_filename(land, config['max_zoom'], 'longitude',
                                     'latitude', add_cyclic, regional))
    assert weights_path.exists(), f'{weights_path} does not exist'
    regridder = LatLon2HealpixRegridder(weights_path=weights_path, zoom_level=max_zoom, add_cyclic=add_cyclic,
                                        regional=regional)
    hpland = regridder.regrid(land, 'longitude', 'latitude')
    hporog = regridder.regrid(orog, 'longitude', 'latitude')
    hpland.attrs['long_name'] = 'land_area_fraction'
    hporog.attrs['long_name'] = 'surface_altitude'

    for zoom in range(max_zoom, -1, -1):
        jasmin_s3 = s3fs.S3FileSystem(
            anon=False,
            secret=s3cfg['secret_key'],
            key=s3cfg['access_key'],
            client_kwargs={'endpoint_url': 'http://hackathon-o.s3.jc.rl.ac.uk'}
        )
        if zoom != max_zoom:
            assert regional == False, 'will not work with regional data yet'
            hpland = hpland.coarsen(cell=4).mean()
            hporog = hporog.coarsen(cell=4).mean()
            hpland['cell'] = np.arange(len(hpland.cell))
            hporog['cell'] = np.arange(len(hporog.cell))

        crs = xr.DataArray(
            name="crs",
            attrs={
                "grid_mapping_name": "healpix",
                "healpix_nside": 2 ** zoom,
                "healpix_order": "nest",
            },
        )

        freq = 'static'
        store_url = zarr_store_url_tpl.format(freq=freq, zoom=zoom)
        logger.info(f'Writing z{zoom} orog/land-sea to {freq}: {store_url}')
        zarr_store = s3fs.S3Map(
            root=store_url,
            s3=jasmin_s3, check=False)

        # TODO: add to existing zarr stores!
        ds_static = xr.Dataset()
        ds_static['orog'] = hporog.copy().assign_coords(crs=crs)
        ds_static['sftlf'] = hpland.copy().assign_coords(crs=crs)
        print(ds_static)
        ds_static.to_zarr(zarr_store, mode='a')
        # for freq in ['PT1H', 'PT3H']:
        #     store_url = zarr_store_url_tpl.format(freq=freq, zoom=zoom)
        #     logger.info(f'Writing z{zoom} orog/land-sea to {freq}: {store_url}')
        #     zarr_store = s3fs.S3Map(
        #         root=store_url,
        #         s3=jasmin_s3, check=False)
        #
        #     ds_static = xr.Dataset()
        #     ds_static['orog'] = hporog.copy().assign_coords(crs=crs)
        #     ds_static['sftlf'] = hpland.copy().assign_coords(crs=crs)
        #     print(ds_static)
        #     ds_static.to_zarr(zarr_store, mode='a')


if __name__ == '__main__':
    logger.remove()
    custom_fmt = ("<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                  "<level>{level: <8}</level> | "
                  "<blue>[{process.id}]</blue><cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    logger.add(sys.stderr, level="TRACE", format=custom_fmt, colorize=not sys.argv[1].startswith('slurm'))
    filepath = Path(__file__)
    logger.debug('{} last edited: {:%Y-%m-%d %H:%M:%S}'.format(filepath.name,
                                                               dt.datetime.fromtimestamp(filepath.stat().st_mtime)))

    logger.debug(' '.join(sys.argv))

    if sys.argv[1] == 'slurm':
        tasks_path = sys.argv[2]
        logger.info(tasks_path)
        with Path(tasks_path).open('r') as f:
            tasks = json.load(f)

        slurm_run(tasks, int(sys.argv[3]))
    elif sys.argv[1] == 'add_orog_land_sea':
        add_orog_land_sea(sys.argv[2])
    else:
        raise Exception(f'Unknown command {sys.argv[1]}')
