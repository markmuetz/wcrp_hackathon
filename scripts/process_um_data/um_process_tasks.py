"""Main script that handles all steps of the processing of UM lat/lon .pp data to healpix in a zarr store.

1. Create empty zarr stores.
2. Populate the highest zoom, region at a time.
3. Once the highest zoom is complete, coarsen to the lower zooms.

Here, a "region" refers to a subset of a given variable:
https://docs.xarray.dev/en/stable/user-guide/io.html#distributed-writes
i.e. if the variable is pr (precip), and it has dimensions (24 * 400, 12 * 4**10) (400 days hourly data at zoom=10)
then the region might be [36:48, :] (i.e. the second half of the second day).

Written by mark.muetzelfeldt@reading.ac.uk for the WCRP hackathon 2025 (UK node).
"""
import asyncio
import datetime as dt
import json
import sys
from timeit import default_timer as timer
from collections import defaultdict
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
import xarray as xr
from loguru import logger

from um_processing_config import processing_config, shared_metadata
from cube_to_da_mapping import DataArrayExtractor
from healpix_coarsen import coarsen_healpix_zarr_region, async_da_to_zarr_with_retries
from latlon_to_healpix import LatLon2HealpixRegridder, gen_weights, get_limited_healpix
from util import async_da_to_zarr_with_retries

# Super simple .s3cfg parser - must be in home directory.
s3cfg = dict([l.split(' = ') for l in (Path.home() / '.s3cfg').read_text().split('\n') if l])
iris.FUTURE.date_microseconds = True


def get_jasmin_s3():
    """These objects seem to go stale after a period of time - recreate when needed"""
    return s3fs.S3FileSystem(
        anon=False,
        secret=s3cfg['secret_key'],
        key=s3cfg['access_key'],
        client_kwargs={'endpoint_url': 'http://hackathon-o.s3.jc.rl.ac.uk'}
    )


def get_crs(zoom):
    return xr.DataArray(
        name="crs",
        attrs={
            "grid_mapping_name": "healpix",
            "healpix_nside": 2 ** zoom,
            "healpix_order": "nest",
        },
    )


def regrid_da_to_healpix(da, zoom, short_name, long_name, weightsdir, drop_vars, add_cyclic=True,
                         regional=False, regional_chunks=None):
    """Create a regridded DataArray at a given zoom level from the original lat/lon DataArray."""
    um_name = da.name

    lonname = [c for c in da.coords if c.startswith('longitude')][0]
    latname = [c for c in da.coords if c.startswith('latitude')][0]
    weights_path = weightsdir / weights_filename(da, zoom, lonname, latname, add_cyclic, regional)
    logger.trace(f'  - using weights: {weights_path}')
    regridder = LatLon2HealpixRegridder(weights_path=weights_path, method='easygems_delaunay', zoom_level=zoom,
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


def healpix_da_to_zarr(da, url, group_name, regional, nan_checks=False):
    """Write a healpix DataArray to the store defined by the URL."""
    name = da.name
    logger.info(f'{name} to zarr => {url}')

    zarr_store = s3fs.S3Map(
        root=url,
        s3=get_jasmin_s3(), check=False)

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

    def _initialize_metadata(self, regional):
        """Initialize metadata dict with default and config values"""
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
        return metadata

    @staticmethod
    def _process_group_cubes(group_name, group, cubes):
        """Process cubes for a group and convert to DataArrays"""
        name_map = group['name_map']
        logger.info(f'Creating {group_name}')
        group_cubes = cubes.extract(group['constraint'])
        logger.info(f'Found {len(group_cubes)} cubes for {group_name}')

        list_da = []
        extractor = DataArrayExtractor(None, None)
        for key, map_item in name_map.items():
            short_name, long_name = key
            cube = extractor.extract_cube(map_item, group_cubes, combine_cubes=False)
            da = xr.DataArray.from_iris(cube).rename(short_name)
            da.attrs['long_name'] = long_name
            list_da.append(da)

        return list_da

    def _gen_orog_land_sea(self):
        """The format of orog and land_sea_mask are slightly different - handle them here.

        generate orog_land_sea (static data) at each zoom level and return as dict."""
        config = self.config
        basedir = config['basedir']
        max_zoom = config['max_zoom']
        add_cyclic = config.get('add_cyclic', True)
        regional = config.get('regional', False)
        zarr_store_url_tpl = config['zarr_store_url_tpl']

        cubes = iris.load(basedir / f'field.pp/apa.pp/{config["name"]}.apa_20200120T00.pp')
        land = xr.DataArray.from_iris(cubes.extract_cube('land_binary_mask'))
        orog = xr.DataArray.from_iris(cubes.extract_cube('surface_altitude'))

        weights_path = (config['weightsdir'] /
                        weights_filename(land, config['max_zoom'],
                                         'longitude', 'latitude', add_cyclic, regional))
        assert weights_path.exists(), f'{weights_path} does not exist'
        regridder = LatLon2HealpixRegridder(weights_path=weights_path, zoom_level=max_zoom, add_cyclic=add_cyclic,
                                            regional=regional)
        hpland = regridder.regrid(land, 'longitude', 'latitude')
        hporog = regridder.regrid(orog, 'longitude', 'latitude')
        hpland.attrs['long_name'] = 'land_area_fraction'
        hporog.attrs['long_name'] = 'surface_altitude'

        orog_land_sea = {}

        for zoom in range(max_zoom, -1, -1):
            if zoom != max_zoom:
                # TODO: Get working for regional.
                assert regional == False, 'will not work with regional data yet'
                hpland = hpland.coarsen(cell=4).mean()
                hporog = hporog.coarsen(cell=4).mean()
                hpland['cell'] = np.arange(len(hpland.cell))
                hporog['cell'] = np.arange(len(hporog.cell))
            ds_static = xr.Dataset()
            ds_static['orog'] = hporog.copy().assign_coords(crs=get_crs(zoom))
            ds_static['sftlf'] = hpland.copy().assign_coords(crs=get_crs(zoom))
            orog_land_sea[zoom] = ds_static

        return orog_land_sea

    def _create_dataarray_template(self, group, da, chunks, zoom, npix, zarr_time, zarr_time_name):
        """Use the provided info to create a *dummy* dataarray.

        The input da is used to take the metadata etc.
        The dummy/template da created will have the correct dimensions.
        """
        logger.info(f'- creating da for {da.name}')

        # coords not always nicely named, but always begin with the obvious thing.
        timename = [c for c in da.coords if c.startswith('time')][0]
        lonname = [c for c in da.coords if c.startswith('longitude')][0]
        latname = [c for c in da.coords if c.startswith('latitude')][0]
        add_cyclic = self.config.get('add_cyclic', True)
        regional = self.config.get('regional', False)

        logger.trace((zoom, self.config['max_zoom']))
        if zoom == self.config['max_zoom']:
            # Gen weights path for regridding (only needed at max zoom) if it doesn't already exist.
            weights_path = self.config['weightsdir'] / weights_filename(da, zoom, lonname, latname, add_cyclic, regional)
            if not weights_path.exists():
                logger.info(f'No weights for {da.name}, generating')
                # chunks[-1] selects the spatial chunk.
                gen_weights(da, weights_path=weights_path, zoom=zoom, lonname=lonname, latname=latname,
                            add_cyclic=add_cyclic, regional=regional, regional_chunks=chunks[-1])
        if regional:
            minlon, maxlon = da[lonname].values[[0, -1]]
            minlat, maxlat = da[latname].values[[0, -1]]
            extent = [minlon, maxlon, minlat, maxlat]
            # Handles the magic of getting only active areas at a given chunking for regional data.
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

        # This is the idiom for using xarray to create a zarr entry with the correct dimensions but no actual data.
        # https://docs.xarray.dev/en/stable/user-guide/io.html#distributed-writes
        dummies = dask.array.zeros(shape, dtype=np.float32, chunks=chunks)
        da_tpl = xr.DataArray(dummies, dims=dims, coords=coords, name=da.name, attrs=da.attrs)
        da_tpl = da_tpl.rename(**{timename: zarr_time_name})
        da_tpl.attrs['UM_name'] = da.name
        da_tpl.attrs['grid_mapping'] = 'healpix_nested'
        da_tpl.attrs['healpix_zoom'] = zoom
        return da_tpl

    def create_empty_zarr_stores(self, task):
        """Use information in metadata to create empty zarr stores that contains all variables

        One zarr store per zoom, each contains all variables and their metadata/dimensions.
        The fundamental idea is to use xarray distributed writes:
        https://docs.xarray.dev/en/stable/user-guide/io.html#distributed-writes

        `self._create_dataarray_template` creates the dummy variables
        `self._write_zarr_store` writes the datasets to the storage backend.
        """
        inpaths = task['inpaths']
        cubes = iris.load(inpaths)
        logger.trace(cubes)

        regional = self.config.get('regional', False)
        metadata = self._initialize_metadata(regional)

        # dict of dataarrays grouped by group_name
        grouped_da = {
            group_name: self._process_group_cubes(group_name, group, cubes)
            for group_name, group in self.config['groups'].items()
        }

        if not regional:
            # TODO: handle regional.
            orog_land_sea = self._gen_orog_land_sea()

        # Create one zarr store with metadata for all variables for each zoom.
        for zoom in range(self.config['max_zoom'], -1, -1):
            npix = 12 * 4 ** zoom
            ds_tpls = defaultdict(xr.Dataset)

            for group_name, group in self.config['groups'].items():
                time = group['time']
                zarr_time_name = 'time'
                zarr_time = time
                zarr_store_name = group['zarr_store']
                chunks = group['chunks'][zoom]

                for da in grouped_da[group_name]:
                    da_tpl = self._create_dataarray_template(
                        group, da, chunks, zoom, npix, zarr_time, zarr_time_name
                    )
                    if regional:
                        if metadata['bounds'] is None:
                            metadata['bounds'] = get_regional_bounds(da)
                            logger.debug('bounds={}'.format(metadata['bounds']))
                    ds_tpls[zarr_store_name][da_tpl.name] = da_tpl
                    if not regional:
                        ds_tpls[zarr_store_name]['orog'] = orog_land_sea[zoom].orog
                        ds_tpls[zarr_store_name]['sftlf'] = orog_land_sea[zoom].sftlf

                if regional and zoom != self.config['max_zoom']:
                    coords = {n: c for n, c in da_tpl.coords.items() if not n == 'time'}
                    dummies = dask.array.zeros(da_tpl.shape[1:], dtype=np.float32, chunks=chunks[1:])
                    ds_tpls[zarr_store_name]['weights'] = xr.DataArray(dummies, name='weights', coords=coords)

            for zarr_store_name, ds_tpl in ds_tpls.items():
                self._write_zarr_store(ds_tpl, zarr_store_name, zoom, metadata, task)

    def _write_zarr_store(self, ds_tpl, zarr_store_name, zoom, metadata, task):
        """Write a zarr store for the dataset template"""
        ds_tpl = ds_tpl.assign_coords(crs=get_crs(zoom))
        ds_tpl.attrs.update(metadata)

        logger.info(f'Saving {task["config_key"]} zoom={zoom}')
        store_url = self.config['zarr_store_url_tpl'].format(freq=zarr_store_name, zoom=zoom)

        zarr_store = s3fs.S3Map(
            root=store_url,
            s3=get_jasmin_s3(), check=False)
        logger.debug(store_url)
        logger.debug(ds_tpl)
        ds_tpl.to_zarr(zarr_store, mode='w', compute=False)

    def regrid(self, task):
        """Regrid all variables from lat/lon to healpix.

        Reads in data from UM .pp files, and outputs to the correct region of an already created zarr store.
        The task has a list of .pp files to read. These will lbe the .pp files for a given date and all streams.
        This is streams a-d currently. These are loaded as one, then cubes are extracted from this superset.

        Uses the config for this class to define groups, and mappings from (multiple) cubes to a single output variable.
        A "group" is a distinct set of output variables, but multiple groups can end up in the same zarr store.
        As of writing, the groups are: 2d, 3d, 3d_ml (3d on model levels, these need to be vertically interp'd and end
        up in the 3d zarr store).

        * Loops over each group
        * For each output variable (which may take data from multiple cubes):
           * extract each variable and map names, attrs. Apply extra processing if nec.
           * do the regridding
           * save to zarr store
        """
        inpaths = task['inpaths']

        logger.info('loading cubes')
        logger.trace(inpaths)
        cubes = iris.load(inpaths)

        add_cyclic = self.config.get('add_cyclic', True)
        regional = self.config.get('regional', False)

        # These are needed by any field which needs 3D interp.
        p = cubes.extract_cube('air_pressure')
        z = cubes.extract_cube('geopotential_height')
        extractor = DataArrayExtractor(p, z)

        for group_name, group in self.groups.items():
            logger.info(f'processing group {group_name}')
            group_constraint = group['constraint']
            name_map = group['name_map']
            chunks = group['chunks'][self.config['max_zoom']]
            group_cubes = cubes.extract(group_constraint)

            # Handle each entry in the mapping from UM cubes to dataarrays.
            # There might be multiple cubes required for a single dataarray due to the need to e.g. combine snow/rain.
            # extractor will handle these.
            for i, key in enumerate(name_map):
                short_name, long_name = key
                msg = f'{(i + 1)}/{len(name_map)}: regridding {short_name}'
                logger.info('=' * len(msg))
                logger.info(msg)
                logger.info('=' * len(msg))
                map_item = name_map[key]
                da = extractor.extract_da(map_item, group_cubes)

                zoom = self.config['max_zoom']
                # Do the regridding.
                da_hp = regrid_da_to_healpix(da, zoom, short_name, long_name,
                                             self.config['weightsdir'], self.drop_vars,
                                             add_cyclic,
                                             regional, regional_chunks=chunks[-1])
                # Write this variable to the zarr store.
                zarr_store_name = group['zarr_store']
                url = self.config['zarr_store_url_tpl'].format(freq=zarr_store_name, zoom=zoom)
                healpix_da_to_zarr(da_hp, url, group_name, self.config['regional'])

    def coarsen_healpix_region(self, task):
        """Coarsen the regions from source to target zooms, as defined by the task."""
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
        jasmin_s3 = get_jasmin_s3()

        src_store = s3fs.S3Map(root=urls[src_zoom], s3=jasmin_s3, check=False)
        tgt_store = s3fs.S3Map(root=urls[tgt_zoom], s3=jasmin_s3, check=False)

        chunks = self.config['groups'][dim]['chunks']
        zarr_chunks = {'time': chunks[tgt_zoom][0], 'cell': -1}
        src_ds = xr.open_zarr(src_store, chunks=zarr_chunks)
        regional = self.config['regional']

        # This will create a cluster with its specifications taken from the current machine.
        # i.e. you can request a SLURM node with lots of CPUs etc and the cluster will reflect this.
        cluster = LocalCluster()
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
    """Dispatch the task to the processing method."""
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
    # If task has a donepath - write out the debug log.
    # Existence of a donepath will stop this task being run again.
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

    logger.debug(' '.join(sys.argv))

    if sys.argv[1] == 'slurm':
        tasks_path = sys.argv[2]
        logger.info(tasks_path)
        with Path(tasks_path).open('r') as f:
            tasks = json.load(f)

        slurm_run(tasks, int(sys.argv[3]))
    else:
        raise Exception(f'Unknown command {sys.argv[1]}')
