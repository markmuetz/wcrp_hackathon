import pickle
from pathlib import Path
import healpix as hp
import numpy as np
import xarray as xr

import easygems.healpix as egh
import easygems.remap as egr
from distributed import LocalCluster


def main(inpaths):
    inpaths = inpaths[:5]
    ds_cache = Path(f'.cache/ir_imerg_mfdataset_{len(inpaths)}.pkl')
    if not ds_cache.exists():
        ds = xr.open_mfdataset(inpaths).isel(time=slice(None, None, 2))
        ds_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(ds_cache, 'wb') as f:
            pickle.dump(ds, f)
    else:
        with open(ds_cache, 'rb') as f:
            ds = pickle.load(f)
    zoom = 9
    nside = 2**zoom
    npix = 12 * 4**zoom
    weights_cache = Path('.cache/weights.nc')
    if not weights_cache.exists():
        hp_lon, hp_lat = hp.pix2ang(nside=nside, ipix=np.arange(npix), lonlat=True, nest=True)
        hp_lon = (hp_lon + 180) % 360 - 180  # [-180, 180)
        lon, lat = np.meshgrid(ds.lon, ds.lat)
        lon = lon.flatten()
        lat = lat.flatten()
        lon_periodic = np.hstack((lon - 360, lon, lon + 360))
        lat_periodic = np.hstack((lat, lat, lat))
        weights = egr.compute_weights_delaunay(
            points=(lon_periodic, lat_periodic),
            xi=(hp_lon, hp_lat)
        )
        weights = weights.assign(src_idx=weights.src_idx % ds.lat.size)
        weights.to_netcdf(weights_cache)
    else:
        weights = xr.load_dataset(weights_cache)

    cluster = LocalCluster()  # Fully-featured local Dask cluster
    client = cluster.get_client()
    print(cluster)
    print(client)

    ds_remap = xr.apply_ufunc(
        egr.apply_weights,
        ds,
        kwargs=weights,
        keep_attrs=True,
        input_core_dims=[["lon", "lat"]],
        output_core_dims=[["cell"]],
        output_dtypes=["f4"],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={
            "allow_rechunk": True,
            "output_sizes": {"cell": npix},
        },
    )
    print(ds_remap)
    ds_remap.to_netcdf('ir_imerg_test.nc')




if __name__ == '__main__':
    paths = sorted(Path('/gws/nopw/j04/hrcm/mmuetz/obs/IR_IMERG_Combined_V07B/').glob('*/*.nc'))
    main(paths)
