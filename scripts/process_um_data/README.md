# High-level overview of UM processing code

* All code written as part of the WCRP hackathon 2025 by Mark Muetzelfeldt
* **Brief:** convert 3 global and 4 regional UM simulations from lat/lon .pp to healpix .zarr in JASMIN object store.
* **Considerations:** 
  * simulations only finished being transferred to JASMIN days before hackathon - had to be able to run incrementally on available data
  * needs to be robust against crashes at various points (mechanisms to handle and easy to rerun using `.done` files): 
    * loading UM data from JASMIN FS/GWS
    * writing data to JASMIN obj store
  * no writing intermediate data (efficiency and storage) - straight from .pp to healpix/zarr in obj store
  * handle regional data
* Extra dependency on `loguru` for nice logs.

## Workflow

* use simple `.done` files to determine if a particular job has already completed
* mainly use single CPU SLURM jobs to do regridding
  * (apart from coarsening which benefitted from more cores using dask)
* wrap code for checking `.done` files and launching remaining jobs into `um_slurm_control.py`
  * use SLURM array jobs
  * write a `.json` file with a list of jobs to be run, and a `.sh` SLURM script
  * each SLURM array index selects one job in the `.json` file, and job dispatched to correct code path.
  * when a job finished, a `.done` file is written
* handle dependency between creating empty zarr store and subsequent tasks 
* coarsening must be run **once all of the highest zoom level has been regridded**

## Structure

* `processing_config.py`: simple dict-based config for different sims
  * why not `.yaml`? Doing as native Python lets you easily reuse sections of config by defining a variable
* `um_slurm_control.py`: entry point. Scans existing files then compares jobs to be done with already completed. Launches SLURM jobs.
* `um_process_tasks.py`: entry point for SLURM jobs. Dispatches to correct code path based on job. Sets up empty/template zarr stores based on UM .pp files.
* `latlon_to_healpix.py`: regridder that handles all UM data (regional/N2560/N1280)
* `healpix_coarsen.py`: coarsen global/regional data

## Processing steps (`um_proces_tasks.py`)

* If first timestep:
  * create empty zarr store by using sim config in `processing_config.py` and combining with info in .pp files.
  * do for all zooms from max to 0.
  * uses `ds.to_zarr(store, compute=False)`
* Load UM .pp file - each contains 12 hours of data
  * hours are not aligned! 1 file can contain e.g. 01:00 - 12:00 and 00:00 - 11:00. This means I have to write to a single time chunk at the highest zoom I think.
  * For each variable:
    * (interpolate in vertical if needed)
    * extract `xr.DataArray` from `iris.cube`, and remap names
    * convert from lat/lon to healpix
    * save variable to zarr store
      * uses `ds.to_zarr(store, region=regslice)`
* Once all variables written at highest zoom:
  * coarsen each variable to next lowest zoom, using time chunk at target zoom as unit of work
  * and so on

## Details

* **regional data**
  * store efficiently by only saving active chunks
  * store weights at all lower zooms for consistent domain mean
  * coarsen by converting to global, coarsening, then converting to (probably new) active-chunks domain at lower zoom
* time coords in UM were inconsistent. Some vars at e.g. 12:02, hourly means at 12:30. Shift the former to 12:00, the latter to 13:00.
* different vars on different grids (Arakawa A vs C) - interp appropriately
* 5 fields on 60 model levels - interp to pressure levels
* some vars defined with flipped signs to standard - flip
* UM precip was stored as rainfall and snow/solid - combine (ongoing)
* N1280 GAL9 has 2 precips: stratiform and convective - combine (ongoing)
