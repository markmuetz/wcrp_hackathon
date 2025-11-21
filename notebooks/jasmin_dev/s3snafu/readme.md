# S3 SNAFU

I accidentally deleted some data (7/5/2025) from the glm.n2560_RAL3p3 2D dataset.
    
* I was trying to add orog/sftlf to an existing dataset 5km RAL3
    * Instead of using mode='a' (append), I experimented with mode='w' (write), because mode='a' was taking prohibitively long
    * I expected that this would only add the new variables I was writing, not delete existing variables. Seems like this was wrong.
    * This deleted approx. 4 months of 2D data for certain fields. (2020-09-05T00:00  to 2020-12-08T12:00)
    * The command failed for 3D data - stopped any data loss.
* Diagnosis
    * Checked in logs for missing data/all nan flags
    * Ran a s3cmd ls scan of affected zarr store dirs (checked 3D as well)
    * Mapped missing indices to times
* Fix
    * Regrid 4 months (1 Sept 2020 - 15 Dec 2020) of 24x 2D fields from N2560 to healpix level 10
    * Coarsen healpix level 10 to levels 9-0 (for all times)
* Checks
    * Check that missing data were filled in at successive zoom levels
    * Checked date at start of Sept (2020-09-05 01:00)
    * Scanned s3 zarr store directory using s3cmd ls (ongoing)