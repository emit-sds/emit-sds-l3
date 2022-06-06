<h1 align="center">
emit-sds-l3
</h1>

_NOTE - See the **develop** branch - set as default - for the latest updates._

Welcome to the EMIT Level 3 science data system repository.  To understand how this repository is linked to the rest of the emit-sds repositories, please see [the repository guide](https://github.com/emit-sds/emit-main/wiki/Repository-Guide).

This repository executes two different actions - mosaic generation and data aggregation to an ESM-scale product.  SpectralUnmixing was previously a part of this repository, but has been [relocated](https://github.com/emit-sds/SpectralUnmixing).



Example call to construct a mosaic-glt (3-band image denoting x-offset, y-offset, and file number):
```
    julia build_mosaic_glt.jl OUTPUT_MOSAIC_GLT IGM_FILE_LIST TARGET_RESOLUTION
```

Additional arguments:

```
    --criteria_mode, type = String, default = "distance", help = "Band-ordering criteria mode.  Options are min or max (require criteria file), or distance (uses closest point)
    --criteria_band, type = Int64, default = 1, help = "band of criteria file to use"
    --criteria_file_list, type = String, help = "file(s) to be used for criteria"
    --target_extent_ul_lr, type = Float64, nargs=4, help = "extent to build the mosaic of"
    --mosaic, type = Int32, default=1, help = "treat as a mosaic"
    --output_epsg, type = Int32, default=4326, help = "epsg to write to destination"
    --log_file, type = String, default = nothing, help = "log file to write to"
```


\
\
Example call to apply a mosaic-glt to create a mosaic:

```
    python apply_glt.py GLT_FILE RAWSPACE_FILES OUTPUT_FILENAME 
```

Additional arguments:

```
    -band_numbers, nargs='+', type=int, default=-1, help='list of 0-based band numbers, or -1 for all'
    -n_cores, type=int, default=-1
    -log_file, type=str, default=None
    -log_level, type=str, default='INFO'
    -run_with_missing_files, type=int, default=0, choices=[0,1]
    -ip_head, type=str
    -redis_password, type=str
    -one_based_glt, type=int, choices=[0,1], default=0
    -mosaic, type=int, choices=[0,1], default=0
```
