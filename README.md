# emit-sds-3

Welcome to the EMIT Level 3 science data system repository.  To understand how this repository is linked to the rest of the emit-sds repositories, please see [the repository guide](https://github.jpl.nasa.gov/emit-sds/emit-main/wiki/Repository-Guide).

This repository executes two different actions. First, it runs a spectral mixture analysis on flightlines to estimate fractional cover, and secon it merges multiple flightlines together and aggregates L2b outputs together with fractional cover to aggregated spectral abundance (an ESM-scale product).  The repo may be broken up enventually.

Spectral unmixture analysis (nominal run configuration outlined below):

```
julia -p 40 unmix.jl REFLECTANCE_FILE ENDMEMBER_LIBRARY ENDMEMBER_CLASS OUTPUT_FILE_BASE --mode sma-best --num_endmembers 10 --n_mc 50 --normalization brightness
```


where:<br>
REFLECTANCE_FILE - input apparent surface reflectance<br>
ENDMEMBER_FILE - spectral endmember library, as built using build_endmember_library.py, or matching that format<br>
ENDMEMBER_CLASS - the class of the spectral endmember library to run MESMA on (in case there are multiple)<br>
OUTPUT_FILE_BASE - base output file, multiple sub-files will be generated<br>

Additional outputs, including keys to run in parallel, are available.  MCMC implementations are ongoing.<br>

