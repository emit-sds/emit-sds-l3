# emit-sds-3

Welcome to the EMIT Level 3 science data system repository.  To understand how this repository is linked to the rest of the emit-sds repositories, please see [the repository guide](https://github.jpl.nasa.gov/emit-sds/emit-main/wiki/Repository-Guide).

This repository does 2 different thigns.  It 1) runs MESMA on flightlines to estimate fractional cover, and 2) creates and applies mosaic GLTs.  The repo may be broken up enventually.

Using MESMA:

Applying mesma is done in two steps.  First, a user must build an endmember library from a set of spectra, e.g.:

:code
  python build_endmemeber_library.py -spectral_data_files LIBRARY -class_names CLASSNAME

LIBRARY is turned into a SpectralLibrary, and the code handles management between different wavelength ranges, but as of now the wavelength ranges are hardcoded in to the instantiation of each SpectralLibrary, matching the demos for the EMIT case.

Following the gneration of the endmember library (or if the user has one pre-build), you can execute mesma on a flightline using:

:code
  python apply_mesma.py REFLECTANCE_FILE ENDMEMBER_FILE ENDMEMBER_CLASS OUTPUT_FILE_BASE
  
which are:\\
REFLECTANCE_FILE - input apparent surface reflectance\\
ENDMEMBER_FILE - spectral endmember library, as built using build_endmember_library.py, or matching that format.\\
ENDMEMBER_CLASS - the class of the spectral endmember library to run MESMA on (in case there are multiple)\\
OUTPUT_FILE_BASE - base output file, multiple sub-files will be generated.

Additional outputs, including keys to run in parallel, are available.  MCMC implementations are ongoing.

