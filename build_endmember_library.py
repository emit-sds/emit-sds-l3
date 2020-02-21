"""
This code is designed to construct an endmember library from in situ spectra, in order to make the MESMA processing chain
fully transparent.  This is intentially written as a script, without inputs and outputs, in order to document the
specific endmember generation process.  Steps:

1) Filter in situ spectra to what is needed.
2) Use the Endmember average RMSE (EAR) tool from MESMA to select representative spectra.
3) Convolve spectra to the appropriate spectral resolution.
4) Combine the selected spectra together into a single, formatted output file.

Written by: Philip G. Brodrick
"""

import numpy as np
import pandas as pd
import sys

#sys.path.extend(['../vipertools-3.3.0/','../vipertools-3.3.0/vipertools'])

def remove_wavelength_regions(input_array, wavelengths, wavelength_pairs, set_as_nans=False):
    good_bands = np.ones(len(wavelengths)).astype(bool)

    for wvp in wavelength_pairs:
        wvl_diff = wavelengths - wvp[0]
        wvl_diff[wvl_diff < 0] = np.max(wvl_diff)
        lower_index = np.argmin(wvl_diff)

        wvl_diff = wvp[1] - wavelengths
        wvl_diff[wvl_diff < 0] = np.max(wvl_diff)
        upper_index = np.argmin(wvl_diff)
        good_bands[lower_index:upper_index+1] = False

    if set_as_nans:
        input_array[:,np.logical_not(good_bands)] = np.nan
        wavelengths[np.logical_not(good_bands)] = np.nan
    else:
        input_array = input_array[:,good_bands]
        wavelengths = wavelengths[good_bands]

    return input_array, wavelengths


class SpectralLibrary():

    def __init__(self, file_name, class_header_name, class_valid_keys, spectral_header, wavelengths):

        self.file_name = file_name
        self.class_header_name = class_header_name
        self.class_valid_keys = class_valid_keys
        self.spectral_header = spectral_header
        self.wavelengths = wavelengths

        self.spectra = None
        self.classes = None

    def load_data(self):
        df = pd.read_csv(self.file_name)

        self.spectra = np.array(df[self.spectral_header])
        self.classes = np.array(df[self.class_header_name])

    def filter_by_class(self):
        valid_classes = np.zeros(self.spectra.shape[0]).astype(bool)
        for cla in self.classes:
            valid_classes[self.classes == cla] = True

        self.spectra = self.spectra[valid_classes, :]
        self.classes = self.classes[valid_classes]

    def remove_wavelength_region_inplace(self, wavelength_pairs, set_as_nans=False):
        self.spectra, self.wavelengths = remove_wavelength_regions(self.spectra, self.wavelengths, wavelength_pairs,
                                                                   set_as_nans)

    def interpolate_library_to_new_wavelengths(self, new_wavelengths):
        old_spectra = self.spectra.copy()

        self.spectra = np.zeros((self.spectra.shape[0],len(new_wavelengths)))
        for _s in range(old_spectra.shape[0]):
            self.spectra[_s,:] = np.interp(new_wavelengths, self.wavelengths, old_spectra[_s,:])
        self.wavelengths = self.wavelengths

    def scale_library(self, scaling_factor):
        self.spectra /= scaling_factor





libraries = []
libraries.append(SpectralLibrary('data/urbanspectraandmeta.csv', 'Level_2',
                                  ['NPV', 'PV', 'SOIL'], np.arange(350, 2499, 2).astype(int).astype(str),
                                  np.arange(350, 2499, 2)))

## Open and filter spectral libraries
#for _f in range(len(libraries)):
#
#    libraries[_f].load_data()
#    libraries[_f].filter_by_class()
#
#
## Now run EAR on each library (independently)
#for _f in range(len(libraries)):
#
#    square = square_array.SquareArray()
#    square_output = square.execute(libraries[_f].spectra.T, constraints=(None, None, None), out_constraints=False)
#
#    rmse = square_output['rmse']
#
#
#    ear = np.zeros(libraries[_f].spectra.shape[0])
#    for cla in libraries[_f].classes:
#        inside_indices = np.where(libraries[_f].classes == cla)[0]
#        #ear[inside_indices[:, None], inside_indices[:,None]] = np.






