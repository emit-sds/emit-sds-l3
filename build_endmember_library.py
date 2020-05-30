"""
This code is designed to construct an endmember library from in situ spectra, in order to make the MESMA processing chain
fully transparent.  This is intentially written as a script, without inputs and outputs, in order to document the
specific endmember generation process.  Steps:

1) Filter in situ spectra to what is needed.
2) Use the Endmember average RMSE (EAR) tool from MESMA to select representative spectra.
3) Combine the selected spectra together into a single, formatted output file.

Spectral will be convolved to match the

Written by: Philip G. Brodrick
"""

import argparse
import numpy as np
import pandas as pd
import os
import logging
from mesma.square_array import SquareArray
from mesma.ear_masa_cob import EarMasaCob

bad_wv_regions = [[0,440],[1310,1490],[1770,2050],[2440,2880]]

def get_good_bands_mask(wavelengths, wavelength_pairs):
    good_bands = np.ones(len(wavelengths)).astype(bool)

    for wvp in wavelength_pairs:
        wvl_diff = wavelengths - wvp[0]
        wvl_diff[wvl_diff < 0] = np.nanmax(wvl_diff)
        lower_index = np.nanargmin(wvl_diff)

        wvl_diff = wvp[1] - wavelengths
        wvl_diff[wvl_diff < 0] = np.nanmax(wvl_diff)
        upper_index = np.nanargmin(wvl_diff)
        good_bands[lower_index:upper_index+1] = False

    return good_bands


class SpectralLibrary():

    def __init__(self, file_name, class_header_name, spectral_header, wavelengths, class_valid_keys=None,scale_factor=None):

        self.file_name = file_name
        self.class_header_name = class_header_name
        self.spectral_header = spectral_header
        self.wavelengths = wavelengths
        self.scale_factor = scale_factor

        self.spectra = None
        self.classes = None
        self.good_bands = None

        if (class_valid_keys is not None):
            self.class_valid_keys = [x.lower() for x in class_valid_keys]
        else:
            self.class_valid_keys = None


    def load_data(self):
        df = pd.read_csv(self.file_name)

        self.spectra = np.array(df[self.spectral_header])
        self.classes = np.array(df[self.class_header_name]).tolist()

        self.classes = np.array([x.lower() for x in self.classes])

        if (self.class_valid_keys is None):
            self.class_valid_keys = np.unique(self.classes).tolist()

    def filter_by_class(self):
        if self.class_valid_keys is None:
            logging.info('No class valid keys provided, no filtering occuring')
            return

        valid_classes = np.zeros(self.spectra.shape[0]).astype(bool)
        for cla in self.class_valid_keys:
            valid_classes[self.classes == cla] = True

        self.spectra = self.spectra[valid_classes, :]
        self.classes = self.classes[valid_classes]

    def remove_wavelength_region_inplace(self, wavelength_pairs, set_as_nans=False):
        good_bands = get_good_bands_mask(self.wavelengths, wavelength_pairs)
        self.good_bands = good_bands
        if set_as_nans:
            self.spectra[:,np.logical_not(good_bands)] = np.nan
            self.wavelengths[np.logical_not(good_bands)] = np.nan
        else:
            self.spectra = self.spectra[:,good_bands]
            self.wavelengths = self.wavelengths[good_bands]


    def interpolate_library_to_new_wavelengths(self, new_wavelengths):
        old_spectra = self.spectra.copy()

        self.spectra = np.zeros((self.spectra.shape[0],len(new_wavelengths)))
        for _s in range(old_spectra.shape[0]):
            self.spectra[_s,:] = np.interp(new_wavelengths, self.wavelengths, old_spectra[_s,:])
        self.wavelengths = new_wavelengths

    def scale_library(self, scaling_factor=None):
        if scaling_factor is None:
            self.spectra /= self.scale_factor
        else:
            self.spectra /= scaling_factor

    def brightness_normalize(self):
        self.spectra = self.spectra / np.sqrt(np.nanmean(np.power(self.spectra,2),axis=1))[:,np.newaxis]


def main():
    parser = argparse.ArgumentParser(description='Execute MESMA in parallel over a BIL line')
    parser.add_argument('-spectral_data_files',type=str,default=['data/urbanspectraandmeta.csv'],nargs='+')
    parser.add_argument('-class_names',type=str,default=['Level_2'],nargs='+')
    args = parser.parse_args()

    if len(args.spectral_data) != len(args.class_name):
        raise AttributeError('Length of input spectral_data must equal input class_name')

    spectra_per_class = 6

    # Create list of all spectral libraries to use
    libraries = []
    for datafile, cla in zip(args.spectral_data_files,args.class_names):
        libraries.append(SpectralLibrary(datafile, cla,
                                         np.arange(350, 2499, 2).astype(int).astype(str),
                                         np.arange(350, 2499, 2).astype(np.float32),
                                         scale_factor=10000.))


    # Open and filter spectral libraries
    for _f in range(len(libraries)):
        libraries[_f].load_data()
        libraries[_f].filter_by_class()
        libraries[_f].scale_library()
        libraries[_f].remove_wavelength_region_inplace(bad_wv_regions,set_as_nans=True)

    # Now run EAR on each library (independently), and assemble the desired number of classes
    ear_masa_cobs = []
    for _f in range(len(libraries)):

        square_output = SquareArray().execute(library=libraries[_f].spectra.T[libraries[_f].good_bands,:], constraints=(-9999, -9999, -9999), out_constraints=True, out_angle=True, out_rmse=True)

        emc = EarMasaCob().execute(spectral_angle_band=square_output['spectral angle'], rmse_band=square_output['rmse'],
                                   constraints_band=square_output['constraints'], class_list=libraries[_f].classes)

        ear_masa_cobs.append(emc)

        ear = emc[0]
        out_df = pd.read_csv(libraries[_f].file_name)
        out_df['EAR'] = emc[0]
        out_df['MASA'] = emc[1]
        out_df['InCOB'] = emc[2]
        out_df['OutCOB'] = emc[3]
        out_df['COBI'] = emc[4]

        good_data = np.zeros(libraries[_f].spectra.shape[0]).astype(bool)
        for cla in libraries[_f].class_valid_keys:
            subset = libraries[_f].classes == cla
            to_sort = ear.copy()
            to_sort[np.logical_not(subset)] = np.nan

            order = np.argsort(to_sort)

            good_data[order[:spectra_per_class]] = True

        out_df = out_df.loc[good_data,:]
        out_df.to_csv(os.path.splitext(libraries[_f].file_name)[0] + '_endmember.csv', index=False)




if __name__ == "__main__":
    main()





























