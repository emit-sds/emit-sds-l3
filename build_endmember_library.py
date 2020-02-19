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






import argparse
import sys, os
import pandas as pd
import numpy as np



class Spectral_Library()

    def __init__(self, file_name, class_header_name, class_valid_keys, spectral_header):

        self.file_name = file_name
        self.class_header_name = class_header_name
        self.class_valid_keys = class_valid_keys
        self.spectral_header = spectral_header

        self.spectra = None
        self.classes = None

    def load_data():
        df = pd.read_csv(self.file_name)

        self.spectra = np.array(df[self.spectral_header])
        self.classes = np.array(df[self.class_header_name])

    def filter_by_class():
        valid_classes = np.zeros(self.spectra.shape[0]).astype(bool)
        for cla in self.classes:
            valid_classes[self.classes == cla] = True

        self.spectra = self.spectra[valid_classes,:]
        self.classes = self.classes[valid_classes]


    
libraries = []
libraries.append(Spectral_Library('data/urbanspectraandmeta.csv','Level_2',['NPV','PV','SOIL'],np.arange(350,2499,2).astype(int).astype(str)))



# Open and filter spectral libraries
for _f in range(len(libraries)):

    libaries[_f].load_data()
    libraries[_f].filter_by_class()



# Now run EAR on each library (independently)

    
