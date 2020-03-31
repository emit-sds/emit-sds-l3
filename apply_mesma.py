"""
This is a simplified input form of the MESMA command line interface (adapted from original) written via vipertools
(https://viper-tools.readthedocs.io/en/latest/reference/scripts/mesma.html), written in such a way that not
QGIS python install is necessary, and to execute quickly on BIL formatted reflectance files, correcting for missing
bands on the fly and operating in parallel if desired.

Written by: Philip. G. Brodrick
"""

import argparse
import os
import numpy as np
import gdal
from tqdm import tqdm
import multiprocessing
from build_endmember_library import SpectralLibrary, get_good_bands_mask, bad_wv_regions
import matplotlib.pyplot as plt
from mesma import mesma

# MESMA throws a series of divide-by-zero errors that produce objectionable output, and can be safely ignored
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Execute MESMA in parallel over a BIL line')
parser.add_argument('reflectance_input_file')
parser.add_argument('endmember_file', default='data/urbanspectraandmeta_endmember.csv')
parser.add_argument('endmember_class')
parser.add_argument('output_file_base')
parser.add_argument('-build_plots', default=0)
parser.add_argument('-plot_line_spectra', default=-1)
parser.add_argument('-n_cores', type=int, default=1)
parser.add_argument('-refl_nodata', type=float, default=-9999)
parser.add_argument('-refl_scale', type=float, default=2.)
parser.add_argument('-complexity_level', metavar='\b', nargs='+', type=int, default=[3, 4],
                    help='the complexity levels for unmixing. e.g. 2 3 4 for 2-, 3- and 4-EM models (default: 2 3)')
args = parser.parse_args()



def get_refl_wavelengths(raster_file):
    ds = gdal.Open(raster_file, gdal.GA_ReadOnly)
    metadata = ds.GetMetadata()
    wavelengths = np.array([float(metadata['Band_' + str(x)]) for x in range(1,ds.RasterCount+1)])
    return wavelengths

# Load endmember library
endmember_library = SpectralLibrary(args.endmember_file, args.endmember_class,
                                    np.arange(350, 2499, 2).astype(int).astype(str), np.arange(350, 2499, 2),
                                    class_valid_keys=['NPV', 'GV', 'SOIL'], scale_factor=20000.)
endmember_library.load_data()
endmember_library.filter_by_class()
endmember_library.scale_library()

# Get reflectance file wavelengths, and interpolate the endmember library to match
refl_file_bands = get_refl_wavelengths(args.reflectance_input_file)
endmember_library.interpolate_library_to_new_wavelengths(refl_file_bands.copy())

# Remove endmember
endmember_library.remove_wavelength_region_inplace(bad_wv_regions, set_as_nans=True)
#endmember_library.brightness_normalize() # currently disabled - would require some mesma source code reworking

# Find the bands we need to eliminate from both endmember library, and eventually from the reflectance file on data load
good_bands = get_good_bands_mask(refl_file_bands, bad_wv_regions)

# Open up the reflectance dataset, store some variables for convenience
reflectance_dataset = gdal.Open(args.reflectance_input_file, gdal.GA_ReadOnly)
x_len = int(reflectance_dataset.RasterXSize)
y_len = int(reflectance_dataset.RasterYSize)

# Build some plots, if requested
if (args.build_plots == 1):
    if (os.path.isdir('figs') is False):
        os.mkdir('figs')
    for _r in range(endmember_library.spectra.shape[0]):
        plt.plot(refl_file_bands, endmember_library.spectra[_r, :])
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Scaled Reflectance')
    plt.savefig('figs/endmembers.png', dpi=200)
    plt.clf()

    if (args.plot_line_spectra > 0):
        if (args.plot_line_spectra > y_len):
            print('Line for plotting, {}, exceeds file length, {}.  Skipping'.format(args.plot_line_spectra, y_len))

        line = np.squeeze(reflectance_dataset.ReadAsArray(0, 1000, x_len, 1))
        line = line[:, np.all(line != args.refl_nodata, axis=0)]
        line /= args.refl_scale
        line[np.logical_not(good_bands), :] = np.nan

        for _r in range(line.shape[1]):
            plt.plot(refl_file_bands[good_bands], line[:,_r], c='black', linewidth=0.1, alpha=0.3)
        plt.savefig('figs/line_spectra.png', dpi=200)

        for _r in range(endmember_library.spectra.shape[0]):
            plt.plot(refl_file_bands, endmember_library.spectra[_r, :])
        plt.savefig('figs/line_and_endmembers.png', dpi=200)

        plt.clf()

# Now that plotting and interpolation are complete, clean out the bad bands from the endmember library
endmember_library.spectra = endmember_library.spectra[:, good_bands]

# Calculate the number of classes in the endmember
n_classes = len(np.unique(endmember_library.classes))

# construct basic mesma model object
models_object = mesma.MesmaModels()
models_object.setup(endmember_library.classes)

# Populate mesma models object with appropriate complexity levels
require_list = [2,3]
for level in require_list:
    if level not in args.complexity_level:
        models_object.select_level(state=False, level=level)
    else:
        args.complexity_level.remove(level)

for level in args.complexity_level:
    models_object.select_level(state=True, level=level)
    for i in np.arange(n_classes):
        models_object.select_class(state=True, index=i, level=level)

print("Total number of models: " + str(models_object.total()))

# Set up output files
n_fraction_bands = n_classes + 1
n_model_bands = n_classes
n_rmse_bands = 1

output_files = [args.output_file_base + '_model', args.output_file_base + '_fraction', args.output_file_base + '_rmse']
output_bands = [n_model_bands, n_fraction_bands, n_rmse_bands]
driver = gdal.GetDriverByName('ENVI')
driver.Register()

for _n in range(len(output_files)):
    outDataset = driver.Create(output_files[_n], x_len, y_len, output_bands[_n], gdal.GDT_Float32,
                               options=['INTERLEAVE=BIL'])
    outDataset.SetGeoTransform(reflectance_dataset.GetGeoTransform())
    outDataset.SetProjection(reflectance_dataset.GetProjection())
    del outDataset

# Define a function to run Mesma on one line of data
def mesma_line(line):

    # open the dataset fresh for proper parallel operation, read and remove dataset from memory
    lds = gdal.Open(args.reflectance_input_file, gdal.GA_ReadOnly)
    img_dat = lds.ReadAsArray(0, int(line), int(x_len), 1).astype(np.float32)[good_bands,...]
    del lds

    # Check from nodata regions
    good_data = np.squeeze(np.all(img_dat != args.refl_nodata,axis=0))

    if np.sum(good_data) > 0:
        img_dat = img_dat[...,good_data]
        #img_dat = img_dat / np.sqrt(np.nanmean(np.power(img_dat,2),axis=1))[:,np.newaxis]
        img_dat /= args.refl_scale

        core = mesma.MesmaCore()
        mesma_results = core.execute(img_dat,
                                     endmember_library.spectra.T,
                                     look_up_table=models_object.return_look_up_table(),
                                     em_per_class=models_object.em_per_class,
                                     constraints=[-9999,-9999,-9999,-9999,-9999,-9999,-9999],
                                     )

        # 'Shade normalize' the fraction output....aka, account for variable surface brightness
        mesma_results[1][:n_classes,...] /= np.sum(mesma_results[1][:n_classes, ...], axis=0)[np.newaxis, ...]

        # Open each dataset, re-order the binary data for BIL write, and dump to file
        for _n in range(len(output_files)):
            lr = mesma_results[_n].copy()
            if (len(lr.shape) == 2):
                lr = lr.reshape((1, lr.shape[0], lr.shape[1]))
            lr = lr.swapaxes(0,1)

            write_lock.acquire()  
            memmap = np.memmap(output_files[_n], mode='r+', shape=(y_len, output_bands[_n], x_len), dtype=np.float32)
            memmap[line:line+1,:,good_data] = lr
            write_lock.release()
            del memmap

# Set up progress bar for output
progress_bar = tqdm(total=y_len, ncols=80)
def progress_update(*a):
    progress_bar.update()

# Establish a write-lock....not strictly necessary, but for formality
write_lock = multiprocessing.Lock()
pool = multiprocessing.Pool(processes=args.n_cores)

# Run asynchronously
results = []
for l in np.arange(0, y_len).astype(int):
    results.append(pool.apply_async(mesma_line, args=(l,), callback=progress_update))
results = [p.get() for p in results]
pool.close()
pool.join()







