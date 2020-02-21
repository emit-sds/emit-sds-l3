"""
This is a simplified input form of the MESMA command line interface (adapted from original) written via vipertools
(https://viper-tools.readthedocs.io/en/latest/reference/scripts/mesma.html), written in such a way that not
QGIS python install is necessary, and to execute quickly on BIL formatted reflectance files, correcting for missing
bands on the fly and operating in parallel if desired.

Written by: Philip. G. Brodrick
"""

import argparse
import sys, os
import pandas as pd
import numpy as np
import gdal
from tqdm import tqdm
import multiprocessing
from build_endmember_library import SpectralLibrary, remove_wavelength_region

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Execute MESMA in parallel over a BIL line')
parser.add_argument('spectral_library_csv') #/Users/brodrick/Projects/EMIT/Vegetation/mesma_test_data/spectral_set_emc.csv
parser.add_argument('reflectance_input_file') #tutorial_data_set_santa_barbara/010614r4_4-5.rfl.reg
parser.add_argument('output_file_base')
parser.add_argument('-spectral_class_name',default='class')
parser.add_argument('-n_cores',type=int,default=1)
parser.add_argument('-complexity_level', metavar='\b', nargs='+', type=int, default=[2, 3],
                    help='the complexity levels for unmixing. e.g. 2 3 4 for 2-, 3- and 4-EM models (default: 2 3)')
parser.add_argument('-reflectance_scale', metavar='\b', type=int,
                    help='image reflectance scale factor (default: derived from data as 1, 1 000 or 10 000)')
parser.add_argument('-vipertools_base',type=str,help='point to the base of the vipertools package, as the pip install is broken')

# constraints
parser.add_argument('-u', '--unconstrained', action='store_true', default=False,
                    help='run mesma without constraints (default off)')
parser.add_argument('--min-fraction', metavar='\b', type=float, default=-0.05,
                    help='minimum allowable endmember fraction (default -0.05), use -9999 to set no constraint')
parser.add_argument('--max-fraction', metavar='\b', type=float, default=1.05,
                    help='maximum allowable endmember fraction (default 1.05), use -9999 to set no constraint')
parser.add_argument('--min-shade-fraction', metavar='\b', type=float, default=0.00,
                    help='minimum allowable shade fraction (default 0.00), use -9999 to set no constraint')
parser.add_argument('--max-shade-fraction', metavar='\b', type=float, default=0.80,
                    help='maximum allowable shade fraction (default 0.80), use -9999 to set no constraint')
parser.add_argument('--max-rmse', metavar='\b', type=float, default=0.025,
                    help='maximum allowable RMSE (default 0.025), use -9999 to set no constraint')
parser.add_argument('--residual-constraint', action='store_true', default=False,
                    help='use a residual constraint (default off)')
parser.add_argument('--residual-constraint-values', metavar='\b', type=float, nargs="+", default=(0.025, 7),
                    help='two values (residual threshold, number of bands): the number of consecutive bands that '
                         'the residual values are allowed to exceed the given threshold (default: 0.025 7)')

# manage good bands
parser.add_argument('-autoid_good_bands', type=bool,default=True, help='flag to auto ID and discard bands that are '
                    'not in the spectral library but are in the reflectance image')
parser.add_argument('-good_bands_spectral_library_header',type=str,
                    help='use a custom spectral library header to ID good bands, defaults to spectral_library_csv with'
                    ' .hdr extention')
parser.add_argument('-good_bands_image_header',type=str,
                    help='use a custom image header to ID good bands, defaults to reflectance_input_file with'
                         ' .hdr extention')
parser.add_argument('-band_id_tolerance', type=float, default=3, help='tolerance for identification of relevant bands, in nm')

"""
Example implementation
script_inputs = [
    '/Users/brodrick/Projects/EMIT/Vegetation/mesma_test_data/spectral_set_emc.csv',
    '/Users/brodrick/Projects/EMIT/Vegetation/mesma_test_data/tutorial_data_set_santa_barbara/010614r4_4-5.rfl.reg',
    'output_full_script',
    '-n_cores', '6',
    '-complexity_level', '2',
    '-good_bands_spectral_library_header', '/Users/brodrick/Projects/EMIT/Vegetation/mesma_test_data/roi_extraction_library_emc.hdr',
    '-vipertools_base', '/Users/brodrick/repos/vipertools-3.0.8',
]
#args = parser.parse_args(script_inputs)
"""
args = parser.parse_args()



def get_refl_wavelengths(raster_file):
    ds = gdal.Open(raster_file, gdal.GA_ReadOnly)
    metadata = ds.GetMetadata()
    wavelengths = np.array(float(metadata['Band_' + str(x)]) for x in range(1,ds.RasterCount+1))
    return wavelengths



sys.path.extend([args.vipertools_base, os.path.join(args.vipertools_base, 'vipertools')])
from vipertools.scripts import mesma


# Remove hardcoding once libary setup is complete
header = list(pd.read_csv('data/basic_endmember_library.csv'))
header.pop(0)
endmember_library = SpectralLibrary('data/basic_endmember_library.csv', 'Class',
                                  ['NPV', 'PV', 'SOIL'], header, header.astype(np.float32))

endmember_library.load_data()
endmember_library.filter_by_class()
endmember_library.scale_library(10000.)

refl_file_bands = get_refl_wavelengths(args.reflectance_input_file)
endmember_library.interpolate_library_to_new_wavelengths(refl_file_bands)

bad_wv_regions = [[0,440],[1330,1490],[1170,2050],[2440,2880]]

for bwv in bad_wv_regions:
    endmember_library.remove_wavelength_region_inplace(bwv[0],bwv[1])

n_classes = len(np.unique(endmember_library.classes))

# construct basic mesma model object
models_object = mesma.MesmaModels()
models_object.setup(endmember_library.classes)

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


######## Open up the dataset
dataset = gdal.Open(args.reflectance_input_file, gdal.GA_ReadOnly)
x_len = int(dataset.RasterXSize)
y_len = int(dataset.RasterYSize)


#spectral_library = np.transpose(spectral_library)

### Set up output files
n_fraction_bands = 8
n_model_bands = 7
n_rmse_bands = 1

output_files = [args.output_file_base + '_model', args.output_file_base + '_fraction', args.output_file_base + '_rmse']
output_bands = [n_model_bands, n_fraction_bands, n_rmse_bands]
driver = gdal.GetDriverByName('ENVI')
driver.Register()

for _n in range(len(output_files)):
    outDataset = driver.Create(output_files[_n],x_len,y_len,output_bands[_n],gdal.GDT_Float32,options=['INTERLEAVE=BIL'])
    outDataset.SetGeoTransform(dataset.GetGeoTransform())
    outDataset.SetProjection(dataset.GetProjection())
    del outDataset


# Define a function to run Mesma on one line of data
def mesma_line(line):
    img_dat = dataset.ReadAsArray(0,int(line),int(x_len),1).astype(np.float32)
    img_dat, refl_file_bands_tmp = remove_wavelength_region(img_dat, refl_file_bands.copy(), bad_wv_regions)

    img_dat = img_dat
    img_dat[img_dat > 1] = -9999
    core = mesma.MesmaCore()
    mesma_results = core.execute(img_dat,
                                 endmember_library.spectra.T,
                                 look_up_table=models_object.return_look_up_table(),
                                 em_per_class=models_object.em_per_class,
                                 )

    for _n in range(len(output_files)):
        lr = mesma_results[_n]
        if (len(lr.shape) == 2):
            lr = lr.reshape((1, lr.shape[0], lr.shape[1]))
        lr = lr.swapaxes(0,1)
        memmap = np.memmap(output_files[_n], mode='r+', shape=(y_len, output_bands[_n], x_len), dtype=np.float32)
        memmap[line,...] = lr
        del memmap


# Run in parallel
progress_bar = tqdm(total=y_len, ncols=80)
def progress_update(*a):
    progress_bar.update()

pool = multiprocessing.Pool(processes=args.n_cores)
results = []
for l in np.arange(0, y_len).astype(int):
    results.append(pool.apply_async(mesma_line, args=(l,), callback=progress_update))
results = [p.get() for p in results]
pool.close()
pool.join()


#for l in tqdm(np.arange(0,y_len).astype(int)):
#    mesma_line(l)












