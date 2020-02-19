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


sys.path.extend([args.vipertools_base, os.path.join(args.vipertools_base, 'vipertools')])
from vipertools.scripts import mesma

# Steal from vipertools.io.imports because we want to avoid any library that requires QGIS
def detect_reflectance_scale_factor(array):
    """ Determine the reflectance scale factor [1, 1000 or 10 000] by looking for the largest value in the array.
    :param array: the array for which the reflectance scale factor is to be found
    :return: the reflectance scale factor [int]
    """
    limit = np.nanmax(array)
    if limit < 1:
        return 1
    if limit < 1000:
        return 1000
    else:
        return 10000


spec_lib_df = pd.read_csv(args.spectral_library_csv, sep='\t')
spectral_cols = [x for x in list(spec_lib_df) if x[0] == 'b' and 'Unknown' in x] # TODO: Update to be an argument
import ipdb; ipdb.set_trace()
spectral_library = np.array(spec_lib_df[spectral_cols])

######## Identify valid bands
if (args.autoid_good_bands):
    if (args.good_bands_spectral_library_header is None):
        spectral_library_header = os.path.splitext(args.spectral_library_csv)[0] + '.hdr'
    else:
        spectral_library_header = args.good_bands_spectral_library_header

    if (args.good_bands_image_header is None):
        image_header = os.path.splitext(args.reflectance_input_file)[0] + '.hdr'
        if (os.path.isfile(image_header) is False):
            image_header = args.reflectance_input_file + '.hdr'
    else:
        image_header = args.good_bands_image_header

    sl_bands = open(spectral_library_header, 'r').read().split('}')
    refl_file_bands = open(image_header, 'r').read().split('}')

    sl_ind = [x for x in range(len(sl_bands)) if 'wavelength = ' in sl_bands[x]][0]
    if_ind = [x for x in range(len(refl_file_bands)) if 'wavelength = ' in refl_file_bands[x]][0]

    sl_bands = np.array(
        sl_bands[sl_ind].replace('\n', '').replace('wavelength = { ', '').replace(' ', '').split(',')).astype(float)
    refl_file_bands = np.array(
        refl_file_bands[if_ind].replace('\n', '').replace('wavelength = { ', '').replace(' ', '').split(',')).astype(
        float)

    good_bands = np.array(
        [x for x in range(len(refl_file_bands)) if np.any(np.abs(sl_bands - refl_file_bands[x]) < args.band_id_tolerance)])
else:
    good_bands = np.arange(spectral_library.shape[1])

## FIX THIS
#full_bad_bands = np.zeros(425).astype(bool)
#full_bad_bands[:10] = True
#full_bad_bands[194:207] = True
#full_bad_bands[286:329] = True
#full_bad_bands[419:] = True
#good_bands = np.logical_not(full_bad_bands)

# Grab spectral library class data for mesma
class_list = np.array(spec_lib_df[args.spectral_class_name],dtype=str)
class_list = np.asarray([x.lower() for x in class_list])
unique_classes = np.unique(class_list)
n_classes = len(unique_classes)

# construct basic mesma model object
models_object = mesma.MesmaModels()
models_object.setup(class_list)
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



# If necessary, determine the reflectance scaling by passing through the image
if (args.reflectance_scale is None):
    img_scale = 1
    #for l in tqdm(np.arange(0,y_len).astype(int),ncols=80):
    #    img_dat = dataset.ReadAsArray(0,int(l),int(x_len),1)
    #    img_scale = max(img_scale, detect_reflectance_scale_factor(img_dat))
else:
    img_scale = args.reflectance_scale

spectral_library = spectral_library / detect_reflectance_scale_factor(spectral_library)
spectral_library = np.transpose(spectral_library)

import ipdb; ipdb.set_trace()


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
    line = 1000
    import ipdb; ipdb.set_trace()
    img_dat = dataset.ReadAsArray(0,int(line),int(x_len),1)[good_bands,...].astype(np.float32)
    img_dat = img_dat / img_scale
    img_dat[img_dat > 1] = -9999
    core = mesma.MesmaCore()
    mesma_results = core.execute(img_dat,
                                 spectral_library,
                                 look_up_table=models_object.return_look_up_table(),
                                 em_per_class=models_object.em_per_class,
                                 )
    import ipdb; ipdb.set_trace()
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

#pool = multiprocessing.Pool(processes=args.n_cores)
#results = []
#for l in np.arange(0, y_len).astype(int):
#    results.append(pool.apply_async(mesma_line, args=(l,), callback=progress_update))
#results = [p.get() for p in results]
#pool.close()
#pool.join()


for l in tqdm(np.arange(0,y_len).astype(int)):
    mesma_line(l)












