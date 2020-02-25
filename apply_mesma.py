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
from build_endmember_library import SpectralLibrary, get_good_bands_mask, bad_wv_regions
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Execute MESMA in parallel over a BIL line')
parser.add_argument('spectral_library_csv') #/Users/brodrick/Projects/EMIT/Vegetation/mesma_test_data/spectral_set_emc.csv
parser.add_argument('reflectance_input_file') #tutorial_data_set_santa_barbara/010614r4_4-5.rfl.reg
parser.add_argument('output_file_base')
parser.add_argument('-n_cores',type=int,default=1)
parser.add_argument('-complexity_level', metavar='\b', nargs='+', type=int, default=[3, 4],
                    help='the complexity levels for unmixing. e.g. 2 3 4 for 2-, 3- and 4-EM models (default: 2 3)')
parser.add_argument('-vipertools_base',type=str,help='point to the base of the vipertools package, as the pip install is broken')


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
    wavelengths = np.array([float(metadata['Band_' + str(x)]) for x in range(1,ds.RasterCount+1)])
    return wavelengths



#sys.path.extend([args.vipertools_base, os.path.join(args.vipertools_base, 'vipertools')])
#from vipertools.scripts import mesma
import mesma


# Remove hardcoding once libary setup is complete
header = list(pd.read_csv('data/basic_endmember_library.csv'))
header.pop(0)
header = np.array(header)
#endmember_library = SpectralLibrary('data/basic_endmember_library.csv', 'Class',
#                                  ['NPV', 'PV', 'SOIL'], header, header.astype(np.float32))
endmember_library = SpectralLibrary('data/urbanspectraandmeta_endmember.csv', 'Level_2',
                                  np.arange(350, 2499, 2).astype(int).astype(str),np.arange(350, 2499, 2), class_valid_keys=['NPV', 'GV', 'SOIL'],scale_factor=10000. )

endmember_library.load_data()
endmember_library.filter_by_class()
endmember_library.scale_library()
#endmember_library.scale_library(0.7)

refl_file_bands = get_refl_wavelengths(args.reflectance_input_file)
endmember_library.interpolate_library_to_new_wavelengths(refl_file_bands.copy())

#bad_wv_regions = [[0,440],[1310,1490],[1770,2050],[2440,2880]]

endmember_library.remove_wavelength_region_inplace(bad_wv_regions,set_as_nans=True)
#endmember_library.brightness_normalize()

for _r in range(endmember_library.spectra.shape[0]):
    plt.plot(refl_file_bands, endmember_library.spectra[_r,:])
plt.savefig('figs/endmembers.png',dpi=200)
plt.clf()

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

#line = np.squeeze(dataset.ReadAsArray(0,1000,x_len,1))
#line = line[:,np.all(line != -9999,axis=0)]
#
good_bands = get_good_bands_mask(refl_file_bands, bad_wv_regions)
#line = line[good_bands,:]
#line = line / np.sqrt(np.nanmean(np.power(line,2),axis=1))[:,np.newaxis]
#
#for _r in range(line.shape[1]):
#    plt.plot(refl_file_bands[good_bands], line[:,_r],c='black',linewidth=0.1,alpha=0.3)
#plt.savefig('figs/line_spectra.png',dpi=200)
#
#for _r in range(endmember_library.spectra.shape[0]):
#    plt.plot(refl_file_bands, endmember_library.spectra[_r,:])
#plt.savefig('figs/line_and_endmembers.png',dpi=200)
#
#plt.clf()

endmember_library.spectra = endmember_library.spectra[:,good_bands]

#core = mesma.MesmaCore()
#mesma_results = core.execute(line,
#                             endmember_library.spectra.T,
#                             look_up_table=models_object.return_look_up_table(),
#                             constraints=[-0.05,1.05,0.,-9999,-9999,-9999,-9999],
#                             em_per_class=models_object.em_per_class,
#                             )
#quit()


#spectral_library = np.transpose(spectral_library)


########## TEST SINGLE SPECTRUM
#img_dat = dataset.ReadAsArray(2000,1400,1,1).astype(np.float32)
#img_dat[img_dat > 1]  /= 2
#img_dat = img_dat[good_bands,...]
#print(img_dat.shape)
#
#core = mesma.MesmaCore()
#mesma_results = core.execute(img_dat,
#                             endmember_library.spectra.T,
#                             look_up_table=models_object.return_look_up_table(),
#                             em_per_class=models_object.em_per_class,
#                             #constraints=[-0.05,1.05,0.,-9999,-9999,-9999,-9999],
#                             constraints=[-9999,-9999,-9999,-9999,-9999,-9999,-9999],
#                             )


### Set up output files
n_fraction_bands = 4
n_model_bands = 3
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

    # open the dataset fresh for proper parallel operation
    lds = gdal.Open(args.reflectance_input_file, gdal.GA_ReadOnly)
    img_dat = lds.ReadAsArray(0,int(line),int(x_len),1).astype(np.float32)
    del lds

    img_dat = img_dat[good_bands,...]
    img_dat[img_dat > 1]  /= 2

    good_data = np.squeeze(np.all(img_dat != -9999,axis=0))

    if np.sum(good_data) > 0:
   
        #img_dat = img_dat[...,good_data]
        #img_dat = img_dat / np.sqrt(np.nanmean(np.power(img_dat,2),axis=1))[:,np.newaxis]

        core = mesma.MesmaCore()
        mesma_results = core.execute(img_dat[...,good_data],
                                     endmember_library.spectra.T,
                                     look_up_table=models_object.return_look_up_table(),
                                     em_per_class=models_object.em_per_class,
                                     #constraints=[-0.05,1.05,0.,-9999,-9999,-9999,-9999],
                                     #constraints=[0.00,1.0,0.0,1.0,-9999,-9999,-9999],
                                     constraints=[-9999,-9999,-9999,-9999,-9999,-9999,-9999],
                                     )
        mesma_results[1][:3,...] /= np.sum(mesma_results[1][:3,...],axis=0)[np.newaxis,...]


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


## Run in parallel
progress_bar = tqdm(total=y_len, ncols=80)
def progress_update(*a):
    progress_bar.update()

write_lock = multiprocessing.Lock()
pool = multiprocessing.Pool(processes=args.n_cores)
results = []
for l in np.arange(0, y_len).astype(int):
    results.append(pool.apply_async(mesma_line, args=(l,), callback=progress_update))
    #results.append(pool.apply_async(dummy_line, args=(l,)))
results = [p.get() for p in results]
pool.close()
pool.join()


#for l in tqdm(np.arange(0,y_len).astype(int),ncols=80):
#for l in np.arange(0,y_len).astype(int):
#    mesma_line(l)
#    
#    if l % int(y_len / 20.) == 0:
#        print(int(l / int(y_len / 100.)),end=' ',flush=True)












