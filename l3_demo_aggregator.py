"""
This is a simplified form of the L3 aggregator code, meant only for demonstration sites.  The actual code will operate
globally on half degree cells.

Written by: Philip. G. Brodrick
"""

import argparse
import gdal
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from scipy import signal

parser = argparse.ArgumentParser(description='DEMO L3 aggregation')
parser.add_argument('emit_mineral_file')
parser.add_argument('fractional_cover_file')
parser.add_argument('upscaling_factor',type=int)
parser.add_argument('-emit_mineral_uncertainty_file')
parser.add_argument('-fractional_cover_uncertainty_file')
parser.add_argument('-earth_band',type=int,default=2)
parser.add_argument('-mineral_bands', metavar='\b', nargs='+', type=int, default=[-1,-1,-1])
args = parser.parse_args()

if len(args.mineral_bands) != 3:
    print('please pick 3 mineral bands for visualization')
    quit()

if args.upscaling_factor %2 != 0:
    print('Upscaling factor must be even, value provided was: {}'.format(args.upscaling_factor))
    quit()

# Assumes files have same resolution and come from the same grid.
def load_matching_extents(file_1, file_2, mineral_bands):
    ds1 = gdal.Open(file_1, gdal.GA_ReadOnly)
    ds2 = gdal.Open(file_2, gdal.GA_ReadOnly)

    trans_1 = ds1.GetGeoTransform()
    trans_2 = ds1.GetGeoTransform()

    ul_x = max(trans_1[0], trans_2[0])
    ul_y = min(trans_1[3], trans_2[3])

    lr_x = min(trans_1[0] + trans_1[1] * ds1.RasterXSize, trans_2[0] + trans_2[1] * ds2.RasterXSize)
    lr_y = max(trans_1[3] + ds1.RasterYSize * trans_1[5], trans_2[3] + ds2.RasterYSize * trans_2[5])

    if (trans_1[1] < 0 or trans_1[0] < 0):
        print('Warning...mod calcs assume x res > 0 and x ul > 0....extents likely incorrect')
    if (trans_1[3] < 0 or trans_1[5] > 0):
        print('Warning...mod calcs assume y res < 0 and y ul > 0....extents likely incorrect')

    ## load data
    raster_1 = ds1.ReadAsArray(int((ul_x - trans_1[0]) / trans_1[1]),
                               int((ul_y - trans_1[3]) / trans_1[5]),
                               int(round((lr_x-ul_x)/trans_1[1])), int(round((lr_y-ul_y)/trans_1[5])))

    raster_2 = ds2.ReadAsArray(int((ul_x - trans_2[0]) / trans_2[1]),
                               int((ul_y - trans_2[3]) / trans_2[5]),
                               int(round((lr_x-ul_x)/trans_2[1])), int(round((lr_y-ul_y)/trans_2[5])))

    return raster_1, raster_2

# set verbose = True to print things out
verbose=False

SA, fractional_cover = load_matching_extents(args.emit_mineral_file, args.fractional_cover_file, args.mineral_bands)
if -1 in args.mineral_bands:
    data_counts = np.sum(SA > 0,axis=(1,2))
    band_order = np.argsort(data_counts)[::-1].tolist()
    band_order = [x for x in band_order if x not in args.mineral_bands]
    bo_index = 0
    for _n, band in enumerate(args.mineral_bands):
        if band == -1:
            args.mineral_bands[_n] = band_order[bo_index]
            bo_index +=1

SA = SA[args.mineral_bands,...]
mineral_band_names = ['Goethite', 'Hematite', 'Kaolinite', 'Dolomite', 'Illite', 'Vermiculite', 'Montmorillonite', 'Gypsum', 'Calcite', 'Chlorite']

# Scale based on bare-earth fraction
SAp = SA / (fractional_cover[args.earth_band, ...])[np.newaxis, ...]
SAp[:,fractional_cover[args.earth_band,...] < 0.5] = np.nan
SAp[:,np.isnan(fractional_cover[args.earth_band,...])] = np.nan

# Aggregated Spectral Abundance
#ASA = np.zeros(SA.shape)
#window = np.ones((args.upscaling_factor,args.upscaling_factor))
#for dim in range(ASA.shape[0]):
#    ASA[dim,...] = signal.convolve2d(SA[dim,...], window, mode='same')
#    ASA[dim,...] /= signal.convolve2d(np.logical_not(np.all(SA == 0,axis=0)).astype(int), window, mode='same').astype(float)
#ASA = ASA[:,int((window.shape[0]-1)/2)::window.shape[0],int((window.shape[1]-1)/2)::window.shape[1]]

ws = int(args.upscaling_factor/2.)
centers_y = np.arange(ws,(SAp.shape[1]-ws+.1),ws*2).astype(int)
centers_x = np.arange(ws,(SAp.shape[2]-ws+.1),ws*2).astype(int)
ASA = np.zeros((SAp.shape[0],len(centers_y),len(centers_x)))

# There's got to be a more efficient way to do this, but it involves rolling numpy axes, and this isn't actually
# expensive in the end
for _y, ypos in enumerate(centers_y):
    for _x, xpos in enumerate(centers_x):
        ASA[:,_y,_x] = np.nanmean(SAp[:,ypos - ws:ypos + ws, xpos - ws: xpos + ws],axis=(1,2))



fig = plt.figure(figsize=(20, 6))
shape=(4,1)
buffer=0.02
plt_xsize = (1-(shape[0]+1)*buffer)/float(shape[0])
plt_ysize = (1-(shape[1]+1)*buffer)/float(shape[1])
print(plt_xsize)
plt_idx = 0

ax = fig.add_axes([buffer*(plt_idx+1) + plt_idx*plt_xsize,buffer + 0*plt_ysize, plt_xsize,plt_ysize], zorder=1)
to_plot = np.transpose(SA.copy(), (1,2,0))
ax.imshow(to_plot,vmin=0,vmax=0.1)
ax.set_title('L2b Output \n Spectral Abundance Estimate')
ax.set_axis_off()

ax = fig.add_axes([buffer*(plt_idx+1) + plt_idx*plt_xsize, buffer, plt_xsize, buffer*2], zorder=2)
mineral_leg_handles = [Patch(facecolor='red', edgecolor='black',label=mineral_band_names[args.mineral_bands[0]]),
               Patch(facecolor='green', edgecolor='black',label=mineral_band_names[args.mineral_bands[1]]),
               Patch(facecolor='blue', edgecolor='black',label=mineral_band_names[args.mineral_bands[2]])]
plt.legend(handles=mineral_leg_handles, loc='center', ncol=3, frameon=False)
ax.set_axis_off()

plt_idx += 1


ax = fig.add_axes([buffer*(plt_idx+1) + plt_idx*plt_xsize,buffer + 0*plt_ysize, plt_xsize,plt_ysize], zorder=1)
to_plot = np.transpose(fractional_cover[:3, ...], (1,2,0))
ax.imshow(to_plot,vmin=0,vmax=1)
ax.set_title('L3 Intermediate \n Fractional Cover Estimate')
ax.set_axis_off()

ax = fig.add_axes([buffer*(plt_idx+1) + plt_idx*plt_xsize, buffer, plt_xsize, buffer*2], zorder=2)
fc_leg_handles = [Patch(facecolor='red', edgecolor='black',label='PV'),
               Patch(facecolor='green', edgecolor='black',label='NPV'),
               Patch(facecolor='blue', edgecolor='black',label='Soil')]
plt.legend(handles=fc_leg_handles, loc='center', ncol=3, frameon=False)
ax.set_axis_off()

plt_idx += 1


ax = fig.add_axes([buffer*(plt_idx+1) + plt_idx*plt_xsize,buffer + 0*plt_ysize, plt_xsize,plt_ysize], zorder=1)
to_plot = np.transpose(SAp.copy(), (1,2,0))
ax.imshow(to_plot,vmin=0,vmax=0.1)
ax.set_title('L3 Intermediate \n Vegetation-Adjusted Spectral Abundance Estimate')
ax.set_axis_off()


ax = fig.add_axes([buffer*(plt_idx+1) + plt_idx*plt_xsize, buffer, plt_xsize, buffer*2], zorder=2)
plt.legend(handles=mineral_leg_handles, loc='center', ncol=3, frameon=False)
ax.set_axis_off()

plt_idx += 1

ax = fig.add_axes([buffer*(plt_idx+1) + plt_idx*plt_xsize,buffer + 0*plt_ysize, plt_xsize,plt_ysize], zorder=1)
to_plot = np.transpose(ASA.copy(), (1,2,0))
ax.imshow(to_plot,vmin=0,vmax=0.1)
ax.set_title('L3 Output \n Aggregated Spectral Abundance Estimate')
ax.set_axis_off()

ax = fig.add_axes([buffer*(plt_idx+1) + plt_idx*plt_xsize, buffer, plt_xsize, buffer*2], zorder=2)
plt.legend(handles=mineral_leg_handles, loc='center', ncol=3, frameon=False)
ax.set_axis_off()

plt_idx += 1

plt.show()


















