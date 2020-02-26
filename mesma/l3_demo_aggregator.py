"""
This is a simplified form of the L3 aggregator code, meant only for demonstration sites.  The actual code will operate
globally on half degree cells.

Written by: Philip. G. Brodrick
"""

import argparse
import gdal
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

parser = argparse.ArgumentParser(description='DEMO L3 aggregation')
parser.add_argument('emit_mineral_file')
parser.add_argument('fractional_cover_file')
parser.add_argument('upscaling_factor',type=int)
parser.add_argument('-emit_mineral_uncertainty_file')
parser.add_argument('-fractional_cover_uncertainty_file')
parser.add_argument('-earth_band',type=int,default=2)
args = parser.parse_args()

if args.upscaling_factor %2 != 0:
    print('Upscaling factor must be even, value provided was: {}'.format(args.upscaling_factor))
    quit()

# Assumes files have same resolution and come from the same grid.
def load_matching_extents(file_1, file_2):
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

SA, fractional_cover = load_matching_extents(args.emit_mineral_file, args.fractional_cover_file)

# Scale based on bare-earth fraction
SAp = SA / (fractional_cover[args.earth_band, ...])[np.newaxis, ...]

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

fig = plt.figure(figsize=(10, 5))

ax = fig.add_axes([0.05,0.05,0.4,0.9], zorder=1)
to_plot = SA[:3,...].copy()
to_plot = to_plot.swapaxes(0,1)
to_plot = to_plot.swapaxes(1,2)
ax.imshow(to_plot,vmin=0,vmax=1)
ax.set_title('L2b output (Spectral Abundance Estimate)')
ax.set_axis_off()

ax = fig.add_axes([0.5,0.05,0.4,0.9], zorder=1)
to_plot = ASA[:3,...].copy()
to_plot = to_plot.swapaxes(0,1)
to_plot = to_plot.swapaxes(1,2)
ax.imshow(to_plot,vmin=0,vmax=1)
ax.set_title('L3 intermediate fractional cover estimate')
ax.set_axis_off()

#ax = fig.add_axes([0.5,0.05,0.4,0.9], zorder=1)
#to_plot = ASA[:3,...].copy()
#to_plot = to_plot.swapaxes(0,1)
#to_plot = to_plot.swapaxes(1,2)
#ax.imshow(to_plot,vmin=0,vmax=1)
#ax.set_title('L3 intermediate vegetation-adjusted Spectral Abundance Estimate')
#
#ax = fig.add_axes([0.5,0.05,0.4,0.9], zorder=1)
#to_plot = ASA[:3,...].copy()
#to_plot = to_plot.swapaxes(0,1)
#to_plot = to_plot.swapaxes(1,2)
#ax.imshow(to_plot,vmin=0,vmax=1)
#ax.set_title('L3 output (Aggregated Spectral Abundance Estimate)')


plt.show()



















