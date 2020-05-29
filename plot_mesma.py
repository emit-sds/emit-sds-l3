
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import gdal
import gzip
import numpy as np
import math
from skimage import transform as tf

import spectral.io.envi as envi
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys, os

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"


mesma_file = sys.argv[1]
refl_file = sys.argv[2]

mesma_dataset = gdal.Open(mesma_file, gdal.GA_ReadOnly)
rows = mesma_dataset.RasterYSize
cols = mesma_dataset.RasterXSize
#use_rows = [11000, 14000]
use_rows = [16000, 19500]
mesma = np.memmap(mesma_file, mode='r', shape=(rows, mesma_dataset.RasterCount, cols), dtype=np.float32)

refl_dataset = gdal.Open(refl_file ,gdal.GA_ReadOnly)
refl = np.memmap(refl_file, mode='r', shape=(rows, refl_dataset.RasterCount, cols), dtype=np.int16)

rgb = refl[: ,np.array([12 ,21 ,30]) ,:].copy().transpose((0 ,2 ,1)).astype(float ) /20000.
rgb = rgb[use_rows[0]:use_rows[1] ,: ,::-1]
rgb -= np.percentile(rgb ,2 ,axis=(0 ,1))[np.newaxis ,np.newaxis ,:]
rgb /= np.percentile(rgb ,98 ,axis=(0 ,1))[np.newaxis ,np.newaxis ,:]

mesma = mesma[use_rows[0]:use_rows[1],...].copy().transpose((0,2,1)).astype(float)
mesma = mesma[:,:,np.array([1,0,2])]
mesma -= np.percentile(mesma ,2 ,axis=(0 ,1))[np.newaxis ,np.newaxis ,:]
mesma /= np.percentile(mesma ,98 ,axis=(0 ,1))[np.newaxis ,np.newaxis ,:]

fig = plt.figure(figsize=(5 ,8))
spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig, left=0.05, wspace=0.05)

ax = fig.add_subplot(spec[: ,0])
ax.imshow(rgb)
ax.axis('off')
ax.set_title('RGB',fontsize=16)

ax = fig.add_subplot(spec[: ,1])
ax.imshow(mesma[:,:,:3],vmin=0,vmax=1)
ax.axis('off')
ax.set_title('Fractional\nCover',fontsize=16)


mineral_leg_handles = [Patch(facecolor='red', edgecolor='black', label='NPV'),
                       Patch(facecolor='green', edgecolor='black', label='PV'),
                       Patch(facecolor='blue', edgecolor='black', label='BARE')]
ax = fig.add_subplot(spec[0,-1])
ax.legend(handles=mineral_leg_handles, loc='center', ncol=1, frameon=False)
plt.axis('off')


plt.savefig('figs/mesma_example.png',dpi=300,bbox_inches='tight')
plt.show()


