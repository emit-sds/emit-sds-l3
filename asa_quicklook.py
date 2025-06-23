import argparse
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import glob
from spectral.io import envi
from osgeo import gdal


deliver_band_names = [
    "Calcite",
    "Chlorite",
    "Dolomite",
    "Goethite",
    "Gypsum",
    "Hematite",
    "Illite+Muscovite",
    "Kaolinite",
    "Montmorillonite",
    "Vermiculite",
]

def main():

    parser = argparse.ArgumentParser(description="Translate to Rrs. and/or apply masks")
    parser.add_argument('input_file', type=str, metavar='as a nc')
    parser.add_argument('output_file', type=str, metavar='output file to write')
    parser.add_argument('--dpi', type=str, default='200')
    args = parser.parse_args()


    matplotlib.rc('font', **{'size'   : 22})
    fig = plt.figure(figsize=(25, 20), edgecolor='w')
    gs = gridspec.GridSpec(5, 2)

    dat = envi.open(args.input_file + '.hdr').open_memmap(interleave='bip').copy()
    trans = gdal.Open(args.input_file).GetGeoTransform()
    band_names = envi.open(args.input_file + '.hdr').metadata['band names']
    
    for n in range(len(deliver_band_names)):
        ax = plt.subplot(gs[n % 5,int(np.floor(n/5))])

        map = Basemap(llcrnrlon=trans[0], llcrnrlat=trans[3]+dat.shape[0]*trans[5], urcrnrlon=trans[0]+dat.shape[1]*trans[1], urcrnrlat=trans[3],
                    lat_0=-0.01, lon_0=15.0, projection="eqdc")
        map.drawcoastlines()

        
        dat[dat == -9999] = np.nan
        ldat = np.sum(dat[...,np.array([deliver_band_names[n] in x for x in band_names])],axis=-1)
        maxval =max(np.nanpercentile(ldat,95), 0.05)
        im = map.imshow(ldat[::-1,...],vmin=0,vmax=maxval)
        plt.title(deliver_band_names[n] + ' Spectral Abundance')
        cax1 = make_axes_locatable(ax).append_axes("right", size="2%", pad=0.02)
        fig.colorbar(im, cax=cax1);
    
    plt.savefig(args.output_file,dpi=200,bbox_inches='tight')

if __name__ == "__main__":
    main()
