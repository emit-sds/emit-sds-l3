


from spectral.io import envi
import argparse
from osgeo import gdal
import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch
from collections import OrderedDict






def write_output_file(source_ds, output_img, output_file):
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    outDataset = driver.Create(output_file,source_ds.RasterXSize,source_ds.RasterYSize,3,gdal.GDT_Byte,options = ['COMPRESS=LZW'])
    outDataset.SetProjection(source_ds.GetProjection())
    outDataset.SetGeoTransform(source_ds.GetGeoTransform())
    for _b in range(output_img.shape[-1]):
        outDataset.GetRasterBand(_b+1).WriteArray(output_img[...,_b])
        outDataset.GetRasterBand(_b+1).SetNoDataValue(0)
    del outDataset




def envi_header(inputpath):
    """
    Convert a envi binary/header path to a header, handling extensions
    Args:
        inputpath: path to envi binary file
    Returns:
        str: the header file associated with the input reference.

    """
    if os.path.splitext(inputpath)[-1] == '.img' or os.path.splitext(inputpath)[-1] == '.dat' or os.path.splitext(inputpath)[-1] == '.raw':
        # headers could be at either filename.img.hdr or filename.hdr.  Check both, return the one that exists if it
        # does, if not return the latter (new file creation presumed).
        hdrfile = os.path.splitext(inputpath)[0] + '.hdr'
        if os.path.isfile(hdrfile):
            return hdrfile
        elif os.path.isfile(inputpath + '.hdr'):
            return inputpath + '.hdr'
        return hdrfile
    elif os.path.splitext(inputpath)[-1] == '.hdr':
        return inputpath
    else:
        return inputpath + '.hdr'



def main():

    parser = argparse.ArgumentParser(description="Translate to Rrs. and/or apply masks")
    parser.add_argument('input_file', type=str, metavar='mosaic_glt_file')
    parser.add_argument('output_file', type=str, metavar='output file to write')
    parser.add_argument('--max_val', type=float, default=30, metavar='maximum value')
    args = parser.parse_args()

    print('read input ds')
    source_ds = gdal.Open(args.input_file,gdal.GA_ReadOnly)
    trans = source_ds.GetGeoTransform()
    print(trans)

    print('read sa')
    revisits = source_ds.GetRasterBand(source_ds.RasterCount).ReadAsArray() 
    subset = revisits > 0

    scaled_revisit = revisits / args.max_val
    scaled_revisit[scaled_revisit > 1] = 1

    output_img = np.zeros((scaled_revisit.shape[0],scaled_revisit.shape[1],3))
    output_img[subset,:] = plt.cm.winter(scaled_revisit[subset])[...,:3]
    output_img = np.round(output_img * 255).astype(np.uint8)
    output_img[subset,:] = np.maximum(1, output_img[subset,:])

    write_output_file(source_ds, output_img, args.output_file)




if __name__ == "__main__":
    main()

