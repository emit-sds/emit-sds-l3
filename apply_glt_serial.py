import argparse
import numpy as np
import pandas as pd
from osgeo import gdal
from spectral.io import envi
import logging
from typing import List
import time
import os
import multiprocessing
import emit_utils.file_checks

from emit_utils.file_checks import envi_header

def _write_bil_chunk(dat, outfile, line, shape, dtype = 'float32'):
    """
    Write a chunk of data to a binary, BIL formatted data cube.
    Args:
        dat: data to write
        outfile: output file to write to
        line: line of the output file to write to
        shape: shape of the output file
        dtype: output data type

    Returns:
        None
    """
    outfile = open(outfile, 'rb+')
    outfile.seek(line * shape[1] * shape[2] * np.dtype(dtype).itemsize)
    outfile.write(dat.astype(dtype).tobytes())
    outfile.close()



def single_image_ortho(img_dat, glt, glt_nodata_value=-9999):
    """Orthorectify a single image
    Args:
        img_dat (array like): raw input image
        glt (array like): glt - 2 band 1-based indexing for output file(x, y)
        glt_nodata_value (int, optional): Value from glt to ignore. Defaults to 0.
    Returns:
        array like: orthorectified version of img_dat
    """
    outdat = np.zeros((glt.shape[0], glt.shape[1], img_dat.shape[-1])) - 9999
    valid_glt = np.all(glt != glt_nodata_value, axis=-1)
    glt[valid_glt] -= 1 # account for 1-based indexing
    outdat[valid_glt, :] = img_dat[glt[valid_glt, 1], glt[valid_glt, 0], :]
    return outdat, valid_glt


def main(input_args=None):
    parser = argparse.ArgumentParser(description="Robust MF")
    parser.add_argument('glt_file', type=str,  metavar='GLT', help='path to glt image')   
    parser.add_argument('raw_file', type=str,  metavar='RAW', help='path to raw image')   
    parser.add_argument('out_file', type=str, metavar='OUTPUT', help='path to output image')   
    parser.add_argument('--mosaic', action='store_true')
    parser.add_argument('-b', type=int, nargs='+',default=-1)   
    args = parser.parse_args(input_args)


    glt_dataset = envi.open(envi_header(args.glt_file))
    glt = glt_dataset.open_memmap(writeable=False, interleave='bip').copy()
    del glt_dataset
    glt_dataset = gdal.Open(args.glt_file)

    if args.mosaic:
        rawspace_files = np.squeeze(np.array(pd.read_csv(args.rawspace_file, header=None)))
        # TODO: make this check more elegant, should run, catch all files present exception, and proceed
        if args.run_with_missing_files is False:
            emit_utils.file_checks.check_raster_files(rawspace_files, map_space=False)
        # TODO: check that all rawspace files have same number of bands
    else:
        emit_utils.file_checks.check_raster_files([args.rawspace_file], map_space=False)
        rawspace_files = [args.rawspace_file]

    ort_img = None
    for rawfile in rawspace_files:
        img_ds = envi.open(envi_header(rawfile))
        if args.b[0] == -1:
            inds = np.arange(int(img_ds.metadata['bands']))
        else:
            inds = np.array(args.b)
        img_dat = img_ds.open_memmap(writeable=False, interleave='bip')[...,inds].copy()

        if ort_img is None:
            ort_img, _ = single_image_ortho(img_dat, glt)
        else:
            ort_img_update, valid_glt = single_image_ortho(img_dat, glt)
            ort_img[valid_glt, :] = ort_img_update[valid_glt, :]

    band_names = None
    if 'band names' in envi.open(envi_header(args.raw_file)).metadata.keys():
        band_names = np.array(envi.open(envi_header(args.raw_file)).metadata['band names'],dtype=str)[inds].tolist()

    # Build output dataset
    driver = gdal.GetDriverByName('ENVI')
    driver.Register()

    #TODO: careful about output datatypes / format
    outDataset = driver.Create(args.out_file, glt.shape[1], glt.shape[0],
                               ort_img.shape[-1], gdal.GDT_Float32, options=['INTERLEAVE=BIL'])
    outDataset.SetProjection(glt_dataset.GetProjection())
    outDataset.SetGeoTransform(glt_dataset.GetGeoTransform())
    for _b in range(1, ort_img.shape[-1]+1):
        outDataset.GetRasterBand(_b).SetNoDataValue(-9999)
        if band_names is not None:
            outDataset.GetRasterBand(_b).SetDescription(band_names[_b-1])
    del outDataset

    _write_bil_chunk(ort_img.transpose((0,2,1)), args.out_file, 0, (glt.shape[0], ort_img.shape[-1], glt.shape[1]))




if __name__ == '__main__':
    main()


