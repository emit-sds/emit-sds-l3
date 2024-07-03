"""
Apply a (possibly multi-file) per-pixel spatial reference, in serial (rayless).

Author: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov
"""


import argparse
import numpy as np
import pandas as pd
import os
from osgeo import gdal
from spectral.io import envi
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



def single_image_ortho(img_dat, glt, img_ind=None, glt_nodata_value=0):
    """Orthorectify a single image
    Args:
        img_dat (array like): raw input image
        glt (array like): glt - 2 band 1-based indexing for output file(x, y)
        img_ind (int): index of image in glt (if mosaic - otherwise ignored)
        glt_nodata_value (int, optional): Value from glt to ignore. Defaults to 0.
    Returns:
        array like: orthorectified version of img_dat
    """
    outdat = np.zeros((glt.shape[0], glt.shape[1], img_dat.shape[-1])) - 9999
    valid_glt = np.all(glt != glt_nodata_value, axis=-1)

    # Only grab data from the correct image, if this is a mosaic
    if glt.shape[2] >= 3:
        valid_glt[glt[:,:,2] != img_ind] = False

    if np.sum(valid_glt) == 0:
        return outdat, valid_glt

    glt[valid_glt] -= 1 # account for 1-based indexing
    outdat[valid_glt, :] = img_dat[glt[valid_glt, 1], glt[valid_glt, 0], :]
    return outdat, valid_glt


def main(input_args=None):
    parser = argparse.ArgumentParser(description="Robust MF")
    parser.add_argument('glt_file', type=str,  metavar='GLT', help='path to glt image')   
    parser.add_argument('raw_file', type=str,  metavar='RAW', help='path to raw image')   
    parser.add_argument('out_file', type=str, metavar='OUTPUT', help='path to output image')   
    parser.add_argument('--mosaic', action='store_true')
    parser.add_argument('--glt_nodata', type=float, default=0)
    parser.add_argument('--run_with_missing_files', action='store_true')
    parser.add_argument('-b', type=int, nargs='+',default=[-1])   
    args = parser.parse_args(input_args)


    glt_dataset = envi.open(envi_header(args.glt_file))
    glt = glt_dataset.open_memmap(writeable=False, interleave='bip').copy().astype(int)
    del glt_dataset
    glt_dataset = gdal.Open(args.glt_file)

    if args.mosaic:
        rawspace_files = [x.strip() for x in open(args.raw_file).readlines()]
        # TODO: make this check more elegant, should run, catch all files present exception, and proceed
        if args.run_with_missing_files is False:
            emit_utils.file_checks.check_raster_files(rawspace_files, map_space=False)
        # TODO: check that all rawspace files have same number of bands
    else:
        emit_utils.file_checks.check_raster_files([args.raw_file], map_space=False)
        rawspace_files = [args.raw_file]

    ort_img = None
    band_names = None
    for _rf, rawfile in enumerate(rawspace_files):
        print(f'{_rf+1}/{len(rawspace_files)}')
        if os.path.isfile(envi_header(rawfile)) and os.path.isfile(rawfile):
            
            # Don't load image data unless we have to
            if args.mosaic:
                if np.sum(glt[:,:,2] == _rf+1) == 0:
                    continue

            img_ds = envi.open(envi_header(rawfile))
            inds = None
            if args.b[0] == -1:
                inds = np.arange(int(img_ds.metadata['bands']))
            else:
                inds = np.array(args.b)
            img_dat = img_ds.open_memmap(writeable=False, interleave='bip')[...,inds].copy()

            if band_names is None and 'band names' in envi.open(envi_header(rawfile)).metadata.keys():
                band_names = np.array(envi.open(envi_header(rawfile)).metadata['band names'],dtype=str)[inds].tolist()

            if ort_img is None:
                ort_img, _ = single_image_ortho(img_dat, glt, img_ind=_rf+1, glt_nodata_value=args.glt_nodata)
            else:
                ort_img_update, valid_glt = single_image_ortho(img_dat, glt, img_ind=_rf+1, glt_nodata_value=args.glt_nodata)
                ort_img[valid_glt, :] = ort_img_update[valid_glt, :]

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


