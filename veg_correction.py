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



def main(input_args=None):
    parser = argparse.ArgumentParser(description="Robust MF")
    parser.add_argument('abun_file', type=str)   
    parser.add_argument('cover_file', type=str)   
    parser.add_argument('--out_file', type=str, default=None)   
    parser.add_argument('--soil_thresh', type=float, default=0.001)   
    parser.add_argument('--coarsened_file', type=str, default=None)   
    parser.add_argument('--mask_fraction_file', type=str, default=None)   
    parser.add_argument('--resolution', type=float, default=None)   
    parser.add_argument('--data_threshold', type=float, default=None)   
    parser.add_argument('--abun_uncert_file', type=str, default=None)   
    parser.add_argument('--cover_uncert_file', type=str, default=None)   
    parser.add_argument('--valid_fraction_file', type=str, default=None)   
    parser.add_argument('--mask_file', type=str, default=None)   
    parser.add_argument('--thresh_only', action='store_true')   
    args = parser.parse_args(input_args)


    abun_ds = envi.open(envi_header(args.abun_file))
    band_names = abun_ds.metadata['band names']
    abun_gdal = gdal.Open(args.abun_file)
    cover_ds = envi.open(envi_header(args.cover_file))

    abun = abun_ds.open_memmap(interleave='bip').copy()
    cover = cover_ds.open_memmap(interleave='bip')[...,2].copy()
    counts = {}

    if args.thresh_only is False:
        abun = abun / cover[:,:,np.newaxis]

    counts['no_abun'] = np.any(np.logical_or.reduce((np.isnan(abun) , np.isfinite(abun) == False, abun == -9999)),axis=-1)
    masked_out = np.any(np.isnan(abun), axis=-1)
    masked_out[np.any(np.isfinite(abun) == False,axis=-1)] = True
    masked_out[np.any(abun == -9999,axis=-1)] = True

    counts['soil_cutoff'] = np.logical_and(masked_out == False, cover < args.soil_thresh)
    masked_out[cover < args.soil_thresh] = True

    if args.mask_file is not None:
        ext_mask = gdal.Open(args.mask_file).ReadAsArray()
        counts['external_mask'] = np.logical_and(masked_out == False, ext_mask == 1)
        masked_out[ext_mask == 1] = True

    abun[masked_out,:] = -9999
    cover[masked_out] = -9999
    #abun[np.isnan(abun)] = -9999
    #abun[np.isfinite(abun) == False] = -9999
    #abun[cover < args.soil_thresh,:] = -9999

    do_uncert = False
    if args.abun_uncert_file is not None and args.cover_uncert_file is not None and args.coarsened_file is not None and args.resolution is not None:
        abununcert_ds = envi.open(envi_header(args.abun_uncert_file))
        coveruncert_ds = envi.open(envi_header(args.cover_uncert_file))
        abununcert = abununcert_ds.open_memmap(interleave='bip').copy()
        coveruncert = coveruncert_ds.open_memmap(interleave='bip')[...,2].copy()

        abununcert[masked_out,:] = -9999
        coveruncert[masked_out] = -9999
        do_uncert = True


    # Build output dataset
    driver = gdal.GetDriverByName('ENVI')
    driver.Register()

    #TODO: careful about output datatypes / format
    if args.out_file is not None:
        outDataset = driver.Create(args.out_file, abun.shape[1], abun.shape[0],
                                   abun.shape[2], gdal.GDT_Float32, options=['INTERLEAVE=BIL'])
        outDataset.SetProjection(abun_gdal.GetProjection())
        outDataset.SetGeoTransform(abun_gdal.GetGeoTransform())
        for _b in range(1, abun.shape[2]+1):
            outDataset.GetRasterBand(_b).SetNoDataValue(-9999)
            if band_names is not None:
                outDataset.GetRasterBand(_b).SetDescription(band_names[_b-1])
        del outDataset

        _write_bil_chunk(abun.transpose((0,2,1)), args.out_file, 0, (abun.shape[0], abun.shape[2], abun.shape[1]))


    if args.coarsened_file is not None and args.resolution is not None:

        trans = abun_gdal.GetGeoTransform()
        num_px = int(round(args.resolution / trans[1]))

        abun[abun == -9999] = np.nan

        numy = int(round(abun.shape[0] / num_px))
        numx = int(round(abun.shape[1] / num_px))
        asa = np.zeros((numy, numx,abun.shape[2])) - 9999
        agg_count = np.zeros((numy, numx,len(counts.keys()))) - 9999
        asa_unc = None
        if do_uncert:
            asa_unc = np.zeros((numy, numx,abun.shape[2])) - 9999

        for _y in range(0,numy):
            for _x in range(0,numx):
                valid_px = np.sum(masked_out[_y*num_px:(_y+1)*num_px,_x*num_px:(_x+1)*num_px] == False)
                if args.data_threshold is not None:
                    complete_frac = valid_px / float(num_px**2)
                    if complete_frac < args.data_threshold:
                        continue
                asa[_y,_x,:] = np.nanmean(abun[_y*num_px:(_y+1)*num_px,_x*num_px:(_x+1)*num_px,:],axis=(0,1))
                for _key, key in enumerate(counts.keys()):
                    agg_count[_y,_x,_key] = np.sum(counts[key][_y*num_px:(_y+1)*num_px,_x*num_px:(_x+1)*num_px]) / np.product(abun[_y*num_px:(_y+1)*num_px,_x*num_px:(_x+1)*num_px,0].shape)

                if do_uncert:
                    valid_unc = abununcert[_y*num_px:(_y+1)*num_px,_x*num_px:(_x+1)*num_px,:]
                    valid_subset = masked_out[_y*num_px:(_y+1)*num_px,_x*num_px:(_x+1)*num_px] == False


                    inner_term =  np.power(abununcert[_y*num_px:(_y+1)*num_px,_x*num_px:(_x+1)*num_px,:][valid_subset,:] / \
                                           abun[_y*num_px:(_y+1)*num_px,_x*num_px:(_x+1)*num_px,:][valid_subset,:],2) +\
                                  np.power(coveruncert[_y*num_px:(_y+1)*num_px,_x*num_px:(_x+1)*num_px][valid_subset] / \
                                           cover[_y*num_px:(_y+1)*num_px,_x*num_px:(_x+1)*num_px][valid_subset],2)[:,np.newaxis]
                    inner_term[abun[_y*num_px:(_y+1)*num_px,_x*num_px:(_x+1)*num_px,:][valid_subset,:] == 0] = np.nan
                    

                    asa_unc[_y,_x,:] = np.sqrt(np.power(asa[_y,_x,:] / valid_px,2) * np.nansum(inner_term,axis=0))

        # Spectral Abundance
        outDataset = driver.Create(args.coarsened_file, asa.shape[1], asa.shape[0],
                                   asa.shape[2], gdal.GDT_Float32, options=['INTERLEAVE=BIL'])
        outDataset.SetProjection(abun_gdal.GetProjection())
        outtrans = list(abun_gdal.GetGeoTransform())
        outtrans[1] = args.resolution
        outtrans[5] = -1*args.resolution
        outDataset.SetGeoTransform(outtrans)
        for _b in range(1, asa.shape[2]+1):
            outDataset.GetRasterBand(_b).SetNoDataValue(-9999)
            if band_names is not None:
                outDataset.GetRasterBand(_b).SetDescription(band_names[_b-1])
        del outDataset
        _write_bil_chunk(asa.transpose((0,2,1)), args.coarsened_file, 0, (asa.shape[0], asa.shape[2], asa.shape[1]))


        # Count fractions
        if args.mask_fraction_file is not None:
            outDataset = driver.Create(args.mask_fraction_file, agg_count.shape[1], agg_count.shape[0],
                                       agg_count.shape[2], gdal.GDT_Float32, options=['INTERLEAVE=BIL'])
            outDataset.SetProjection(abun_gdal.GetProjection())
            outtrans = list(abun_gdal.GetGeoTransform())
            outtrans[1] = args.resolution
            outtrans[5] = -1*args.resolution
            outDataset.SetGeoTransform(outtrans)
            for _b in range(1, agg_count.shape[2]+1):
                outDataset.GetRasterBand(_b).SetNoDataValue(-9999)
                if band_names is not None:
                    outDataset.GetRasterBand(_b).SetDescription(list(counts.keys())[_b-1])
            del outDataset
            _write_bil_chunk(agg_count.transpose((0,2,1)), args.mask_fraction_file, 0, (agg_count.shape[0], agg_count.shape[2], agg_count.shape[1]))

        # Now uncertainty
        if do_uncert:
            outDataset = driver.Create(args.coarsened_file + '_uncert', asa.shape[1], asa.shape[0],
                                       asa.shape[2], gdal.GDT_Float32, options=['INTERLEAVE=BIL'])
            outDataset.SetProjection(abun_gdal.GetProjection())
            outtrans = list(abun_gdal.GetGeoTransform())
            outtrans[1] = args.resolution
            outtrans[5] = -1*args.resolution
            outDataset.SetGeoTransform(outtrans)
            for _b in range(1, asa_unc.shape[2]+1):
                outDataset.GetRasterBand(_b).SetNoDataValue(-9999)
                if band_names is not None:
                    outDataset.GetRasterBand(_b).SetDescription(band_names[_b-1])
            del outDataset
            _write_bil_chunk(asa_unc.transpose((0,2,1)), args.coarsened_file + '_uncert', 0, (asa_unc.shape[0], asa_unc.shape[2], asa_unc.shape[1]))

                   

       


if __name__ == '__main__':
    main()



