"""
Apply a (possibly multi-file) per-pixel spatial reference.

Author: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov
"""

import argparse
import numpy as np
import pandas as pd
import gdal
import logging
import multiprocessing
from typing import List
import time

import emit_utils

GLT_NODATA_VALUE=-9999
CRITERIA_NODATA_VALUE=-9999


def main():
    parser = argparse.ArgumentParser(description='Integrate multiple GLTs with a mosaicing rule')
    parser.add_argument('glt_file')
    parser.add_argument('rawspace_file', desc='filename of rawspace source file or, in the case of a mosaic_glt, a text-file list of raw space files')
    parser.add_argument('output_filename')
    parser.add_argument('-band_numbers', nargs='+', default=-1, desc='list of 0-based band numbers, or -1 for all')
    parser.add_argument('-n_cores', type=int, default=-1)
    parser.add_argument('-log_file', type=str, default=None)
    parser.add_argument('-log_level', type=str, default='INFO')
    args = parser.parse_args()

    # Set up logging per arguments
    if args.log_file is None:
        logging.basicConfig(format='%(message)s', level=args.log_level)
    else:
        logging.basicConfig(format='%(message)s', level=args.log_level, filename=args.log_file)

    # Log the current time
    logging.info('Starting apply_glt, arguments given as: {}'.format(args))
    emit_utils.common_logs.logtime()

    # Do some checks on input raster files
    emit_utils.file_checks.check_raster_files(args.glt_file, map_space=True)

    # Open the GLT dataset
    glt_dataset = gdal.Open(args.glt_file, gdal.GA_ReadOnly)

    is_mosaic = glt_dataset.RasterCount == 3
    logging.info('GLT is a 3-band file, running in mosaic mode.')

    if is_mosaic:
        rawspace_files = np.squeeze(np.array(pd.read_csv(args.rawspace_file)))
        emit_utils.file_checks.check_raster_files(rawspace_files, map_space=False)
        # TODO: check that all rawspace files have same number of bands
    else:
        emit_utils.file_checks.check_raster_files([args.rawspace_file], map_space=False)
        args.rawspace_file = [args.rawspace_file]

    # TODO: consider adding check for the right number of rawspace_files - requires
    # reading the GLT through, which isn't free

    first_file_dataset = gdal.Open(args.rawspace_file[0], gdal.GA_ReadOnly)

    if args.bands_numbers == -1:
        output_bands = np.array(first_file_dataset.RasterCount)
    else:
        output_bands = np.array(args.band_numbers)

    # Build output dataset
    driver = gdal.GetDriverByName('ENVI')
    driver.Register()

    #TODO: careful about output datatypes / format
    outDataset = driver.Create(args.output_filename, glt_dataset.RasterXSize, glt_dataset.RasterYSize,
                               len(output_bands), gdal.GDT_Float32, options=['INTERLEAVE=BIL'])
    outDataset.SetProjection(glt_dataset.GetProjection())
    outDataset.SetGeotransform(glt_dataset.GetGeoTransform)
    del outDataset

    pool = multiprocessing.Pool(processes=args.n_cores)
    results = []
    for idx_y in range(glt_dataset.RasterYSize):
        if args.n_cores == 1:
            apply_mosaic_glt_line(args.glt_file, args.output_filename, args.rawspace_file, idx_y)
        else:
            results.append(pool.apply_async(apply_mosaic_glt_line, args=(args.glt_file, args.output_filename, args.arwspace_file, idx_y)))
    if args.n_cores != 1:
        results = [p.get() for p in results]
    pool.close()
    pool.join()

    # Log final time and exit
    logging.info('GLT application complete, output available at: {}'.format(args.output_filename))
    emit_utils.common_logs.logtime()


def apply_mosaic_glt_line(glt_filename: str, output_filename: str, rawspace_files: List, output_bands: np.array,
                          line_index: int):
    """
    Create one line of an output mosaic in mapspace
    Args:
        glt_filename: pre-built single or mosaic glt
        output_filename: output destination, assumed to location where a pre-initialized raster exists
        rawspace_files: list of rawspace input locations
        output_bands: array-like of bands to use from the rawspace file in the output
        line_index: line of the glt to process
    Returns:
        None
    """

    glt_dataset = gdal.Open(glt_filename, gdal.GA_ReadOnly)

    if line_index % 1000:
        logging.info('Beginning application of line {}/{}'.format(line_index, glt_dataset.RasterYSize))

    glt_line = glt_dataset.ReadAsArray(0, line_index, glt_dataset.RasterYSize, 1)
    valid_glt = np.all(glt_line != GLT_NODATA_VALUE, axis=0)

    if np.sum(valid_glt) == 0:
        return

    if glt_line.shape[0] == 3:
        necessary_file_idxs = np.unique(glt_line[2,valid_glt])
    else:
        necessary_file_idxs = 0

    for file_index in necessary_file_idxs:

        if glt_line.shape[0] == 3:
            pixel_subset = glt_line[2, ...] == file_index
        else:
            pixel_subset = valid_glt.copy()

        min_glt_y = np.min(glt_line[1, pixel_subset])
        max_glt_y = np.max(glt_line[1, pixel_subset])

        # Open up the criteria dataset
        rawspace_dataset = gdal.Open(rawspace_files[file_index], gdal.GA_ReadOnly)

        # Read in the block of data necessary to get the criteria
        rawspace_block = np.zeros((max_glt_y - min_glt_y, rawspace_dataset.RasterXSize))
        for gltindex in range(min_glt_y, max_glt_y):
            direct_read = np.squeeze(rawspace_dataset.ReadAsArray(0, gltindex,rawspace_dataset.RasterXSize,1))

            # account for extra bands, if this isn't a single-band dataset (gdal drops the 0th dimension if it is)
            if glt_dataset.RasterCount > 0:
                direct_read = direct_read[output_bands,...]

            # assign data to block
            rawspace_block[gltindex - min_glt_y, :] = direct_read

        # convert rawspace to mapspace through lookup
        mapspace_line = rawspace_block[glt_line[0, pixel_subset].flatten(), glt_line[1, pixel_subset].flatten() - min_glt_y]

        # write the output
        #TODO: careful about output datatype
        output_memmap = np.memmap(output_filename, mode='r+', shape=(glt_dataset.RasterXSize, len(output_bands), glt_dataset.RasterYSize), dtype=np.float32)
        output_memmap[line_index, pixel_subset] = np.transpose(mapspace_line)
        del output_memmap


if __name__ == "__main__":
    main()
