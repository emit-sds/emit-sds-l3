"""
Build a multi-file per-pixel spatial reference.

Author: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov
"""

import argparse
import numpy as np
import pandas as pd
import gdal
import logging
import multiprocessing
from typing import List

import emit_utils

GLT_NODATA_VALUE=-9999
CRITERIA_NODATA_VALUE=-9999

#TODO: add logging

def main():
    parser = argparse.ArgumentParser(description='Integrate multiple GLTs with a mosaicing rule')
    parser.add_argument('output_filename')
    parser.add_argument('-glt_file_list')
    parser.add_argument('-criteria_file_list')
    parser.add_argument('-criteria_nodata', type=float, default=None)
    parser.add_argument('-criteria_band', type=int, default=1)
    parser.add_argument('-n_cores', type=int, default=-1)
    parser.add_argument('-log_file', type=str, default=None)
    parser.add_argument('-log_level', type=str, default='INFO')
    parser.add_argument('-glt_files', nargs='+', type=str,
                        help='a space-separated list of the input glt files.  Glob tokens accepted')
    args = parser.parse_args()

    # Set up logging per arguments
    if args.log_file is None:
        logging.basicConfig(format='%(message)s', level=args.log_level)
    else:
        logging.basicConfig(format='%(message)s', level=args.log_level, filename=args.log_file)

    # Get input files from file list
    glt_files = np.array(pd.read_csv(args.glt_file_list))
    if args.criteria_file_list is not None:
        criteria_files = np.array(pd.read_csv(args.criteria_file_list))
    else:
        criteria_files = None

    # Do some checks on input raster files
    emit_utils.file_checks.check_raster_files(glt_files, map_space=True)
    if criteria_files is not None:
        emit_utils.file_checks.check_raster_files(criteria_files, map_space=False)

    min_x_map, max_y_map, max_x_map, min_y_map, px_offsets, lr_px_coords = emit_utils.multi_raster_info.get_bounding_extent(glt_files,
                                                                                            return_pixel_offsets=True,
                                                                                            return_global_lower_rights=True)

    # Get some info about the first file, which we'll need shortly
    first_file_dataset = gdal.Open(glt_files[0], gdal.GA_ReadOnly)
    geotransform = first_file_dataset.GetGeoTransform()
    geotransform[0] = min_x_map
    geotransform[3] = max_y_map

    x_size_px = (max_x_map - min_x_map) / geotransform[1]
    y_size_px = (min_y_map - max_y_map) / geotransform[5]

    # Build output dataset
    driver = gdal.GetDriverByName('ENVI')
    driver.Register()

    # Create dataset to set matadata and reserve disk space.  Then delete (we'll memmap it out later)
    outDataset = driver.Create(args.output_filename, x_size_px, y_size_px, 3, gdal.GDT_Int32, options=['INTERLEAVE=BIL'])
    outDataset.SetProjection(first_file_dataset.GetProjection())
    outDataset.SetGeotransform(geotransform)
    del outDataset

    pool = multiprocessing.Pool(processes=args.n_cores)
    results = []
    for idx_y in range(y_size_px):
        if args.n_cores == 1:
            construct_mosaic_glt_line(glt_files, criteria_files, x_size_px, px_offsets, lr_px_coords, idx_y)
        else:
            results.append(pool.apply_async(construct_mosaic_glt_line,
                                            args=(glt_files, criteria_files, x_size_px, px_offsets, lr_px_coords, idx_y,)))
    if args.n_cores != 1:
        results = [p.get() for p in results]
    pool.close()
    pool.join()


def construct_mosaic_glt_line(output_file: str, glt_files: np.ndarray, criteria_files: np.nd_array, size_px: tuple,
                              px_offsets: List, lr_px_coords: List, line_index: int, criteria_band: int = 1):
    """
    Build one line of a mosaic GLT.  Must be able to hold entire line in memory space.
    Args:
        output_file: output 3-band mosaic glt file, presumed to already be initialized
        glt_files: glt files (2-band images of (x,y) raw-space coordinates in map-space)
        criteria_files: files to use to determine how to order overlaps, or None to do an naive implementation
        size_px: mosaic glt file size in pixels (x,y)
        px_offsets: coordinate-pair offsets from global upper-left for each glt file (which is local to its own raw-space)
        lr_px_coords: coordinate-pair of lr pixel locations in global space
        line_index: the line of the mosaic glt to build
        criteria_band: which band of the criteria file to use
    Returns:
        None
    """

    # initialize null best criteria values and glt values
    line_criteria = np.zeros(size_px[0])
    line_criteria[:] = np.nan

    line_glt = np.zeros((size_px[0], 3))
    line_glt[...] = np.nan

    # Start a loop through each file
    for file_index, (glt_file, px_offset, lr_px_coord) in enumerate(zip(glt_files, px_offsets, lr_px_coords)):

        # No need to proceed with this file if it's out of bounds
        if line_index < px_offset[1] or line_index > lr_px_coords[1]:
            continue

        # Get source-locations of this file
        glt_dataset = gdal.Open(glt_file, gdal.GA_ReadOnly)
        glt = glt_dataset.ReadAsArray(0, line_index - px_offset[1], glt_dataset.RasterXSize, 1)
        valid_glt = np.all(glt != GLT_NODATA_VALUE, axis=0)

        # account for the fact that negative glt values simply account for interpolation,
        # and should be cast as ingetgers
        glt = np.abs(glt).astype(int)

        # Get criteria values of this file (if provided)
        if criteria_files is not None:
            # Find range of glt values, so we know what block to read
            min_glt_x = np.min(glt[0, valid_glt])
            max_glt_x = np.max(glt[0, valid_glt])
            min_glt_y = np.min(glt[1, valid_glt])
            max_glt_y = np.max(glt[1, valid_glt])

            # Open up the criteria dataset
            criteria_dataset = gdal.Open(criteria_files[file_index], gdal.GA_ReadOnly)

            # Read in the block of data necessary to get the criteria
            criteria_block = np.zeros((max_glt_y - min_glt_y, criteria_dataset.RasterXSize))
            for gltindex in range(min_glt_y, max_glt_y):
                criteria_block[gltindex-min_glt_y, :] = np.squeeze(criteria_dataset.ReadAsArray(0, gltindex,
                                                                criteria_dataset.RasterXSize, 1)[criteria_band,...])

            # Now get the specific info we want
            criteria = criteria_block[glt[0, valid_glt].flatten(), glt[1, valid_glt].flatten() - min_glt_y].astype(np.float32)
            criteria[criteria == CRITERIA_NODATA_VALUE] = np.nan

            # Now we can determine priority
            current_line_superior = np.logical_or(criteria > line_criteria, np.isnan(line_criteria))

            # copy in new values
            line_criteria[current_line_superior] = criteria
            line_glt[current_line_superior, :2] = glt[current_line_superior]
            line_glt[current_line_superior, 3] = file_index

    # Loop is finished, convert nans to nodata and write the output
    # TODO: careful about output datatype
    line_glt[np.isnan(line_glt)] = GLT_NODATA_VALUE
    output_memmap = np.memmap(output_file, mode='r+', shape=(size_px[1], 3, size_px[0]), dtype=np.int32)
    output_memmap[line_index, ...] = np.transpose(line_glt)
    del output_memmap

if __name__ == "__main__":
    main()
