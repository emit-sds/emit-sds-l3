"""
Build a multi-file per-pixel spatial reference.

Author: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov
"""

import argparse
import numpy as np
import pandas as pd
from osgeo import gdal, osr
import math
import logging
import multiprocessing
from typing import List
import scipy.spatial.distance

import emit_utils.file_checks
import emit_utils.multi_raster_info
import emit_utils.common_logs

GLT_NODATA_VALUE=-9999
IGM_NODATA_VALUE=-9999
CRITERIA_NODATA_VALUE=-9999

#TODO: add logging

def main():
    parser = argparse.ArgumentParser(description='Integrate multiple GLTs with a mosaicing rule')
    parser.add_argument('output_filename')
    parser.add_argument('target_resolution', nargs=2, type=float)
    parser.add_argument('-target_extent_ul_lr', nargs=4, type=float)
    parser.add_argument('-glt_file_list')
    parser.add_argument('-igm_file_list')
    parser.add_argument('-criteria_file_list')
    parser.add_argument('-criteria_nodata', type=float, default=None)
    parser.add_argument('-criteria_band', type=int, default=1)
    parser.add_argument('-n_cores', type=int, default=-1)
    parser.add_argument('-log_file', type=str, default=None)
    parser.add_argument('-log_level', type=str, default='INFO')
    parser.add_argument('-glt_files', nargs='+', type=str,
                        help='a space-separated list of the input glt files.  Glob tokens accepted')
    parser.add_argument('-igm_files', nargs='+', type=str,
                        help='a space-separated list of the input igm files.  Glob tokens accepted')
    args = parser.parse_args()

    # Set up logging per arguments
    if args.log_file is None:
        logging.basicConfig(format='%(message)s', level=args.log_level)
    else:
        logging.basicConfig(format='%(message)s', level=args.log_level, filename=args.log_file)

    # Log the current time
    logging.info('Starting build_mosaic_glt, arguments given as: {}'.format(args))
    emit_utils.common_logs.logtime()

    if int(args.glt_file_list is None) + int(args.igm_file_list is None) + int(args.glt_files is None) + \
        int(args.igm_files is None) != 3:
        raise IOError('One and only one of the arguments: [glt_file_list, igm_file_list, glt_files, '
                      'igm_files] is accepted')

    # Get input files from file list
    #TODO: handle multiple types of glt inputs
    if args.glt_file_list is not None:
        logging.info('Checking input GLT files')
        glt_files = np.squeeze(np.array(pd.read_csv(args.glt_file_list)))
        emit_utils.file_checks.check_raster_files(glt_files, map_space=True)

    #TODO: handle multiple types of igm inputs
    if args.igm_file_list is not None:
        logging.info('Checking input IGM files')
        igm_files = np.squeeze(np.array(pd.read_csv(args.igm_file_list)))
        emit_utils.file_checks.check_raster_files(igm_files, map_space=False)

    if args.criteria_file_list is not None:
        logging.info('Checking input criteria files')
        criteria_files = np.squeeze(np.array(pd.read_csv(args.criteria_file_list)))
        emit_utils.file_checks.check_raster_files(criteria_files, map_space=False)
    else:
        criteria_files = None


    min_x_map, max_y_map, max_x_map, min_y_map, file_min_xy, file_max_xy = emit_utils.\
        multi_raster_info.get_bounding_extent_igms(igm_files, return_per_file_xy=True)

    if args.target_extent_ul_lr is not None:
        min_x_map = args.target_extent_ul_lr[0]
        max_x_map = args.target_extent_ul_lr[2]

        max_y_map = args.target_extent_ul_lr[1]
        min_y_map = args.target_extent_ul_lr[3]
        logging.info('Revised min xy: {}, max xy: {}'.format((min_x_map, min_y_map),(max_x_map, max_y_map)))

    #TODO: place check on target_resolution input argument prior to this point
    x_size_px = int(math.ceil((max_x_map - min_x_map) / float(args.target_resolution[0])))
    y_size_px = int(math.ceil((max_y_map - min_y_map) / float(-args.target_resolution[1])))

    logging.info('Output map size (y,x): {}, {}'.format(y_size_px,x_size_px))
        


    #TODO: insert any appropriate tapping or grid-subsetting HERE

    # Build output dataset
    driver = gdal.GetDriverByName('ENVI')
    driver.Register()

    geotransform = [min_x_map, args.target_resolution[0], 0, max_y_map, 0, args.target_resolution[1]]

    outDataset = driver.Create(args.output_filename, x_size_px, y_size_px, 3, gdal.GDT_Int32, options=['INTERLEAVE=BIL'])
    outDataset.SetGeoTransform(geotransform)
    #TODO: allow projection as input argument

    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS("EPSG:4326")
    outDataset.SetProjection(srs.ExportToWkt())
    del outDataset

    if args.n_cores == -1:
        args.n_cores = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(processes=args.n_cores)
    results = []
    for idx_y in range(y_size_px):
        if args.n_cores == 1:
            construct_mosaic_glt_from_igm_line(args.output_filename, geotransform, igm_files, criteria_files, (x_size_px, y_size_px), file_min_xy,
                                               file_max_xy, idx_y)
        else:
            results.append(pool.apply_async(construct_mosaic_glt_from_igm_line,
                                            args=(args.output_filename, geotransform, igm_files, criteria_files, (x_size_px, y_size_px), file_min_xy,
                                                file_max_xy, idx_y
                                                ,)))
    if args.n_cores != 1:
        results = [p.get() for p in results]
    pool.close()
    pool.join()


    """
    ############ uniform GLT-grid assumption - deprecating, but preserving till speed tests are complete ########
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
            construct_mosaic_glt_line(args.output_filename, glt_files, criteria_files, (x_size_px, y_size_px), px_offsets, lr_px_coords, idx_y)
        else:
            results.append(pool.apply_async(construct_mosaic_glt_line,
                                            args=(args.output_filename, glt_files, criteria_files, (x_size_px, y_size_px), px_offsets, lr_px_coords, idx_y,)))
    if args.n_cores != 1:
        results = [p.get() for p in results]
    pool.close()
    pool.join()
    """



def construct_mosaic_glt_from_igm_line(output_file: str, output_geotransform: tuple, igm_files: np.array, criteria_files: np.array, size_px: tuple,
                              file_min_xy: List, file_max_xy: List, line_index: int, criteria_band: int = 1):
    """
    Build one line of a mosaic GLT.  Must be able to hold entire line in memory space.
    Args:
        output_file: output 3-band mosaic glt file, presumed to already be initialized
        output_geotransform: gdal-style geotranform of output file
        igm_files: glt files (3-band images of (x,y,x) map-space coordinates in raw-space
        criteria_files: files to use to determine how to order overlaps, or None to do an naive implementation
        size_px: mosaic glt file size in pixels (x,y)
        file_min_xy: coordinate-pair map-space mins of each file
        file_max_xy: coordinate-pair map-space maxs of each file
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

    #line_distance = np.zeros((size_px[0], 3))
    #line_distance[...] = np.nan

    # Find min and max map-space y values for this line
    line_min_y = output_geotransform[3] + (line_index + 1)*output_geotransform[5]
    line_max_y = output_geotransform[3]

    map_match_centers = np.zeros((2,1,size_px[0]))
    map_match_centers[0,...] = output_geotransform[0] + (np.arange(size_px[0])+0.5) * output_geotransform[1]
    map_match_centers[1,...] = output_geotransform[3] + (line_index + 0.5) * output_geotransform[5]

    # Start a loop through each file
    for file_index, (igm_file, f_min_xy, f_max_xy) in enumerate(zip(igm_files, file_min_xy, file_max_xy)):

        # No need to proceed with this file if it's out of bounds
        if f_max_xy[1] < line_min_y or f_min_xy[1] > line_max_y:
            continue

        #TODO: Do x checking as well

        # unfortunately, we have to read the whole file in now.
        igm = gdal.Open(igm_file, gdal.GA_ReadOnly).ReadAsArray()

        # Get source-locations of this file
        valid_igm = np.all(igm != IGM_NODATA_VALUE, axis=0)
        igm[:,np.logical_not(valid_igm)] = np.nan

        closest_x_px, closest_y_px, closest_distance = match_map_centers(igm, map_match_centers, y_bounds = (line_min_y, line_max_y))
        #TODO: make distance threshold an argument
        close_enough = closest_distance < 2*output_geotransform[1]
        closest_x_px = closest_x_px[close_enough]
        closest_y_px = closest_y_px[close_enough]
        closest_distance = closest_distance[close_enough]

        # Only procede if we have some points to use
        if len(closest_x_px) > 0:

            # Get criteria values of this file (if provided)
            if criteria_files is not None:
                min_glt_y = np.min(closest_y_px)
                max_glt_y = np.max(closest_y_px)

                # Open up the criteria dataset
                criteria_dataset = gdal.Open(criteria_files[file_index], gdal.GA_ReadOnly)

                # Read in the block of data necessary to get the criteria
                criteria_block = np.zeros((max_glt_y - min_glt_y, criteria_dataset.RasterXSize))
                for gltindex in range(min_glt_y, max_glt_y):
                    criteria_block[gltindex-min_glt_y, :] = np.squeeze(criteria_dataset.ReadAsArray(0, gltindex,
                                                                    criteria_dataset.RasterXSize, 1)[criteria_band,...])

                # Now get the specific info we want
                # TODO: careful about criteria datatype
                criteria = criteria_block[closest_y_px - min_glt_y, closest_x_px].astype(np.float32)
                criteria[criteria == CRITERIA_NODATA_VALUE] = np.nan

                # Now we can determine priority
                current_line_superior = np.logical_or(criteria > line_criteria, np.isnan(line_criteria))

                # copy in new values
                line_criteria[current_line_superior] = criteria
                line_glt[current_line_superior, 0] = closest_x_px[current_line_superior]
                line_glt[current_line_superior, 1] = closest_y_px[current_line_superior]
                line_glt[current_line_superior, 2] = file_index
            else:
                line_glt[close_enough, 0] = closest_x_px
                line_glt[close_enough, 1] = closest_y_px
                line_glt[close_enough, 2] = file_index

    # Loop is finished, convert nans to nodata and write the output
    # TODO: careful about output datatype
    line_glt[np.isnan(line_glt)] = GLT_NODATA_VALUE
    output_memmap = np.memmap(output_file, mode='r+', shape=(size_px[1], 3, size_px[0]), dtype=np.int32)
    output_memmap[line_index, ...] = np.transpose(line_glt)
    logging.info('Completed line {}/{}'.format(line_index,size_px[1]))
    del output_memmap


def match_map_centers(igm_input, map_match_centers, y_bounds=None):
    """
    Find closes px coordinate matches between an igm and a grid of locations in glt_format
    Args:
        igm_input: array of igm point locations in map space
        map_match_centers: locations to match the igm to
        y_bounds: pair map-space (y_min, y_max) values to crop the igm search with
    Return:
        closest_x_px: array of closest x-pixels
        closest_y_px: array of closest y-pixels
        closest_distance: distance between the igm and map centers for each of the closest pixels
    """
    igm_shape = igm_input.shape
    if y_bounds is None:
        igm_xy = np.hstack([igm_input[0,...].flatten().reshape(-1,1), igm_input[1,...].flatten().reshape(-1,1)])
        y_offset = 0
    else:
        subset = np.logical_and(igm_input[1, ...] > y_bounds[0], igm_input[1, ...] < y_bounds[1])
        igm_xy = np.hstack([igm_input[0, subset].flatten().reshape(-1, 1), igm_input[1, subset].flatten().reshape(-1,1)])


    map_centers_xy = np.hstack([map_match_centers[0,...].flatten().reshape(-1,1), map_match_centers[1,...].flatten().reshape(-1,1)])

    distance = scipy.spatial.distance.cdist(map_centers_xy, igm_xy, metric='euclidean')
    del map_centers_xy, igm_xy

    closest_idx = np.nanargmin(distance,axis=1)
    closest_distance = np.nanmin(distance,axis=1)
    del distance

    x_px = np.arange(igm_shape[2])[np.newaxis,:] * np.ones((igm_shape[1],igm_shape[2]))
    y_px = np.arange(igm_shape[1])[:,np.newaxis] * np.ones((igm_shape[1],igm_shape[2]))
    if y_bounds is not None:
        x_px = x_px[subset]
        y_px = y_px[subset]

    closest_x_px = x_px.flatten()[closest_idx]
    closest_y_px = y_px.flatten()[closest_idx]

    return closest_x_px, closest_y_px, closest_distance


def construct_mosaic_glt_line(output_file: str, glt_files: np.array, criteria_files: np.array, size_px: tuple,
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
            line_glt[current_line_superior, 2] = file_index

    # Loop is finished, convert nans to nodata and write the output
    # TODO: careful about output datatype
    line_glt[np.isnan(line_glt)] = GLT_NODATA_VALUE
    output_memmap = np.memmap(output_file, mode='r+', shape=(size_px[1], 3, size_px[0]), dtype=np.int32)
    output_memmap[line_index, ...] = np.transpose(line_glt)
    del output_memmap

if __name__ == "__main__":
    main()
