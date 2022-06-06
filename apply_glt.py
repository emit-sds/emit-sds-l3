"""
Apply a (possibly multi-file) per-pixel spatial reference.

Author: Philip G. Brodrick, philip.brodrick@jpl.nasa.gov
"""

import argparse
import numpy as np
import pandas as pd
from osgeo import gdal
from spectral.io import envi
import logging
import ray
from typing import List
import time
import os
import multiprocessing

import emit_utils.common_logs
import emit_utils.file_checks
import emit_utils.multi_raster_info
from emit_utils.file_checks import envi_header

GLT_NODATA_VALUE=-9999
#GLT_NODATA_VALUE=0
CRITERIA_NODATA_VALUE=-9999



def main():
    parser = argparse.ArgumentParser(description='Integrate multiple GLTs with a mosaicing rule')
    parser.add_argument('glt_file')
    parser.add_argument('rawspace_file', help='filename of rawspace source file or, in the case of a mosaic_glt, a text-file list of raw space files')
    parser.add_argument('output_filename')
    parser.add_argument('-band_numbers', nargs='+', type=int, default=-1, help='list of 0-based band numbers, or -1 for all')
    parser.add_argument('-n_cores', type=int, default=-1)
    parser.add_argument('-log_file', type=str, default=None)
    parser.add_argument('-log_level', type=str, default='INFO')
    parser.add_argument('-run_with_missing_files', type=int, default=0, choices=[0,1])
    parser.add_argument('-ip_head', type=str)
    parser.add_argument('-redis_password', type=str)
    parser.add_argument('-one_based_glt', type=int, choices=[0,1], default=0)
    parser.add_argument('-mosaic', type=int, choices=[0,1], default=0)
    args = parser.parse_args()

    # Set up logging per arguments
    if args.log_file is None:
        logging.basicConfig(format='%(message)s', level=args.log_level)
    else:
        logging.basicConfig(format='%(message)s', level=args.log_level, filename=args.log_file)

    args.one_based_glt = args.one_based_glt == 1
    args.run_with_missing_files = args.run_with_missing_files == 1
    args.mosaic = args.mosaic == 1

    # Log the current time
    logging.info('Starting apply_glt, arguments given as: {}'.format(args))
    emit_utils.common_logs.logtime()

    # Do some checks on input raster files
    #emit_utils.file_checks.check_raster_files([args.glt_file], map_space=True)

    # Open the GLT dataset
    glt_dataset = gdal.Open(args.glt_file, gdal.GA_ReadOnly)
    glt = envi.open(envi_header(args.glt_file)).open_memmap(writeable=False, interleave='bip')


    if args.mosaic:
        rawspace_files = np.squeeze(np.array(pd.read_csv(args.rawspace_file, header=None)))
        # TODO: make this check more elegant, should run, catch all files present exception, and proceed
        if args.run_with_missing_files is False:
            emit_utils.file_checks.check_raster_files(rawspace_files, map_space=False)
        # TODO: check that all rawspace files have same number of bands

    else:
        emit_utils.file_checks.check_raster_files([args.rawspace_file], map_space=False)
        rawspace_files = [args.rawspace_file]

    # TODO: consider adding check for the right number of rawspace_files - requires
    # reading the GLT through, which isn't free


    band_names = None
    for _ind in range(len(rawspace_files)):
        first_file_dataset = gdal.Open(rawspace_files[_ind], gdal.GA_ReadOnly)
        if first_file_dataset is not None:
            if 'band names' in envi.open(envi_header(rawspace_files[_ind])).metadata.keys():
                if args.band_numbers != -1:
                    band_names = [x for _x, x in enumerate(envi.open(envi_header(rawspace_files[_ind])).metadata['band names']) if _x in args.band_numbers]
                else:
                    band_names = envi.open(envi_header(rawspace_files[_ind])).metadata['band names']
                break
            else:
                band_names = [f'Band {x}' for x in range(first_file_dataset.RasterCount)]
                if args.band_numbers != -1:
                    band_names = [x for _x, x in enumerate(band_names) if _x in args.band_numbers]

    if args.band_numbers == -1:
        output_bands = np.arange(first_file_dataset.RasterCount)
    else:
        output_bands = np.array(args.band_numbers)

    # Build output dataset
    driver = gdal.GetDriverByName('ENVI')
    driver.Register()

    #TODO: careful about output datatypes / format
    outDataset = driver.Create(args.output_filename, glt.shape[1], glt.shape[0],
                               len(output_bands), gdal.GDT_Float32, options=['INTERLEAVE=BIL'])
    outDataset.SetProjection(glt_dataset.GetProjection())
    outDataset.SetGeoTransform(glt_dataset.GetGeoTransform())
    for _b in range(1, len(output_bands)+1):
        outDataset.GetRasterBand(_b).SetNoDataValue(-9999)
        if band_names is not None:
            outDataset.GetRasterBand(_b).SetDescription(band_names[_b-1])

    del outDataset

    if args.n_cores == -1:
        args.n_cores = multiprocessing.cpu_count()

    rayargs = {'address': args.ip_head,
               '_redis_password': args.redis_password,
               'local_mode': args.n_cores == 1}
    if args.n_cores < 40:
        rayargs['num_cpus'] = args.n_cores
    ray.init(**rayargs)
    print(ray.cluster_resources())

    jobs = []
    for idx_y in range(glt.shape[0]):
        jobs.append(apply_mosaic_glt_line.remote(args.glt_file, 
                                                 args.output_filename, 
                                                 rawspace_files, 
                                                 output_bands, 
                                                 idx_y,
                                                 args))
    rreturn = [ray.get(jid) for jid in jobs]
    ray.shutdown()

    #if args.n_cores == -1:
    #    args.n_cores = multiprocessing.cpu_count()
    #pool = multiprocessing.Pool(processes=args.n_cores)
    #results = []
    #for idx_y in range(glt_dataset.RasterYSize):
    #    if args.n_cores == 1:
    #        apply_mosaic_glt_line(args.glt_file, args.output_filename, rawspace_files, output_bands, idx_y)
    #    else:
    #        results.append(pool.apply_async(apply_mosaic_glt_line, args=(args.glt_file, args.output_filename, rawspace_files, output_bands, idx_y)))
    #if args.n_cores != 1:
    #    results = [p.get() for p in results]

    #for idx_f in range(len(rawspace_files)):
    #    if args.n_cores == 1:
    #        apply_mosaic_glt_image(args.glt_file, args.output_filename, rawspace_files[idx_f], idx_f, output_bands)
    #    else:
    #        results.append(pool.apply_async(apply_mosaic_glt_image, args=(args.glt_file, args.output_filename, rawspace_files[idx_f], idx_f, output_bands)))
    #if args.n_cores != 1:
    #    results = [p.get() for p in results]

    #pool.close()
    #pool.join()

    # Log final time and exit
    logging.info('GLT application complete, output available at: {}'.format(args.output_filename))
    emit_utils.common_logs.logtime()


def _write_bil_chunk(dat: np.array, outfile: str, line: int, shape: tuple, dtype: str = 'float32') -> None:
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


@ray.remote
def apply_mosaic_glt_line(glt_filename: str, output_filename: str, rawspace_files: List, output_bands: np.array,
                          line_index: int, args: List):
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

    logging.basicConfig(format='%(message)s', level=args.log_level, filename=args.log_file)

    glt_dataset = envi.open(envi_header(glt_filename))
    glt = glt_dataset.open_memmap(writeable=False, interleave='bip')

    if line_index % 100 == 0:
        logging.info('Beginning application of line {}/{}'.format(line_index, glt.shape[0]))

    #glt_line = glt_dataset.ReadAsArray(0, line_index, glt_dataset.RasterXSize, 1)
    #glt_line = glt[0][:,line_index:line_index+1, :]

    glt_line = np.squeeze(glt[line_index,...]).copy().astype(int)
    valid_glt = np.all(glt_line != GLT_NODATA_VALUE, axis=-1)

    glt_line[valid_glt,1] = np.abs(glt_line[valid_glt,1]) 
    glt_line[valid_glt,0] = np.abs(glt_line[valid_glt,0]) 
    glt_line[valid_glt,-1] = glt_line[valid_glt,-1]

    if args.one_based_glt:
        glt_line[valid_glt,:] = glt_line[valid_glt,:] - 1

    if np.sum(valid_glt) == 0:
        return

    
    if args.mosaic:
        un_file_idx = np.unique(glt_line[valid_glt,-1])
    else:
        un_file_idx = [0]

    output_dat = np.zeros((glt.shape[1],len(output_bands)),dtype=np.float32) - 9999
    for _idx in un_file_idx:
        if os.path.isfile(rawspace_files[_idx]):
            rawspace_dataset = envi.open(envi_header(rawspace_files[_idx]))
            rawspace_dat = rawspace_dataset.open_memmap(interleave='bip')

            if args.mosaic:
                linematch = np.logical_and(glt_line[:,-1] == _idx, valid_glt)
            else:
                linematch = valid_glt

            if np.sum(linematch) > 0:
                output_dat[linematch,:] = rawspace_dat[glt_line[linematch,1][:,None], glt_line[linematch,0][:,None],output_bands[None,:]].copy()


    _write_bil_chunk(np.transpose(output_dat), output_filename, line_index, (glt.shape[0], len(output_bands), glt.shape[1]))



    #for file_index in necessary_file_idxs:

    #    if glt_line.shape[0] == 3:
    #        pixel_subset = glt_line[..., -1] == file_index
    #    else:
    #        pixel_subset = valid_glt.copy()

    #    for ind in range(len(glt_line)):
    #        
    #    min_glt_y = np.min(glt_line[pixel_subset, 1])
    #    max_glt_y = np.max(glt_line[pixel_subset, 1])

    #    # Open up the criteria dataset
    #    #rawspace_dataset = gdal.Open(rawspace_files[file_index], gdal.GA_ReadOnly)
    #    if os.path.isfile(rawspace_files[file_index]): 

    #        for

    #        # Read in the block of data necessary to get the criteria
    #        rawspace_block = rawspace_dat[np.zeros((len(output_bands), max_glt_y - min_glt_y + 1, rawspace_dataset.RasterXSize))
    #        for gltindex in range(min_glt_y, max_glt_y+1):
    #            direct_read = np.squeeze(rawspace_dataset.ReadAsArray(0, gltindex,rawspace_dataset.RasterXSize,1))

    #            # TODO: account for extra bands, if this isn't a single-band dataset (gdal drops the 0th dimension if it is)
    #            if rawspace_dataset.RasterCount > 0:
    #                direct_read = direct_read[output_bands,...]

    #            # assign data to block
    #            rawspace_block[:, gltindex - min_glt_y, :] = direct_read

    #        # convert rawspace to mapspace through lookup
    #        mapspace_line = rawspace_block[:,glt_line[1, pixel_subset].flatten() - min_glt_y, glt_line[0, pixel_subset].flatten()]


    #        # write the output
    #        #TODO: careful about output datatype
    #        output_memmap = np.memmap(output_filename, mode='r+', shape=(glt[0].shape[1], len(output_bands), glt[0].shape[2]), dtype=np.float32)
    #        output_memmap[line_index, :, np.squeeze(pixel_subset)] = np.transpose(mapspace_line)
    #        del output_memmap



def apply_mosaic_glt_image(glt_filename: str, output_filename: str, rawspace_file: str, rawspace_file_index: int, output_bands: np.array):
    """
    Apply glt to one files worth of raw-space data, suitable for instances with large numbers of files.
    Args:
        glt_filename: pre-built single or mosaic glt
        output_filename: output destination, assumed to location where a pre-initialized raster exists
        rawspace_file: list of rawspace input locations
        rawspace_file_index: index of rawspace file in the mosaic_glt
        output_bands: array-like of bands to use from the rawspace file in the output
    Returns:
        None
    """

    # Open up the criteria dataset
    rawspace_dataset = gdal.Open(rawspace_file, gdal.GA_ReadOnly)
    if rawspace_dataset is None:
        return

    rawspace = np.zeros((rawspace_dataset.RasterCount, rawspace_dataset.RasterYSize, rawspace_dataset.RasterXSize))
    rawspace = np.memmap(rawspace_file, mode='r', shape=(rawspace_dataset.RasterYSize, rawspace_dataset.RasterCount, rawspace_dataset.RasterXSize), dtype=np.float32)
    #for line_index in range(rawspace.shape[-2]):
    #    rawspace[:,line_index:line_index+1,:] = rawspace_dataset.ReadAsArray(0,line_index,rawspace.shape[-1],rawspace.shape[-2])
    #    if (line_index % 100) == 0:
    #        print('Loading line: {}/{} of file {}'.format(line_index,rawspace.shape[-1],rawspace_file))
        
    logging.info('Successfully complted read of file {}'.format(rawspace_file))

    output_y_loc, output_x_loc = np.where(glt[0][2, ...] == rawspace_file_index+1)
    print('{} pixel values identified'.format(len(output_y_loc)))

    rawspace_y_loc = np.abs(glt[0][0,output_y_loc,output_x_loc])
    rawspace_x_loc = np.abs(glt[0][1,output_y_loc,output_x_loc])
    #rawspace_y_loc = np.abs(glt[0][1,output_y_loc,output_x_loc]) - 1
    #rawspace_x_loc = np.abs(glt[0][0,output_y_loc,output_x_loc]) - 1

    order = np.argsort(output_y_loc)

    output_y_loc = output_y_loc[order]
    output_x_loc = output_x_loc[order]

    rawspace_y_loc = rawspace_y_loc[order]
    rawspace_x_loc = rawspace_x_loc[order]

    line_index = 0
    while line_index < len(rawspace_y_loc):
        chunk = output_y_loc == output_y_loc[line_index]

        output_memmap = np.memmap(output_filename, mode='r+', shape=(glt[0].shape[-2], len(output_bands), glt[0].shape[-1]), dtype=np.float32)
        output_memmap[output_y_loc[chunk],:,output_x_loc[chunk]] = rawspace[rawspace_y_loc[chunk],:,rawspace_x_loc[chunk]][:,output_bands]
        del output_memmap

        if (line_index % int(len(rawspace_y_loc)/10.)) < np.sum(chunk).astype(int):
            logging.debug('File {}, {} % written'.format(rawspace_file_index, round(line_index/float(len(rawspace_y_loc))*100.,2) ))

        line_index += np.sum(chunk).astype(int)

    #logging.debug('Starting write')
    #output_memmap = np.memmap(output_filename, mode='r+', shape=(glt[0].shape[-2], len(output_bands), glt[0].shape[-1]), dtype=np.float32)
    #output_memmap[output_y_loc,:,output_x_loc] = rawspace[rawspace_y_loc,:,rawspace_x_loc]
    #del output_memmap
    #logging.debug('Write complete')




if __name__ == "__main__":
    main()
