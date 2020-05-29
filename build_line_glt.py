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


def main():
    parser = argparse.ArgumentParser(description='Integrate multiple GLTs with a mosaicing rule')
    parser.add_argument('igm_filename')
    parser.add_argument('output_glt_filename')
    parser.add_argument('target_resolution', nargs=2, type=float)
    parser.add_argument('reference_point', nargs=2, type=float, help='location to align grid with')
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
    logging.info('Starting build_line_glt, arguments given as: {}'.format(args))
    emit_utils.common_logs.logtime()


