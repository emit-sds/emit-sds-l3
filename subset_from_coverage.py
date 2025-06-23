"""
Written by: Philip G. Brodrick
"""
import argparse
import numpy as np
import pandas as pd
import os
import logging
import subprocess
import json
from copy import deepcopy
from shapely.geometry import Polygon
from datetime import datetime


def roi_filter(coverage, roi):
    subdir = deepcopy(coverage)
    subdir['features'] = []
    target = Polygon(roi)
    for feat in coverage['features']:
        source = Polygon(feat['geometry']['coordinates'][0])
        if source.intersects(roi):
            subdir['features'].append(feat)
    return subdir
    

def time_filter(coverage, start_time, end_time):
    subdir = deepcopy(coverage)
    subdir['features'] = []
    for feat in coverage['features']:
      if start_time is None or datetime.strptime(feat['properties']['start_time'],'%Y-%m-%dT%H:%M:%SZ') >= start_time:
        if end_time is None or datetime.strptime(feat['properties']['end_time'],'%Y-%m-%dT%H:%M:%SZ') <= end_time:
          subdir['features'].append(feat)
    return subdir

#def roi_filter(coverage, roi):
#    subdir = deepcopy(coverage) 
#    cov_df = pd.json_normalize(coverage['features'])
#    cov_df['geometry.coordinates'] = cov_df['geometry.coordinates'].apply(lambda s: Polygon(s[0]) )
#    inds = np.where(cov_df['geometry.coordinates'].apply(lambda s,roi=roi: s.intersects(roi)))[0]
#    subdir['features'] = [subdir['features'][i] for i in inds]
#    return subdir
# 
#def time_filter(coverage, start_time, end_time):
#    subdir = deepcopy(coverage)
#    cov_df = pd.json_normalize(coverage['features'])
#    inds = np.where(np.logical_and(pd.to_datetime(cov_df['properties.start_time']) >= pd.to_datetime(start_time) , pd.to_datetime(cov_df['properties.end_time']) <= pd.to_datetime(end_time)))[0]
#    subdir['features'] = [subdir['features'][i] for i in inds]
#    return subdir



def cloud_filter(coverage, max_cloud_fraction):
    subdir = deepcopy(coverage)
    subdir['features'] = []
    for feat in coverage['features']:
        if 'Total Cloud Fraction' in feat['properties'] and feat['properties']['Total Cloud Fraction'] <= max_cloud_fraction:
          subdir['features'].append(feat)
    return subdir
    

def main():

    parser = argparse.ArgumentParser(description='Get subsets from coverage')
    parser.add_argument('--lat_bounds',type=float,default=[-89.9,89.9],nargs=2)
    parser.add_argument('--lon_bounds',type=float,default=[-179.9,179.9],nargs=2)
    parser.add_argument('--min_date',type=str,default=None)
    parser.add_argument('--max_date',type=str,default=None)
    parser.add_argument('--max_cloud_fraction',type=float,default=1)
    parser.add_argument('--coverage_file',type=str,default=None)
    parser.add_argument('--coverage_save_file',type=str)
    parser.add_argument('--fid_output_file',type=str,default=None)
    args = parser.parse_args()

    if args.coverage_file is None and args.coverage_save_file is None:
        raise AttributeError('Either coverage_file or coverage_save_file must be specified')


    if args.coverage_file is None:
        subprocess.call(f'wget -O {args.coverage_save_file} https://earth.jpl.nasa.gov/emit-mmgis-lb/Missions/EMIT/Layers/coverage/coverage_pub.json', shell=True)
        args.coverage_file = args.coverage_save_file

    coverage = json.load(open(args.coverage_file,'r'))

    ll_boundary = [\
                  [args.lon_bounds[0], args.lat_bounds[0]], 
                  [args.lon_bounds[1], args.lat_bounds[0]], 
                  [args.lon_bounds[1], args.lat_bounds[1]], 
                  [args.lon_bounds[0], args.lat_bounds[1]], 
                  [args.lon_bounds[0], args.lat_bounds[0]]\
                  ]

    ll_boundary = Polygon(ll_boundary)

    print(len(coverage['features']))
    filt_coverage = roi_filter(coverage, ll_boundary)
    print(len(filt_coverage['features']))

    if args.min_date is not None:
        args.min_date = datetime.strptime(args.min_date, '%Y%m%dT%H:%M:%S')

    if args.max_date is not None:
        args.max_date = datetime.strptime(args.max_date, '%Y%m%dT%H:%M:%S')

    filt_coverage = time_filter(filt_coverage, args.min_date, args.max_date)
    print(len(filt_coverage['features']))

    filt_coverage = cloud_filter(filt_coverage, args.max_cloud_fraction)
    print(len(filt_coverage['features']))
    
    if args.fid_output_file is not None:
        fids = np.array([x['properties']['fid'] for x in filt_coverage['features']], dtype=str)
        np.savetxt(args.fid_output_file, fids, fmt="%s")




if __name__ == "__main__":
    main()
