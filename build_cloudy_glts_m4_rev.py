

import datetime
import argparse
import glob
from spectral.io import envi
import numpy as np
import subprocess
import os
import pandas as pd
import filenames as fn
from osgeo import gdal

def runcall(cmdst, dryrun=False):
    print(cmdst)
    if dryrun is False:
        subprocess.call(cmdst,shell=True)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('lon_bounds', type=float, nargs=2)
    parser.add_argument('lat_bounds', type=float, nargs=2)
    parser.add_argument('--glt_dir', type=str)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--m4s', type=str, default='m4fc20qv12')
    args = parser.parse_args()

    # Modify with your own repo location
    emit_l3_base = '/beegfs/scratch/brodrick/emit/emit-sds-l3'

    path = os.environ['PATH']
    path = path.replace('\Library\\bin;',':')
    os.environ['PATH'] = path

    ll_str = f'{int(args.lon_bounds[0])}_{int(args.lon_bounds[1])}_{int(args.lat_bounds[0])}_{int(args.lat_bounds[1])}'

    fn.make_dirnames(args.glt_dir)
    abun_base = args.m4s
    fno = fn.filenames(args.glt_dir, ll_str, abun_base, 0.5)

    # List of file IDs to use
    fidlist = f'{args.glt_dir}/fids_{ll_str}_0.50_none_20240120.txt'

    # If needed, regenerate file IDs based on coverage .json downloaded from the website
    if os.path.isfile(fidlist) is False:
        subprocess.call(f'python {emit_l3_base}/subset_from_coverage.py --coverage_file {emit_l3_base}/visuals/track_coverage_pub.json --lon_bounds {args.lon_bounds[0]} {args.lon_bounds[1]} --lat_bounds {args.lat_bounds[0]} {args.lat_bounds[1]} --max_cloud_fraction 0.50 --max_date 20240120T00:00:00 --fid_output_file {fidlist}',shell=True)

    args.fidlist = fidlist 

    # If needed, regenerate a raster format erodible terrain file from the vector version
    if os.path.isfile(fno.erodible_vector_file) and os.path.isfile(fno.erodible_file) is False:
        if os.path.isfile(fno.l2b_abun_mosaic_file):
            ds = gdal.Open(fno.l2b_abun_mosaic_file)
            trans = ds.GetGeoTransform()
            cmd_str = f'gdal_rasterize -init 1 -burn 0 -tr {trans[1]} {trans[5]} -te {trans[0]} {trans[3]+ds.RasterYSize*trans[5]} {trans[0]+ds.RasterXSize*trans[1]} {trans[3]} -ot Byte -of GTiff -co COMPRESS=LZW {fno.erodible_vector_file} {fno.erodible_file}'
        else:
            cmd_str = f'gdal_rasterize -init 1 -burn 0 -tr 0.00055 -0.00055 -te {args.lon_bounds[0]} {args.lat_bounds[0]} {args.lon_bounds[1]} {args.lat_bounds[1]} -ot Byte -of GTiff -co COMPRESS=LZW {fno.erodible_vector_file} {fno.erodible_file}'
        runcall(cmd_str, args.dry_run)  

    if os.path.getsize(args.fidlist) == 0:
        quit()

    # List of all native-resolution files contributing to this aggregate file
    fids = open(args.fidlist,'r').readlines()
    fids = [x.strip() for x in fids]
    print(fids)

    # Get input files for each file ID
    igm_files = [fn.get_igm(x) for x in fids]
    obs_files = [fn.get_obs(x) for x in fids]
    mask_files = [fn.get_mask(x) for x in fids]
    l2b_files = [x.replace('l2a','l2b').replace('mask','abun') for x in mask_files]
    l2b_unc_files = [x.replace('l2a','l2b').replace('mask','abununcert') for x in mask_files]
    l3_files = [x.replace('l2a','l3').replace('mask','cover') for x in mask_files]
    l3_unc_files = [x.replace('l2a','l3').replace('mask','coveruncert') for x in mask_files]
    grainsize_files = [fno.grainsize_basepath + os.path.basename(x).split('_')[0] + '_grainsize' for x in mask_files]
    rfl_files = [x.replace('mask','rfl') for x in mask_files]

    np.savetxt(fno.igm_filelist, igm_files, fmt="%s")
    np.savetxt(fno.obs_filelist, obs_files, fmt="%s")
    np.savetxt(fno.mask_filelist, mask_files, fmt="%s")
    np.savetxt(fno.l2b_filelist, l2b_files, fmt="%s")
    np.savetxt(fno.l2b_unc_filelist, l2b_unc_files, fmt="%s")
    np.savetxt(fno.l3_filelist, l3_files, fmt="%s")
    np.savetxt(fno.l3_unc_filelist, l3_unc_files, fmt="%s")
    np.savetxt(fno.grainsize_filelist, grainsize_files, fmt="%s")
    np.savetxt(fno.rfl_filelist, rfl_files, fmt="%s")


    # If needed, build the mosaic Geographic Lookup Table (GLT) file
    if os.path.isfile(fno.glt_file) is False:
        cmd_str = f'julia --threads 4 {emit_l3_base}/build_mosaic_glt.jl {fno.glt_file} {fno.igm_filelist} 0.00055 0.00055 --mask_file_list {fno.mask_filelist} --target_extent_ul_lr " {args.lon_bounds[0]}" " {args.lat_bounds[1]}" " {args.lon_bounds[1]}" " {args.lat_bounds[0]}" --criteria_file_list {fno.obs_filelist} --criteria_band 5 --criteria_mode min'
        runcall(cmd_str, args.dry_run)  

    # If needed, build the coverage file
    if os.path.isfile(fno.glt_file) and os.path.isfile(fno.coverage_file) is False:
        cmd_str = f'python {emit_l3_base}/visuals/coverage_ql.py {fno.glt_file} {fno.coverage_file} --max_val 30'
        runcall(cmd_str, args.dry_run)  

    # If needed, build the grainsize mosaic file
    if os.path.isfile(fno.glt_file) and os.path.isfile(fno.l2b_grainsize_mosaic_file) is False:
        cmd_str = f'python {emit_l3_base}/apply_glt_serial.py {fno.glt_file} {fno.grainsize_filelist} {fno.l2b_grainsize_mosaic_file} --mosaic --run_with_missing_files'
        runcall(cmd_str, args.dry_run)

    # If needed, build the L2B mineral detection/depth mosaic files
    if os.path.isfile(fno.glt_file) and os.path.isfile(fno.l2b_mosaic_file) is False:
        cmd_str = f'python {emit_l3_base}/apply_glt_serial.py {fno.glt_file} {fno.l2b_filelist} {fno.l2b_mosaic_file} --mosaic --run_with_missing_files'
        runcall(cmd_str, args.dry_run)  

    # This is the fractional cover output
    if os.path.isfile(fno.glt_file) and os.path.isfile(fno.l3_mosaic_file) is False:
        cmd_str = f'python {emit_l3_base}/apply_glt_serial.py {fno.glt_file} {fno.l3_filelist} {fno.l3_mosaic_file} --mosaic --run_with_missing_files'
        runcall(cmd_str, args.dry_run)  

    # The mineral map uncertainties
    if os.path.isfile(fno.glt_file) and os.path.isfile(fno.l2b_unc_mosaic_file) is False:
        cmd_str = f'python {emit_l3_base}/apply_glt_serial.py {fno.glt_file} {fno.l2b_unc_filelist} {fno.l2b_unc_mosaic_file} --mosaic --run_with_missing_files'
        runcall(cmd_str, args.dry_run)  

    # The fractional cover uncertainties
    if os.path.isfile(fno.glt_file) and os.path.isfile(fno.l3_unc_mosaic_file) is False:
        cmd_str = f'python {emit_l3_base}/apply_glt_serial.py {fno.glt_file} {fno.l3_unc_filelist} {fno.l3_unc_mosaic_file} --mosaic --run_with_missing_files'
        runcall(cmd_str, args.dry_run)  

    if os.path.isfile(fno.l2b_mosaic_file) and os.path.isfile(fno.l2b_abun_mosaic_file) is False:

        if abun_base == 'm4fc20qv12':
            # This performs the abundance estimate
            cmd_str = f'julia m4p.jl {fno.l2b_abun_mosaic_base} {fno.l2b_mosaic_file} 12 1 --model_style m3 --abundance_metadata data/abundance_metadata_c20f.csv --quartz_size_file {fno.l2b_grainsize_mosaic_file} --quartz_file_scaling_factor 0.08043 --quartz_gs_lb 0.5'

        # Uncertainty runs once we clip to > 65% soil
        if abun_base == 'm4fc20qv12l65':
            # Lower uncertainty interval
            cmd_str = f'julia m4p.jl {fno.l2b_abun_mosaic_base} {fno.l2b_mosaic_file} 12 1 --model_style m3 --abundance_metadata data/abundance_metadata_c20f.csv --quartz_size_file {fno.l2b_grainsize_mosaic_file} --quartz_file_scaling_factor 0.08043 --quartz_gs_lb 0.5 --quartz_uncertainty_delta " -49.53"'
        if abun_base == 'm4fc20qv12u65':
            # Upper uncertainty interval
            cmd_str = f'julia m4p.jl {fno.l2b_abun_mosaic_base} {fno.l2b_mosaic_file} 12 1 --model_style m3 --abundance_metadata data/abundance_metadata_c20f.csv --quartz_size_file {fno.l2b_grainsize_mosaic_file} --quartz_file_scaling_factor 0.08043 --quartz_gs_lb 0.5 --quartz_uncertainty_delta " 80.01"'

        # Call Julia
        runcall(cmd_str, args.dry_run)  

    # Translate erodible vector file into erodible raster file
    if os.path.isfile(fno.erodible_vector_file) and os.path.isfile(fno.l2b_abun_mosaic_file) and os.path.isfile(fno.erodible_file) is False:
        ds = gdal.Open(fno.l2b_abun_mosaic_file)
        trans = ds.GetGeoTransform()
        cmd_str = f'gdal_rasterize -init 1 -burn 0 -tr {trans[1]} {trans[5]} -te {trans[0]} {trans[3]+ds.RasterYSize*trans[5]} {trans[0]+ds.RasterXSize*trans[1]} {trans[3]} -ot Byte -of GTiff -co COMPRESS=LZW {fno.erodible_vector_file} {fno.erodible_file}'
        runcall(cmd_str, args.dry_run)  

    # Now coarsen raw L3 products to 0.5 degree
    if os.path.isfile(fno.l3_mosaic_file) and os.path.isfile(fno.l3_coarse_mosaic_file) is False:
        cmd_str = f'gdalwarp -tr 0.5 -0.5 -of GTiff -co COMPRESS=LZW -r average {fno.l3_mosaic_file} {fno.l3_coarse_mosaic_file}'
        runcall(cmd_str, args.dry_run)  

    # Coarsen raw mineral abundances to 0.5 degree
    if os.path.isfile(fno.l2b_abun_mosaic_file) and os.path.isfile(fno.l2b_coarse_mosaic_file) is False:
        cmd_str = f'gdalwarp -tr 0.5 -0.5 -of GTiff -co COMPRESS=LZW -r average {fno.l2b_abun_mosaic_file} {fno.l2b_coarse_mosaic_file}'
        runcall(cmd_str, args.dry_run)  

    # Coarsen raw mineral abundances to 0.1 degree
    if os.path.isfile(fno.l2b_abun_mosaic_file) and os.path.isfile(fno.l2b_hd_mosaic_file) is False:
        cmd_str = f'gdalwarp -tr 0.1 -0.1 -of GTiff -co COMPRESS=LZW -r average {fno.l2b_abun_mosaic_file} {fno.l2b_hd_mosaic_file}'
        runcall(cmd_str, args.dry_run)  

    if os.path.isfile(fno.l2b_scatter_mosaic_file) and os.path.isfile(fno.scatter_coarse_mosaic_file) is False:
        cmd_str = f'gdalwarp -tr 0.5 -0.5 -of GTiff -co COMPRESS=LZW -r average {fno.l2b_scatter_mosaic_file} {fno.scatter_coarse_mosaic_file}'
        runcall(cmd_str, args.dry_run)  
    if os.path.isfile(fno.l2b_scatter_mosaic_file) and os.path.isfile(fno.scatter_hd_mosaic_file) is False:
        cmd_str = f'gdalwarp -tr 0.1 -0.1 -of GTiff -co COMPRESS=LZW -r average {fno.l2b_scatter_mosaic_file} {fno.scatter_hd_mosaic_file}'
        runcall(cmd_str, args.dry_run)  

    # Veg correction and masked aggregation with erodibility to 0.5 ang 0.1 degree sampling
    if os.path.isfile(fno.l2b_abun_mosaic_file) and os.path.isfile(fno.l3_mosaic_file) and os.path.isfile(fno.l2b_coarse_erod_mosaic_file) is False:
        cmd_str = f'python {emit_l3_base}/veg_correction.py {fno.l2b_abun_mosaic_file} {fno.l3_mosaic_file} --coarsened_file {fno.l2b_coarse_erod_mosaic_file} --resolution 0.5 --soil_thresh 0 --thresh_only --mask_file {fno.erodible_file}'
        runcall(cmd_str, args.dry_run)  
    if os.path.isfile(fno.l2b_abun_mosaic_file) and os.path.isfile(fno.l3_mosaic_file) and os.path.isfile(fno.l2b_hd_erod_mosaic_file) is False:
        cmd_str = f'python {emit_l3_base}/veg_correction.py {fno.l2b_abun_mosaic_file} {fno.l3_mosaic_file} --coarsened_file {fno.l2b_hd_erod_mosaic_file} --resolution 0.1 --soil_thresh 0 --thresh_only --mask_file {fno.erodible_file}'
        runcall(cmd_str, args.dry_run)  

    # Veg correction with soil threshold, masked aggregation to 0.1 and 0.5 degree 
    if os.path.isfile(fno.l2b_abun_mosaic_file) and os.path.isfile(fno.l3_mosaic_file) and os.path.isfile(fno.l2b_coarse_vegadj_thresh_mosaic_file) is False:
        cmd_str = f'python {emit_l3_base}/veg_correction.py {fno.l2b_abun_mosaic_file} {fno.l3_mosaic_file} --soil_thresh {fno.vegthresh} --coarsened_file {fno.l2b_coarse_vegadj_thresh_mosaic_file} --resolution 0.1 --data_threshold {fno.vegthresh} --mask_file {fno.erodible_file}'
        runcall(cmd_str, args.dry_run)  
    if os.path.isfile(fno.l2b_abun_mosaic_file) and os.path.isfile(fno.l3_mosaic_file) and os.path.isfile(fno.l2b_hd_vegadj_thresh_mosaic_file) is False:
        cmd_str = f'python {emit_l3_base}/veg_correction.py {fno.l2b_abun_mosaic_file} {fno.l3_mosaic_file} --soil_thresh {fno.vegthresh} --coarsened_file {fno.l2b_hd_vegadj_thresh_mosaic_file} --resolution 0.5 --data_threshold {fno.vegthresh} --mask_file {fno.erodible_file}'
        runcall(cmd_str, args.dry_run)  

    # Now aggregate grain size to 0.1 and 0.5 degrees
    if os.path.isfile(fno.l2b_grainsize_mosaic_file) and os.path.isfile(fno.l3_mosaic_file) and os.path.isfile(fno.l2b_coarse_grainsize_mosaic_file) is False:
        cmd_str = f'python {emit_l3_base}/veg_correction.py {fno.l2b_grainsize_mosaic_file} {fno.l3_mosaic_file} --coarsened_file {fno.l2b_coarse_grainsize_mosaic_file} --resolution 0.5 --soil_thresh 0 --thresh_only --mask_file {fno.erodible_file}'
        runcall(cmd_str, args.dry_run)
    if os.path.isfile(fno.l2b_grainsize_mosaic_file) and os.path.isfile(fno.l3_mosaic_file) and os.path.isfile(fno.l2b_hd_grainsize_mosaic_file) is False:
        cmd_str = f'python {emit_l3_base}/veg_correction.py {fno.l2b_grainsize_mosaic_file} {fno.l3_mosaic_file} --coarsened_file {fno.l2b_hd_grainsize_mosaic_file} --resolution 0.1 --soil_thresh 0 --thresh_only --mask_file {fno.erodible_file}'
        runcall(cmd_str, args.dry_run)

    # Now aggregate grain size with soil threshold of 0.65
    if os.path.isfile(fno.l2b_grainsize_mosaic_file) and os.path.isfile(fno.l3_mosaic_file) and os.path.isfile(fno.l2b_coarse_vegadj_grainsize_mosaic_file) is False:
        cmd_str = f'python {emit_l3_base}/veg_correction.py {fno.l2b_grainsize_mosaic_file} {fno.l3_mosaic_file} --soil_thresh 0.65 --coarsened_file {fno.l2b_coarse_vegadj_grainsize_mosaic_file} --resolution 0.5 --data_threshold 0.01 --thresh_only --mask_file {fno.erodible_file}'
        runcall(cmd_str, args.dry_run)
    if os.path.isfile(fno.l2b_grainsize_mosaic_file) and os.path.isfile(fno.l3_mosaic_file) and os.path.isfile(fno.l2b_hd_vegadj_grainsize_mosaic_file) is False:
        cmd_str = f'python {emit_l3_base}/veg_correction.py {fno.l2b_grainsize_mosaic_file} {fno.l3_mosaic_file} --soil_thresh 0.65 --coarsened_file {fno.l2b_hd_vegadj_grainsize_mosaic_file} --resolution 0.1 --data_threshold 0.01 --thresh_only --mask_file {fno.erodible_file}'
        runcall(cmd_str, args.dry_run)

    # Aggregate abundances with soil threshold of 0.65
    if os.path.isfile(fno.l2b_abun_mosaic_file) and os.path.isfile(fno.l3_mosaic_file) and os.path.isfile(fno.l2b_coarse_vegcut_thresh_mosaic_file) is False:
        cmd_str = f'python {emit_l3_base}/veg_correction.py {fno.l2b_abun_mosaic_file} {fno.l3_mosaic_file} --soil_thresh 0.65 --coarsened_file {fno.l2b_coarse_vegcut_thresh_mosaic_file} --resolution 0.5 --data_threshold 0.01 --thresh_only --mask_file {fno.erodible_file}'
        runcall(cmd_str, args.dry_run)
    if os.path.isfile(fno.l2b_abun_mosaic_file) and os.path.isfile(fno.l3_mosaic_file) and os.path.isfile(fno.l2b_hd_vegcut_thresh_mosaic_file) is False:
        cmd_str = f'python {emit_l3_base}/veg_correction.py {fno.l2b_abun_mosaic_file} {fno.l3_mosaic_file} --soil_thresh 0.65 --coarsened_file {fno.l2b_hd_vegcut_thresh_mosaic_file} --resolution 0.1 --data_threshold 0.01 --thresh_only --mask_file {fno.erodible_file}'
        runcall(cmd_str, args.dry_run)




if __name__ == "__main__":
    main()





