# Top-level EMIT L3 runscript - create unmixing and tetracorder input files
# Written Philip G. Brodrick
# Archived by David R. Thompson

import datetime
import argparse
import glob
from spectral.io import envi
import numpy as np
import subprocess
import os


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('fidlist', type=str)
    parser.add_argument('output_base', type=str, metavar='output file')
    parser.add_argument('--l3',help='l3', action='store_true') 
    args = parser.parse_args()

    fids = np.genfromtxt(args.fidlist,dtype=str)
    
    # Replace with the path to your installation
    unmixing_executable = '/beegfs/scratch/brodrick/emit/SpectralUnmixing/unmix.jl'
    endmember_library = '/beegfs/store/shared/unmixing_libraries/pc-endmember_before-split_num-dims_3_num-samples_8_library.csv'

    subprocess.call(f'mkdir {args.output_base}',shell=True)
    subprocess.call(f'mkdir {args.output_base}/l2b',shell=True)
    subprocess.call(f'mkdir {args.output_base}/l3',shell=True)

    for fid in fids:

        # Identify the reflectance for this file ID
        try:
            rfl = sorted(glob.glob(f'/beegfs/store/emit/ops/data/acquisitions/{fid[4:12]}/{fid.split("_")[0]}/l2a/*_l2a_rfl_*.img'))[-1]
            rflunc = sorted(glob.glob(f'/beegfs/store/emit/ops/data/acquisitions/{fid[4:12]}/{fid.split("_")[0]}/l2a/*_l2a_rfluncert_*.img'))[-1]
        except:
            print(f'skipping {fid}')
            continue
        glt = rfl.replace('rfl','glt').replace('l2a','l1b')

        tetra_base = f'{args.output_base}/l2b/{fid}'
        sma_base = f'{args.output_base}/l3/{fid}'
        
        # Run tetracorder if the output files do not exist
        if os.path.isfile(f'{tetra_base}_sa') is False:
            call = f'sbatch -N 1 -c 5 -p emit --mem=20G --wrap="sh run_tetracorder.sh {fid} {os.path.splitext(rfl)[0]} {tetra_base} {rflunc}"'
            print(call)
            subprocess.call(call,shell=True)

        # Run the spectral unmixing code if the output files do not exist
        if args.l3 and os.path.isfile(f'{sma_base}_sma') is False:
            call = f'sbatch -N 1 -c 5 -p emit --mem=20G --wrap="julia -p 5 {unmixing_executable} {rfl} {endmember_library} level_1 {sma_base}_sma --spectral_starting_column 8 --reflectance_uncertainty_file {rflunc} --n_mc 20 --normalization brightness --num_endmembers 50 --mode sma-best --log_file {sma_base}_l3_log.txt"'
            subprocess.call(call,shell=True)
         

if __name__ == "__main__":
    main()

