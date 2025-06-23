# Run grain size prediction on all EMIT scenes 
# Philip Brodrick (author)
# David R Thompson (archiver)

import subprocess
import argparse
import os
import time
import numpy as np

parser = argparse.ArgumentParser()
args = parser.parse_args()

slurmres = subprocess.run(['squeue'], stdout=subprocess.PIPE, text=True)
slurmcount = len(slurmres.stdout.split('\n'))
max_runcount=1000
if slurmcount + max_runcount > 2000:
    print(f'Too many running jobs ({slurmcount})...terminating')
    exit()

runcount=0
for x in range(-180,180,5):
    for y in range(-55,55,5):

        print(x,y)

        ll_str = f'{x}_{x+5}_{y}_{y+5}'

        rfl_lines = f'glts_5_cloudy_20240120_global/l2b/rfl_{ll_str}.txt'
        if os.path.isfile(rfl_lines) is False:
            print('nofile')
            continue
        fids = open(rfl_lines,'r').readlines()
        fids = [x.strip() for x in fids]
        fids = np.unique(fids)

        outfiles = [os.path.join('/beegfs/scratch/brodrick/emit/aggregation','grainsize',os.path.basename(fid).split('_')[0] + '_grainsize') for fid in fids]

        for _f in range(len(fids)):
           cmd_str = f'python grainsize.py {fids[_f]} /beegfs/scratch/brodrick/emit/aggregation/multiRF.sav {outfiles[_f]}'
           if os.path.isfile(outfiles[_f]) is False:
               print(cmd_str)
               subprocess.call(f'sbatch -N 1 -c 1 --mem=30G --wrap="{cmd_str}"',shell=True)
               runcount += 1
               if runcount > max_runcount:
                   exit()
     
