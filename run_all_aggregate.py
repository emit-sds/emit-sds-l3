# Top-level EMIT L3 runscript
# Written Philip G. Brodrick
# Archived by David R. Thompson

import subprocess
import argparse
import os
import time
import filenames as fn

# Location of SpectralUnmixing repository in local filesystem
julia_project = '/beegfs/scratch/brodrick/emit/SpectralUnmixing'

parser = argparse.ArgumentParser()
parser.add_argument('--prebuilt_fidlist', action='store_true')
parser.add_argument('--glt_dir', type=str, default='glts_5_cloudy_20240120_global')
parser.add_argument('--m4s', type=str, default=None)
args = parser.parse_args()

fn.make_dirnames(args.glt_dir)

for x in range(-180,181,5):
    for y in range(-55,56,5):
        cmd_str=f'python build_cloudy_glts_m4_rev.py {x} {x+5} {y} {y+5} --glt_dir {args.glt_dir}'
        if args.m4s is not None:
            cmd_str += f' --m4s {args.m4s}'

        ll_str = f'{x}_{x+5}_{y}_{y+5}'
        fidlist = f'{args.glt_dir}/fids_{ll_str}_0.50_none_20240120.txt'
        if args.prebuilt_fidlist:
            if os.path.getsize(fidlist) > 0:
                print(x,y)
                print(cmd_str)
                subprocess.call(f'sbatch -N 1 -c 1 --mem=18G --job-name emit-l3 -p standard,emit --wrap="source /tmp/miniconda/bin/activate isofit_env; export JULIA_PROJECT={julia_project}; {cmd_str}"',shell=True)
                time.sleep(0.1)
            else:
                print('skipping',x,y)
        else:
            subprocess.call(f'sbatch -N 1 -c 1 --mem=18G --job-name emit-l3 -p emit,standard --wrap="source /tmp/miniconda/bin/activate isofit_env; export JULIA_PROJECT={julia_project}; {cmd_str}"',shell=True)

