import os
import subprocess
import glob

dirnames = ['{glt_dir}', 
           '{glt_dir}/l2a', 
           '{glt_dir}/l2b', 
           '{glt_dir}/l2b_aux', 
           '{glt_dir}/l3', 
           '{glt_dir}/l2b_coarse', 
           '{glt_dir}/l2b_coarse_hd', 
           '{glt_dir}/l2b_coarse_erod', 
           '{glt_dir}/l2b_coarse_hd_erod', 
           '{glt_dir}/l3_coarse', 
           '{glt_dir}/l2b_vegadj', 
           '{glt_dir}/l2b_vegadj_coarse', 
           '{glt_dir}/l2b_vegadj_05', 
           '{glt_dir}/l2b_vegadj_05_coarse',
           '{glt_dir}/l2b_vegadj_05_coarse_hd',
           '{glt_dir}/l2b_vegadj_25', 
           '{glt_dir}/l2b_vegadj_25_coarse',
           '{glt_dir}/l2b_vegadj_25_coarse_hd',
           '{glt_dir}/l2b_vegadj_50', 
           '{glt_dir}/l2b_vegadj_50_coarse',
           '{glt_dir}/l2b_vegadj_50_coarse_hd',
           '{glt_dir}/l2b_vegcut', 
           '{glt_dir}/l2b_vegcut_coarse',
           '{glt_dir}/l2b_vegcut_coarse_hd',
]   

def make_dirnames(glt_dir):
    for dirname in dirnames:
        dn = dirname.format(glt_dir=glt_dir)
        if os.path.isdir(dn) is False:
            subprocess.call(f'mkdir {dn}',shell=True)

# This finds the filepath for an IGM file with the given file ID
def get_igm(fid):
    igm_file = sorted(glob.glob(f'/beegfs/store/emit/ops/data/acquisitions/{fid[4:12]}/{fid.split("_")[0]}/l1b/*_l1b_loc_*.img'))[-1]
    return igm_file

# This finds the filepath for an OBS file with the given file ID
def get_obs(fid):
    obs_file = sorted(glob.glob(f'/beegfs/store/emit/ops/data/acquisitions/{fid[4:12]}/{fid.split("_")[0]}/l1b/*_l1b_obs_*.img'))[-1]
    return obs_file

# This finds the filepath for a mask file with the given file ID
def get_mask(fid):
    mask_file = sorted(glob.glob(f'/beegfs/store/emit/ops/data/acquisitions/{fid[4:12]}/{fid.split("_")[0]}/l2a/*_l2a_mask_*.img'))[-1]
    return mask_file


class filenames:
    def __init__(self, glt_dir, ll_str, abun_base, vegthresh):
        self.ll_str = ll_str
        self.glt_dir = glt_dir
        self.abun_base = abun_base
        self.vegthresh = vegthresh
        self.vegthresh_str = format(int(self.vegthresh*100), '02d') 

        self.glt_file = f'{glt_dir}/mosaic_glt_{ll_str}'
        self.coverage_file = f'{glt_dir}/coverage_{ll_str}.tif'

        self.l2b_mosaic_file =                       f'{glt_dir}/l2b/min_{ll_str}'
        self.l2b_unc_mosaic_file =                   f'{glt_dir}/l2b/minunc_{ll_str}'
        self.l2b_abun_mosaic_base =                  f'{glt_dir}/l2b/{abun_base}_{ll_str}'
        self.l2b_abun_mosaic_file =                  f'{glt_dir}/l2b/{abun_base}_{ll_str}_rel_abundance'
        self.l2b_scatter_mosaic_file =               f'{glt_dir}/l2b/{abun_base}_{ll_str}_scatter'
        self.l2b_ql_mosaic_file =                    f'{glt_dir}/l2b/{abun_base}_{ll_str}_ql_'
        self.l2b_abun_unc_mosaic_file =              f'{glt_dir}/l2b/{abun_base}_{ll_str}_mineraluncert'
        self.l2b_grainsize_mosaic_file =             f'{glt_dir}/l2b/grainsize_{ll_str}_classbased'

        self.l2a_rfl2_mosaic_file =                  f'{glt_dir}/l2a/rfl2_{ll_str}'

        self.l3_mosaic_file =                        f'{glt_dir}/l3/sma_{ll_str}'
        self.l3_unc_mosaic_file =                    f'{glt_dir}/l3/smaunc_{ll_str}'

        self.qmf_file =                              f'{glt_dir}/l2b_aux/qmf_{ll_str}.tif'
        self.erodible_file =                         f'{glt_dir}/l2b/erodible_{ll_str}'


        self.l2b_coarse_mosaic_file =                f'{glt_dir}/l2b_coarse/{abun_base}_{ll_str}.tif'
        self.l2b_hd_mosaic_file =                    f'{glt_dir}/l2b_coarse_hd/{abun_base}_{ll_str}.tif'
        self.l2b_coarse_erod_mosaic_file =           f'{glt_dir}/l2b_coarse_erod/{abun_base}_{ll_str}.tif'
        self.l2b_hd_erod_mosaic_file =               f'{glt_dir}/l2b_coarse_hd_erod/{abun_base}_{ll_str}.tif'
        self.scatter_coarse_mosaic_file =            f'{glt_dir}/l2b_coarse/scatter_{abun_base}_{ll_str}.tif'
        self.scatter_hd_mosaic_file =                f'{glt_dir}/l2b_coarse_hd/scatter_{abun_base}_{ll_str}.tif'
        self.l3_coarse_mosaic_file =                 f'{glt_dir}/l3_coarse/sma_{ll_str}.tif'

        self.l2b_coarse_grainsize_mosaic_file =      f'{glt_dir}/l2b_coarse/grainsize_{ll_str}_classbased'
        self.l2b_coarse_vegadj_grainsize_mosaic_file=f'{glt_dir}/l2b_coarse/grainsize_{ll_str}_vegadj'
        self.l2b_hd_grainsize_mosaic_file =          f'{glt_dir}/l2b_coarse_hd/grainsize_{ll_str}_classbased'
        self.l2b_hd_vegadj_grainsize_mosaic_file =   f'{glt_dir}/l2b_coarse_hd/grainsize_{ll_str}_vegadj'

        self.l2b_vegcut_thresh_mosaic_file =         f'{glt_dir}/l2b_vegcut/{abun_base}_{ll_str}'
        self.l2b_coarse_vegcut_thresh_mosaic_file =  f'{glt_dir}/l2b_vegcut_coarse/{abun_base}_{ll_str}'
        self.l2b_hd_vegcut_thresh_mosaic_file =      f'{glt_dir}/l2b_vegcut_coarse_hd/{abun_base}_{ll_str}'


        self.l2b_vegadj_thresh_mosaic_file =         f'{glt_dir}/l2b_vegadj_{self.vegthresh_str}/{abun_base}_{ll_str}'
        self.l2b_coarse_vegadj_thresh_mosaic_file =  f'{glt_dir}/l2b_vegadj_{self.vegthresh_str}_coarse/{abun_base}_{ll_str}'
        self.l2b_hd_vegadj_thresh_mosaic_file =      f'{glt_dir}/l2b_vegadj_{self.vegthresh_str}_coarse_hd/{abun_base}_{ll_str}'

        self.igm_filelist =      f'{glt_dir}/loc_{ll_str}.txt'
        self.obs_filelist =      f'{glt_dir}/obs_{ll_str}.txt'
        self.mask_filelist =     f'{glt_dir}/mask_{ll_str}.txt'
        self.rfl_filelist =      f'{glt_dir}/l2b/rfl_{ll_str}.txt'
        self.l2b_filelist =      f'{glt_dir}/l2b/min_{ll_str}.txt'
        self.l2b_unc_filelist =  f'{glt_dir}/l2b/minunc_{ll_str}.txt'
        self.l3_filelist =       f'{glt_dir}/l3/sma_{ll_str}.txt'
        self.l3_unc_filelist =   f'{glt_dir}/l3/smaunc_{ll_str}.txt'
        self.grainsize_filelist= f'{glt_dir}/l2b/grainsize_{ll_str}.txt'

        # Replace with filepath to the location where grain size estimates are stored
        self.grainsize_basepath='/beegfs/scratch/brodrick/emit/aggregation/grainsize/'

        # Replace with filepath to your copy of the erodible vector file
        # The original is linked from the article https://doi.org/10.1002/2017GC007273
        # A manually gap-filled verison is found at https://10.5281/zenodo.15723784
        self.erodible_vector_file='/beegfs/scratch/brodrick/emit/aggregation/aux/GUM_Revision_v2.1.shp'


class mosaic_filenames:
    def __init__(self, glt_dir, abun_base, vegthresh):

        self.glt_dir = glt_dir
        self.abun_base = abun_base
        self.vegthresh = vegthresh

        self.abun_file_grepstr =             f'{glt_dir}/l2b_coarse/{abun_base}_*'
        self.abun_filenames =                f'{glt_dir}/l2b_coarse/global_mosaic_{abun_base}_filenames.txt'
        self.abun_mosaic_base =              f'{glt_dir}/l2b_coarse/mosaic_{abun_base}'

        self.abun_hd_file_grepstr =          f'{glt_dir}/l2b_coarse_hd/{abun_base}_*'
        self.abun_hd_filenames =             f'{glt_dir}/l2b_coarse_hd/global_mosaic_{abun_base}_filenames.txt'
        self.abun_hd_mosaic_base =           f'{glt_dir}/l2b_coarse_hd/mosaic_{abun_base}'


        self.abun_erod_file_grepstr =        f'{glt_dir}/l2b_coarse_erod/{abun_base}_*'
        self.abun_erod_filenames =           f'{glt_dir}/l2b_coarse_erod/global_mosaic_{abun_base}_filenames.txt'
        self.abun_erod_mosaic_base =         f'{glt_dir}/l2b_coarse_erod/mosaic_{abun_base}'

        self.abun_erod_hd_file_grepstr =     f'{glt_dir}/l2b_coarse_hd_erod/{abun_base}_*'
        self.abun_erod_hd_filenames =        f'{glt_dir}/l2b_coarse_hd_erod/global_mosaic_{abun_base}_filenames.txt'
        self.abun_erod_hd_mosaic_base =      f'{glt_dir}/l2b_coarse_hd_erod/mosaic_{abun_base}'


        self.abun_vegadj_file_grepstr =      f'{glt_dir}/l2b_vegadj_{vegthresh}_coarse/{abun_base}_*'
        self.abun_vegadj_filenames =         f'{glt_dir}/l2b_vegadj_{vegthresh}_coarse/global_mosaic_{abun_base}_filenames.txt'
        self.abun_vegadj_mosaic_base =       f'{glt_dir}/l2b_vegadj_{vegthresh}_coarse/mosaic_{abun_base}'

        self.abun_vegadj_hd_file_grepstr =   f'{glt_dir}/l2b_vegadj_{vegthresh}_coarse_hd/{abun_base}_*'
        self.abun_vegadj_hd_filenames =      f'{glt_dir}/l2b_vegadj_{vegthresh}_coarse_hd/global_mosaic_{abun_base}_filenames.txt'
        self.abun_vegadj_hd_mosaic_base =    f'{glt_dir}/l2b_vegadj_{vegthresh}_coarse_hd/mosaic_{abun_base}'


        self.abun_vegcut_file_grepstr =   f'{glt_dir}/l2b_vegcut_coarse/{abun_base}_*'
        self.abun_vegcut_filenames =      f'{glt_dir}/l2b_vegcut_coarse/global_mosaic_{abun_base}_filenames.txt'
        self.abun_vegcut_mosaic_base =    f'{glt_dir}/l2b_vegcut_coarse/mosaic_{abun_base}'

        self.abun_vegcut_hd_file_grepstr =   f'{glt_dir}/l2b_vegcut_coarse_hd/{abun_base}_*'
        self.abun_vegcut_hd_filenames =      f'{glt_dir}/l2b_vegcut_coarse_hd/global_mosaic_{abun_base}_filenames.txt'
        self.abun_vegcut_hd_mosaic_base =    f'{glt_dir}/l2b_vegcut_coarse_hd/mosaic_{abun_base}'


        self.scat_hd_file_grepstr =          f'{glt_dir}/l2b_coarse_hd/scatter_{abun_base}_*'
        self.scat_hd_filenames =             f'{glt_dir}/l2b_coarse_hd/global_mosaic_{abun_base}_scatter_filenames.txt'
        self.scat_hd_mosaic_base =           f'{glt_dir}/l2b_coarse_hd/mosaic_{abun_base}_scatter'

        self.scat_file_grepstr =             f'{glt_dir}/l2b_coarse/scatter_{abun_base}_*'
        self.scat_filenames =                f'{glt_dir}/l2b_coarse/global_mosaic_{abun_base}_scatter_filenames.txt'
        self.scat_mosaic_base =              f'{glt_dir}/l2b_coarse/mosaic_{abun_base}_scatter'


        self.coarse_filter_file =            f'aux/World_Continents_buffer_coarse.tif'    
        self.hd_filter_file =                f'aux/World_Continents_buffer.tif'    
















