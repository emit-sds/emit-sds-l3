"""
This is a simplified form of the L3 aggregator code, meant only for demonstration sites.  The actual code will operate
globally on half degree cells.

Written by: Philip. G. Brodrick
"""

import argparse
import gdal
from netCDF4 import Dataset
from spectral.io import envi
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os
from emit_utils import daac_converter
import ray, multiprocessing

from scipy import signal
import warnings
warnings.filterwarnings("ignore")

@ray.remote
def aggregate_single_cell(SA_file: str, fractional_cover_file: str, emit_mineral_uncertainty_file:str, fractional_cover_uncertainty_file:str, mask_file: str, lxpos: int,lypos: int,lssx: int,lssy: int, xind: int, yind: int, mask_band: int, earth_band: int):
    print(xind,yind)
    SA_ds = gdal.Open(SA_file, gdal.GA_ReadOnly)
    fractional_cover_ds = gdal.Open(fractional_cover_file, gdal.GA_ReadOnly)

    calc_uncertainty=False
    if emit_mineral_uncertainty_file is not None and fractional_cover_uncertainty_file is not None:
        calc_uncertainty=True
        SA_unc_ds = gdal.Open(emit_mineral_uncertainty_file, gdal.GA_ReadOnly)
        fractional_cover_unc_ds = gdal.Open(fractional_cover_uncertainty_file, gdal.GA_ReadOnly)

    SA = SA_ds.ReadAsArray(lxpos,lypos,lssx,lssy).astype(np.float32)
    if np.sum(SA[0,...] == -9999) > SA.shape[1]*SA.shape[2] / 2:
        if calc_uncertainty:
            return yind, xind, -9999, -9999
        else:
            return yind, xind, -9999, None
    fractional_cover = fractional_cover_ds.ReadAsArray(lxpos,lypos,lssx,lssy).astype(np.float32)
    if mask_file is not None:
        mask_ds = gdal.Open(mask_file, gdal.GA_ReadOnly)
        mask = mask_ds.GetRasterBand(mask_band).ReadAsArray(lxpos,lypos,lssx,lssy).astype(np.float32)
        SA[:, mask == 1] = np.nan
    
    SAp = SA / (fractional_cover[earth_band, ...])[np.newaxis, ...]
    SAp[:,fractional_cover[earth_band,...] < 0.1] = np.nan
    SAp[:,np.isnan(fractional_cover[earth_band,...])] = np.nan
    SA[np.isnan(SAp)] = np.nan

    # Should really be spatially weighted, but will have minimal effect over 0.5 degrees.
    lASA = np.nanmean(SAp,axis=(1,2))

    if calc_uncertainty:
        SA_unc = SA_unc_ds.ReadAsArray(lxpos,lypos,lssx,lssy).astype(np.float32)
        SA_unc[:,fractional_cover[earth_band,...] < 0.1] = np.nan
        SA_unc[:,np.isnan(fractional_cover[earth_band,...])] = np.nan
        fractional_cover_unc = fractional_cover_unc_ds.ReadAsArray(lxpos,lypos,lssx,lssy).astype(np.float32)
        
        rel_earth_unc = np.power(fractional_cover_unc[earth_band,...] / fractional_cover[earth_band,...],2)
        lASA_unc = np.zeros(lASA.shape)
        for _i in range(SA.shape[0]):
            unmasked = np.sum(np.isnan(SAp) == False)
            if lASA[_i] > 0:
                rel_SA_unc = SA_unc[_i,...] / SA[_i,...]
                rel_SA_unc[np.isfinite(rel_SA_unc) == False] = np.nan
                lASA_unc[_i] = np.sqrt(np.power(lASA[_i] / unmasked,2) * np.nansum(rel_SA_unc + rel_earth_unc ) )

    else:
        lASA_unc = None
    del SA_ds, fractional_cover_ds
    if calc_uncertainty:
        del SA_unc_ds, fractional_cover_unc_ds
    return yind, xind, lASA, lASA_unc



def main():

    parser = argparse.ArgumentParser(description='DEMO L3 aggregation')
    parser.add_argument('emit_mineral_file', type=str)
    parser.add_argument('fractional_cover_file', type=str)
    parser.add_argument('aggregate_size', type=float)
    parser.add_argument('output_base', type=str)
    parser.add_argument('--emit_mineral_uncertainty_file', type=str, default=None)
    parser.add_argument('--fractional_cover_uncertainty_file', type=str, default=None)
    parser.add_argument('--mask_file', type=str, default=None)
    parser.add_argument('--mask_band', type=int, default=8)
    parser.add_argument('--of',type=str, choices=['GTiff', 'NetCDF', 'png'], default='NetCDF')
    parser.add_argument('--earth_band',type=int,default=2)
    parser.add_argument('--mineral_bands', metavar='\b', nargs='+', type=int, default=[-1,-1,-1])
    parser.add_argument('--n_cores', type=int, default=-1)
    args = parser.parse_args()

    if args.of == 'png' and len(args.mineral_bands) != 3:
        print('please pick 3 mineral bands for visualization')
        quit()

    if args.of == 'png' and -1 in args.mineral_bands:
        data_counts = np.sum(SA > 0,axis=(1,2))
        band_order = np.argsort(data_counts)[::-1].tolist()
        band_order = [x for x in band_order if x not in args.mineral_bands]
        bo_index = 0
        for _n, band in enumerate(args.mineral_bands):
            if band == -1:
                args.mineral_bands[_n] = band_order[bo_index]
                bo_index +=1

    SA_ds = gdal.Open(args.emit_mineral_file, gdal.GA_ReadOnly)
    fractional_cover_ds = gdal.Open(args.fractional_cover_file, gdal.GA_ReadOnly)
    if args.mask_file is not None:
        mask_ds = gdal.Open(args.mask_file, gdal.GA_ReadOnly)

    calc_uncertainty = False
    if args.emit_mineral_uncertainty_file is not None and args.fractional_cover_uncertainty_file is not None:
        calc_uncertainty = True
        SA_unc_ds = gdal.Open(args.emit_mineral_uncertainty_file, gdal.GA_ReadOnly)
        fractional_cover_unc_ds = gdal.Open(args.fractional_cover_uncertainty_file, gdal.GA_ReadOnly)


    trans = SA_ds.GetGeoTransform()

    emit_mineral_file_header = args.emit_mineral_file + '.hdr'
    if os.path.isfile(emit_mineral_file_header):
        mineral_band_names = envi.open(emit_mineral_file_header).metadata['band names']
    else:
        mineral_band_names = ['Goethite', 'Hematite', 'Kaolinite', 'Dolomite', 'Illite', 'Vermiculite', 'Montmorillonite', 'Gypsum', 'Calcite', 'Chlorite']

    step_size_y = int(round(abs(args.aggregate_size / trans[5])))
    step_size_x = int(round(args.aggregate_size / trans[1]))
    ul_edges_y = np.arange(0, SA_ds.RasterYSize, step_size_y).astype(int)
    ul_edges_x = np.arange(0, SA_ds.RasterXSize, step_size_x).astype(int)

    ASA = np.zeros((SA_ds.RasterCount,len(ul_edges_y),len(ul_edges_x))) - 9999
    ASA_unc = np.zeros((SA_ds.RasterCount,len(ul_edges_y),len(ul_edges_x))) - 9999

    rayargs = {'local_mode': args.n_cores == 1}
    if args.n_cores <= 0:
        args.n_cores = multiprocessing.cpu_count()
    rayargs['num_cpus'] = args.n_cores
    ray.init(**rayargs)



    jobs = []
    for _y, ypos in enumerate(ul_edges_y):
        for _x, xpos in enumerate(ul_edges_x):
            lssx = int(min(step_size_x, SA_ds.RasterXSize - xpos))
            lssy = int(min(step_size_y, SA_ds.RasterYSize - ypos))
            lxpos = int(xpos)
            lypos = int(ypos)
            jobs.append(aggregate_single_cell.remote(args.emit_mineral_file, args.fractional_cover_file, args.emit_mineral_uncertainty_file, args.fractional_cover_uncertainty_file, args.mask_file, lxpos, lypos, lssx, lssy, _x, _y, args.mask_band, args.earth_band))

    rreturn = [ray.get(jid) for jid in jobs]
    for _y, _x, lASA, lASAu in rreturn:
        ASA[:,_y,_x] = lASA
        if lASAu is not None:
            ASA_unc[:,_y,_x] = lASAu


    if args.of != 'png':
        # Build output dataset
        if args.of == 'NetCDF':
            nc_ds = Dataset(args.output_base + '.nc', 'w', clobber=True, format='NETCDF4')
            daac_converter.makeGlobalAttrBase(nc_ds)
            nc_ds.title = "EMIT L3 Aggregated Mineral Spectral Abundance 0.5 Deg. V001"
            nc_ds.summary = nc_ds.summary + \
            f"\\n\\nThis collection contains L3 Aggregated Mineral Spectral Abundance (ASA), at 0.5 degree resolution, \
            for use in Earth System Models.  ASA has been masked in areas with high vegetation, water, cloud, or urban cover.\
            "
            nc_ds.createDimension('bands', int(len(mineral_band_names)))
            nc_ds.createDimension('y', ASA.shape[1])
            nc_ds.createDimension('x', ASA.shape[2])
            daac_converter.add_variable(nc_ds, "ASA", "f4", "Aggregated Mineral Spectral Abundance", None,
                                        ASA, {"dimensions": ("bands", "y", "x")})
            daac_converter.add_variable(nc_ds, "sensor_band_parameters/mineral_names", str, "ASA Mineral Band Names", None,
                                        mineral_band_names, {"dimensions": ("bands",)})
            
            coordinate_grids = np.meshgrid(ul_edges_x, ul_edges_y)

            daac_converter.add_variable(nc_ds, "location/lat", "f4", "latitude", None, coordinate_grids[0], {"dimensions": ("y", "x")})
            daac_converter.add_variable(nc_ds, "location/lon", "f4", "longitude", None, coordinate_grids[1], {"dimensions": ("y", "x")})

            nc_ds.spatial_ref = SA_ds.GetProjection()
            nc_ds.geotransform = [trans[0], args.aggregate_size, 0, trans[3], 0, -1*args.aggregate_size]

            nc_ds.sync()
            nc_ds.close()

            if calc_uncertainty:

                nc_ds = Dataset(args.output_base + '_unc.nc', 'w', clobber=True, format='NETCDF4')
                daac_converter.makeGlobalAttrBase(nc_ds)
                nc_ds.title = "EMIT L3 Aggregated Mineral Spectral Abundance Uncertainty 0.5 Deg. V001"
                nc_ds.summary = nc_ds.summary + \
                f"\\n\\nThis collection contains L3 Aggregated Mineral Spectral Abundance (ASA) Uncertainty, at 0.5 degree resolution, \
                for use in Earth System Models.  ASA uncertainty has been masked in areas with high vegetation, water, cloud, or urban cover.\
                "
                nc_ds.createDimension('bands', int(len(mineral_band_names)))
                nc_ds.createDimension('y', ASA.shape[1])
                nc_ds.createDimension('x', ASA.shape[2])

                daac_converter.add_variable(nc_ds, "ASA_unc", "f4", "Aggregated Mineral Spectral Abundance Uncertainty", None,
                                            ASA_unc, {"dimensions": ("bands", "y", "x")})
                daac_converter.add_variable(nc_ds, "sensor_band_parameters/mineral_names", str, "ASA Mineral Band Names", None,
                                            mineral_band_names, {"dimensions": ("bands",)})
            
                coordinate_grids = np.meshgrid(ul_edges_x, ul_edges_y)

                daac_converter.add_variable(nc_ds, "location/lat", "f4", "latitude", None, coordinate_grids[0], {"dimensions": ("y", "x")})
                daac_converter.add_variable(nc_ds, "location/lon", "f4", "longitude", None, coordinate_grids[1], {"dimensions": ("y", "x")})

                nc_ds.spatial_ref = SA_ds.GetProjection()
                nc_ds.geotransform = [trans[0], args.aggregate_size, 0, trans[3], 0, -1*args.aggregate_size]

                nc_ds.sync()
                nc_ds.close()

        else:
            driver = gdal.GetDriverByName(args.of)
            driver.Register()

            if args.of == 'GTiff':
                outDataset = driver.Create(args.output_base + '.tif', ASA.shape[2], ASA.shape[1], ASA.shape[0], gdal.GDT_Float32, options=['COMPRESS=LZW'])
            else:
                outDataset = driver.Create(args.output_base, ASA.shape[2], ASA.shape[1], ASA.shape[0], gdal.GDT_Float32)
            outDataset.SetProjection(SA_ds.GetProjection())
            outDataset.SetGeoTransform([trans[0], args.aggregate_size, 0, trans[3], 0, -1*args.aggregate_size])
            for _b in range(ASA.shape[0]):
                outDataset.GetRasterBand(_b+1).WriteArray(ASA[_b,...])
                outDataset.GetRasterBand(_b+1).SetNoDataValue(-9999)
                outDataset.GetRasterBand(_b+1).SetDescription(mineral_band_names[_b])
            del outDataset

            if calc_uncertainty:
                if args.of == 'GTiff':
                    outDataset = driver.Create(args.output_base + '_unc.tif', ASA.shape[2], ASA.shape[1], ASA.shape[0], gdal.GDT_Float32, options=['COMPRESS=LZW'])
                else:
                    outDataset = driver.Create(args.output_base + '_unc', ASA.shape[2], ASA.shape[1], ASA.shape[0], gdal.GDT_Float32)
                outDataset.SetProjection(SA_ds.GetProjection())
                outDataset.SetGeoTransform([trans[0], args.aggregate_size, 0, trans[3], 0, -1*args.aggregate_size])
                for _b in range(ASA_unc.shape[0]):
                    outDataset.GetRasterBand(_b+1).WriteArray(ASA_unc[_b,...])
                    outDataset.GetRasterBand(_b+1).SetNoDataValue(-9999)
                    outDataset.GetRasterBand(_b+1).SetDescription(mineral_band_names[_b])
                del outDataset

    else:
        ASA = ASA[args.mineral_bands, ...]


        fig = plt.figure(figsize=(20, 6))
        shape=(4,1)
        buffer=0.02
        plt_xsize = (1-(shape[0]+1)*buffer)/float(shape[0])
        plt_ysize = (1-(shape[1]+1)*buffer)/float(shape[1])
        print(plt_xsize)
        plt_idx = 0

        ax = fig.add_axes([buffer*(plt_idx+1) + plt_idx*plt_xsize,buffer + 0*plt_ysize, plt_xsize,plt_ysize], zorder=1)
        to_plot = np.transpose(SA.copy(), (1,2,0))
        ax.imshow(to_plot,vmin=0,vmax=0.1)
        ax.set_title('L2b Output \n Spectral Abundance Estimate')
        ax.set_axis_off()

        ax = fig.add_axes([buffer*(plt_idx+1) + plt_idx*plt_xsize, buffer, plt_xsize, buffer*2], zorder=2)
        mineral_leg_handles = [Patch(facecolor='red', edgecolor='black',label=mineral_band_names[args.mineral_bands[0]]),
                       Patch(facecolor='green', edgecolor='black',label=mineral_band_names[args.mineral_bands[1]]),
                       Patch(facecolor='blue', edgecolor='black',label=mineral_band_names[args.mineral_bands[2]])]
        plt.legend(handles=mineral_leg_handles, loc='center', ncol=3, frameon=False)
        ax.set_axis_off()

        plt_idx += 1


        ax = fig.add_axes([buffer*(plt_idx+1) + plt_idx*plt_xsize,buffer + 0*plt_ysize, plt_xsize,plt_ysize], zorder=1)
        to_plot = np.transpose(fractional_cover[:3, ...], (1,2,0))
        ax.imshow(to_plot,vmin=0,vmax=1)
        ax.set_title('L3 Intermediate \n Fractional Cover Estimate')
        ax.set_axis_off()

        ax = fig.add_axes([buffer*(plt_idx+1) + plt_idx*plt_xsize, buffer, plt_xsize, buffer*2], zorder=2)
        fc_leg_handles = [Patch(facecolor='red', edgecolor='black',label='PV'),
                       Patch(facecolor='green', edgecolor='black',label='NPV'),
                       Patch(facecolor='blue', edgecolor='black',label='Soil')]
        plt.legend(handles=fc_leg_handles, loc='center', ncol=3, frameon=False)
        ax.set_axis_off()

        plt_idx += 1


        ax = fig.add_axes([buffer*(plt_idx+1) + plt_idx*plt_xsize,buffer + 0*plt_ysize, plt_xsize,plt_ysize], zorder=1)
        to_plot = np.transpose(SAp.copy(), (1,2,0))
        ax.imshow(to_plot,vmin=0,vmax=0.1)
        ax.set_title('L3 Intermediate \n Vegetation-Adjusted Spectral Abundance Estimate')
        ax.set_axis_off()


        ax = fig.add_axes([buffer*(plt_idx+1) + plt_idx*plt_xsize, buffer, plt_xsize, buffer*2], zorder=2)
        plt.legend(handles=mineral_leg_handles, loc='center', ncol=3, frameon=False)
        ax.set_axis_off()

        plt_idx += 1

        ax = fig.add_axes([buffer*(plt_idx+1) + plt_idx*plt_xsize,buffer + 0*plt_ysize, plt_xsize,plt_ysize], zorder=1)
        to_plot = np.transpose(ASA.copy(), (1,2,0))
        ax.imshow(to_plot,vmin=0,vmax=0.1)
        ax.set_title('L3 Output \n Aggregated Spectral Abundance Estimate')
        ax.set_axis_off()

        ax = fig.add_axes([buffer*(plt_idx+1) + plt_idx*plt_xsize, buffer, plt_xsize, buffer*2], zorder=2)
        plt.legend(handles=mineral_leg_handles, loc='center', ncol=3, frameon=False)
        ax.set_axis_off()

        plt_idx += 1

        plt.savefig(args.output_base + '.png', dpi=200, bbox_inches='tight')





if __name__ == "__main__":
    main()














