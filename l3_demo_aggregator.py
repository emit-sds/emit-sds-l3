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

from scipy import signal


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
    parser.add_argument('--emit_mineral_uncertainty_file')
    parser.add_argument('--fractional_cover_uncertainty_file')
    parser.add_argument('--earth_band',type=int,default=2)
    parser.add_argument('--mineral_bands', metavar='\b', nargs='+', type=int, default=[-1,-1,-1])
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

    SA = SA[args.mineral_bands,...]
    emit_mineral_file_header = args.emit_mineral_file + '.hdr'
    if os.path.isfile(emit_mineral_file_header):
        mineral_band_names = envi.open(emit_mineral_file_header).metadata['band names']
    else:
        mineral_band_names = ['Goethite', 'Hematite', 'Kaolinite', 'Dolomite', 'Illite', 'Vermiculite', 'Montmorillonite', 'Gypsum', 'Calcite', 'Chlorite']

    step_size_y = int(round(abs(args.aggregate_size / trans[5])))
    step_size_x = int(round(args.aggregate_size / trans[1]))
    ul_edges_y = np.arange(0, SA.RasterYSize, step_size_y).astype(int)
    ul_edges_x = np.arange(0, SA.RasterXSize, step_size_x).astype(int)

    ASA = np.zeros((SAp.shape[0],len(ul_edges_y),len(ul_edges_x)))
    ASA_unc = np.zeros((SAp.shape[0],len(ul_edges_y),len(ul_edges_x)))

    for _y, ypos in enumerate(ul_edges_y):
        for _x, xpos in enumerate(ul_edges_x):
            SA = SA_ds.ReadAsArray(xpos, ypos, min(step_size_x, SA_ds.RasterXSize - xpos), min(step_size_y, SA_ds.RasterYSize - ypos)).astype(np.float32)
            fractional_cover = fractional_cover_ds.ReadAsArray(xpos, ypos, min(step_size_x, fractional_cover_ds.RasterXSize - xpos), min(step_size_y, fractional_cover_ds.RasterYSize - ypos)).astype(np.float32)
            if args.mask_file is not None:
                mask = mask_ds.GetRasterBand(args.mask_band).ReadAsArray(xpos, ypos, min(step_size_x, mask_ds.RasterXSize - xpos), min(step_size_y, mask_ds.RasterYSize - ypos))
                SA[:, mask == 1] = np.nan
            
            SAp = SA / (fractional_cover[args.earth_band, ...])[np.newaxis, ...]
            SAp[:,fractional_cover[args.earth_band,...] < 0.1] = np.nan
            SAp[:,np.isnan(fractional_cover[args.earth_band,...])] = np.nan

            # Should really be spatially weighted, but will have minimal effect over 0.5 degrees.
            ASA[:,_y,_x] = np.nanmean(SAp,axis=(1,2))

            if calc_uncertainty:
                SA_unc = SA_unc_ds.ReadAsArray(xpos, ypos, min(step_size_x, SA_unc_ds.RasterXSize - xpos), min(step_size_y, SA_unc_ds.RasterYSize - ypos)).astype(np.float32)
                fractional_cover_unc = fractional_cover_unc_ds.ReadAsArray(xpos, ypos, min(step_size_x, fractional_cover_unc_ds.RasterXSize - xpos), min(step_size_y, fractional_cover_unc_ds.RasterYSize - ypos)).astype(np.float32)
                
                rel_earth_unc = np.power(fractional_cover_unc[args.earth_band,...] / fractional_cover[args.earth_band,...],2)
                for _i in range(ASA.shape[0]):
                    unmasked = np.sum(np.isnan(SAp) == False)
                    ASA_unc[_i,...] = np.sqrt(np.power(ASA[_i,...] / unmasked,2) * np.nansum(np.power(SA_unc[_i,...] / SA[_i,...],2) + rel_earth_unc ) )


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
            nc_ds.createDimension('mineral_bands', int(len(mineral_band_names)))
            nc_ds.createDimension('y', ASA.shape[2])
            nc_ds.createDimension('x', ASA.shape[1])
            daac_converter.add_variable(nc_ds, "ASA", "f4", "Aggregated Mineral Spectral Abundance", None,
                                        ASA, {"dimensions": ("mineral_bands", "y", "x")})
            if calc_uncertainty:
                daac_converter.add_variable(nc_ds, "ASA_unc", "f4", "Aggregated Mineral Spectral Abundance Uncertainty", None,
                                            ASA_unc, {"dimensions": ("mineral_bands", "y", "x")})
            daac_converter.add_variable(nc_ds, "sensor_band_parameters/mineral_names", str, "ASA Mineral Band Names", None,
                                        mineral_band_names, {"dimensions": ("mineral_bands",)})
            
            coordinate_grids = np.meshgrid(ul_edges_x, ul_edges_y)

            daac_converter.add_variable(nc_ds, "location/lat", "f4", "latitude", None, coordinate_grids[0], {"dimensions": ("y", "x")})
            daac_converter.add_variable(nc_ds, "location/lon", "f4", "longitude", None, coordinate_grids[1], {"dimensions": ("y", "x")})

            nc_ds.sync()
            nc_ds.close()
        else:
            driver = gdal.GetDriverByName(args.of)
            driver.Register()

            if args.of == 'GTiff':
                outDataset = driver.Create(args.output_base + '.tif', ASA.shape[2], ASA.shape[1], ASA.shape[0], gdal.GDT_Float32, options=['COMPRESS==LZW'])
            else:
                outDataset = driver.Create(args.output_base, ASA.shape[2], ASA.shape[1], ASA.shape[0], gdal.GDT_Float32)
            outDataset.SetProjection(SA_ds.GetProjection())
            outDataset.SetGeoTransform(SA_ds.GetGeoTransform())
            for _b in range(ASA.shape[0]):
                outDataset.GetRasterBand(_b+1).WriteAsArray(ASA[_b,...])
            del outDataset

            if calc_uncertainty:
                if args.of == 'GTiff':
                    outDataset = driver.Create(args.output_base + '_unc.tif', ASA.shape[2], ASA.shape[1], ASA.shape[0], gdal.GDT_Float32, options=['COMPRESS==LZW'])
                else:
                    outDataset = driver.Create(args.output_base + '_unc', ASA.shape[2], ASA.shape[1], ASA.shape[0], gdal.GDT_Float32)
                outDataset.SetProjection(SA_ds.GetProjection())
                outDataset.SetGeoTransform(SA_ds.GetGeoTransform())
                for _b in range(ASA_unc.shape[0]):
                    outDataset.GetRasterBand(_b+1).WriteAsArray(ASA_unc[_b,...])
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














