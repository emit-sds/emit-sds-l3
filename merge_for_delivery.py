


from netCDF4 import Dataset
import argparse
import numpy as np
import os
from datetime import datetime
import pandas as pd
from osgeo import osr
import subprocess


NODATA = -9999


def main():
    parser = argparse.ArgumentParser(description='netcdf conversion')
    parser.add_argument('baseline_file', type=str)
    parser.add_argument('unc_lower_file', type=str)
    parser.add_argument('unc_upper_file', type=str)
    parser.add_argument('output_file', type=str)
    args = parser.parse_args()


    baseline = Dataset(args.baseline_file, 'r')
    uq_l = Dataset(args.unc_lower_file, 'r')
    uq_u = Dataset(args.unc_upper_file, 'r')

    output_dataset = Dataset(args.output_file, 'w', close=False, format='NETCDF4')
    output_dataset.setncatts(baseline.__dict__)
    for dname, dim in baseline.dimensions.items():
        output_dataset.createDimension(dname, len(dim) if not dim.isunlimited() else None)


    output_dataset.title = "EMIT L3 Aggregated Mineral Spectral Abundance 0.5 Deg. V002"
    output_dataset.date_created = "2024-07-24T00:00:00Z"
    output_dataset.license = "Freely Distributed"
    output_dataset.creator_url = "https://www.jpl.nasa.gov"
    output_dataset.summary = "The Earth Surface Mineral Dust Source Investigation (EMIT) is an Earth Ventures-Instrument (EVI-4) \
Mission that maps the surface mineralogy of arid dust source regions via imaging spectroscopy in the visible and \
short-wave infrared (VSWIR). Installed on the International Space Station (ISS), the EMIT instrument is a Dyson \
imaging spectrometer that uses contiguous spectroscopic measurements from 410 to 2450 nm to resolve absoprtion \
features of iron oxides, clays, sulfates, carbonates, and other dust-forming minerals. During its one-year mission, \
EMIT will observe the sunlit Earth's dust source regions that occur within +/-52° latitude and produce maps of the \
source regions that can be used to improve forecasts of the role of mineral dust in the radiative forcing \
(warming or cooling) of the atmosphere.\n"
    output_dataset.summary = output_dataset.summary + \
    f"  This collection contains L3 Aggregated Mineral Spectral Abundance (ASA), at 0.5 degree resolution, \
for use in Earth System Models.  ASA has been masked in areas with high vegetation, water, cloud, or urban cover. \
The primary driver of uncertainty in the mass fractions is the grainsize used in the model.  As such, the uncertainty \
is represented through the 2.5 and 97.5 percentiles of the grainsize distribution, designated as UQ_low_grainsize and UQ_high_grainsize."


    for name, variable in baseline.variables.items():
      if name not in ['latitude','longitude']:
        new_var = output_dataset.createVariable(name, variable.datatype, variable.dimensions)
        new_var.setncatts(variable.__dict__)
        new_var.long_name = name + "_Baseline"
        new_var[:] = variable[:]

    for name, variable in uq_u.variables.items():
      if name not in ['latitude','longitude']:
        new_var = output_dataset.createVariable(name + "_UQ_high_grainsize", variable.datatype, variable.dimensions)
        new_var.setncatts(variable.__dict__)
        new_var.long_name = name + "_UQ_high_grainsize"
        new_var[:] = variable[:]

    for name, variable in uq_l.variables.items():
      if name not in ['latitude','longitude']:
        new_var = output_dataset.createVariable(name + "_UQ_low_grainsize", variable.datatype, variable.dimensions)
        new_var.setncatts(variable.__dict__)
        new_var.long_name = name + "_UQ_low_grainsize"
        new_var[:] = variable[:]
    
    new_var = output_dataset.createVariable('lat', baseline.variables['latitude'].datatype, baseline.variables['latitude'].dimensions)
    new_var.standard_name = "latitude"
    new_var.setncatts(baseline.variables['latitude'].__dict__)
    new_var[:] = baseline.variables['latitude'][:]

    new_var = output_dataset.createVariable('lon', baseline.variables['longitude'].datatype, baseline.variables['longitude'].dimensions)
    new_var.standard_name = "longitude"
    new_var.setncatts(baseline.variables['longitude'].__dict__)
    new_var[:] = baseline.variables['longitude'][:]

    # Add grid mapping variable if doesn't exist
    if 'latitude_longitude' not in output_dataset.dimensions:
        grid_mapping = output_dataset.createVariable('latitude_longitude', 'i4')


        lat = np.sort(output_dataset.variables['lat'])
        lon = np.sort(output_dataset.variables['lon'])
        dlat = lat[-3]-lat[-2]
        dlon = lon[2]-lon[1]
        grid_mapping.GeoTransform = f"{lon[0] - dlon/2.} {dlon} 0 {lat[-1] + dlat/2.} 0 {dlat} "
        print(grid_mapping.GeoTransform)

        spatial_ref = osr.SpatialReference()
        spatial_ref.ImportFromEPSG(4326)
        wkt = spatial_ref.ExportToWkt()
        grid_mapping.spatial_ref = wkt

    output_dataset.sync()
    output_dataset.close()
 

    




if __name__ == "__main__":
    main()

