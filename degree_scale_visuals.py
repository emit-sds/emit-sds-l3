"""
This is a simplified form of the L3 aggregator code, meant only for demonstration sites, but which operates at the
half degree scale.  The actual code will operate globally.

Written by: Philip. G. Brodrick
"""

import argparse
import gdal
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import matplotlib.gridspec as gridspec

from scipy import signal

parser = argparse.ArgumentParser(description='DEMO L3 aggregation')
parser.add_argument('emit_mineral_file')
parser.add_argument('fractional_cover_file')
parser.add_argument('-emit_mineral_uncertainty_file')
parser.add_argument('-fractional_cover_uncertainty_file')
parser.add_argument('-mosaic_glt_file')
parser.add_argument('-rgb_file')
parser.add_argument('-earth_band',type=int,default=2)
parser.add_argument('-mineral_bands', metavar='\b', nargs='+', type=int, default=[-1,-1,-1])
args = parser.parse_args()

if len(args.mineral_bands) != 3:
    print('please pick 3 mineral bands for visualization')
    quit()

# Assumes files have same resolution and come from the same grid.
SA_dataset = gdal.Open(args.emit_mineral_file,gdal.GA_ReadOnly)
SAcomplete = SA_dataset.ReadAsArray()
fractional_cover = gdal.Open(args.fractional_cover_file,gdal.GA_ReadOnly).ReadAsArray()
if args.emit_mineral_uncertainty_file is not None:
    SAcomplete_uncert = gdal.Open(args.emit_mineral_uncertainty_file,gdal.GA_ReadOnly).ReadAsArray()
if args.mosaic_glt_file is not None:
    mosaic_glt = gdal.Open(args.mosaic_glt_file,gdal.GA_ReadOnly).ReadAsArray().astype(float)
    mosaic_glt[:,np.all(mosaic_glt == 0,axis=0)] = np.nan
    mosaic_glt[:,np.all(mosaic_glt == -9999,axis=0)] = np.nan
if args.rgb_file is not None:
    rgb = gdal.Open(args.rgb_file,gdal.GA_ReadOnly).ReadAsArray().astype(float)
    rgb[:,np.all(rgb == 0,axis=0)] = np.nan
    rgb[:,np.all(rgb == -9999,axis=0)] = np.nan

geotransform = SA_dataset.GetGeoTransform()
if (SAcomplete.shape[1] != fractional_cover.shape[1] and
    SAcomplete.shape[2] != fractional_cover.shape[2]):
    print('file shapes dont match')
    quit()

if -1 in args.mineral_bands:
    data_counts = np.sum(SAcomplete > 0,axis=(1,2))
    band_order = np.argsort(data_counts)[::-1].tolist()
    band_order = [x for x in band_order if x not in args.mineral_bands]
    bo_index = 0
    for _n, band in enumerate(args.mineral_bands):
        if band == -1:
            args.mineral_bands[_n] = band_order[bo_index]
            bo_index +=1

SAcomplete[SAcomplete == -9999] = np.nan
SA = SAcomplete[args.mineral_bands,...]
if args.emit_mineral_uncertainty_file is not None:
    SAcomplete_uncert[SAcomplete_uncert == -9999] = np.nan
    SA_uncert = SAcomplete_uncert[args.mineral_bands,...]
mineral_band_names = ['Goethite', 'Hematite', 'Kaolinite', 'Dolomite', 'Illite', 'Vermiculite', 'Montmorillonite', 'Gypsum', 'Calcite', 'Chlorite']
# Scale based on bare-earth fraction
SAp = SA / (fractional_cover[args.earth_band, ...])[np.newaxis, ...]
SAp[:,fractional_cover[args.earth_band,...] < 0.5] = np.nan
SAp[:,np.isnan(fractional_cover[args.earth_band,...])] = np.nan



#figsize = (16,6)
#fig = plt.figure(figsize=figsize)
#spec = gridspec.GridSpec(ncols=3, nrows=1, figure=fig, left=0.05, wspace=0.15)
#
#x = SA[0,...].flatten()
#y = SA_uncert[0,...].flatten()
#pvnpv = 1 - fractional_cover[args.earth_band,...].flatten()
#
#ax = fig.add_subplot(spec[0, 0])
#good_data = np.logical_not(np.logical_or.reduce((x == 0, y == 0, np.isnan(x), np.isnan(y), np.isnan(pvnpv))))
#print(x[good_data].shape)
#plt.scatter(x[good_data],y[good_data],s=1,c='black',alpha=0.1)
#plt.xlim([0,1.02])
#plt.ylim([0,1.02])
#plt.xlabel('{} SA'.format(mineral_band_names[args.mineral_bands[0]]))
#plt.ylabel('{} SA Uncertainty'.format(mineral_band_names[args.mineral_bands[0]]))
#
#ax = fig.add_subplot(spec[0, 1])
#plt.scatter(pvnpv[good_data],y[good_data],s=1,c='black',alpha=0.1)
#plt.xlim([0,1.02])
#plt.ylim([0,1.02])
#plt.xlabel('{} Substrate (MESMA)'.format(mineral_band_names[args.mineral_bands[0]]))
#plt.ylabel('{} SA Uncertainty'.format(mineral_band_names[args.mineral_bands[0]]))
#
#ax = fig.add_subplot(spec[0, 2])
#plt.scatter(pvnpv[good_data],x[good_data],s=1,c='black',alpha=0.1)
#plt.xlim([0,1.02])
#plt.ylim([0,1.02])
#plt.xlabel('{} Substrate (MESMA)'.format(mineral_band_names[args.mineral_bands[0]]))
#plt.ylabel('{} SA'.format(mineral_band_names[args.mineral_bands[0]]))



def make_figure(imgs,title,outname,mineral_legend=False, cmaps=None, vbounds=None, order=None):
    figsize=(14, 6)
    shape=(1,1)
    buffer=0.02
    plt_xsize = (1-(shape[0]+1)*buffer)/float(shape[0])
    plt_ysize = (1-(shape[1]+1)*buffer)/float(shape[1])
    plt_idx = 0
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([buffer*(plt_idx+1) + plt_idx*plt_xsize,buffer + 0*plt_ysize, plt_xsize,plt_ysize], zorder=1)

    for _ind, img in enumerate(imgs):
        if cmaps == None:
            ax.imshow(img)
        elif cmaps[_ind] == None:
            ax.imshow(img)
        else:
            ax.imshow(img,cmaps[_ind],vmin=vbounds[_ind][0],vmax=vbounds[_ind][1])
    ax.set_title(title)
    ax.set_axis_off()



    labels = []
    if order is None:
        labels.append(mineral_band_names[args.mineral_bands[0]] + ': Scaled 0-{}'.format(
                                         round(scaler[0], 2)))
        labels.append(mineral_band_names[args.mineral_bands[1]] + ': Scaled 0-{}'.format(
                          round(scaler[1], 2)))
        labels.append(mineral_band_names[args.mineral_bands[2]] + ': Scaled 0-{}'.format(
                          round(scaler[2], 2)))
    else:
        labels.append(mineral_band_names[args.mineral_bands[0]] + ': Scaled 0-{} x$10^{{-{}}}$'.format(round(scaler[0]*np.power(10,order),2),order))
        labels.append(mineral_band_names[args.mineral_bands[1]] + ': Scaled 0-{} x$10^{{-{}}}$'.format(round(scaler[1]*np.power(10,order),2),order))
        labels.append(mineral_band_names[args.mineral_bands[2]] + ': Scaled 0-{} x$10^{{-{}}}$'.format(round(scaler[2]*np.power(10,order),2),order))


    if mineral_legend:
        ax = fig.add_axes([buffer * 3, buffer * 3, buffer, buffer * 2], zorder=2)
        mineral_leg_handles = [Patch(facecolor='red', edgecolor='black',label=labels[0]),
                               Patch(facecolor='green', edgecolor='black',label=labels[1]),
                               Patch(facecolor='blue', edgecolor='black',label=labels[2])]
        ax.legend(handles=mineral_leg_handles, loc='center left', ncol=1, frameon=False)
        ax.set_axis_off()

    plt.savefig(outname,dpi=300,bbox_inches='tight')
    plt.clf()
    del fig



if args.mosaic_glt_file is not None:
    to_plot = np.transpose(mosaic_glt.copy(), (1,2,0))
    to_plot -= np.nanmin(to_plot, axis=(0, 1))[np.newaxis, np.newaxis, :]
    to_plot /= np.nanmax(to_plot, axis=(0, 1))[np.newaxis, np.newaxis, :]
    to_plot[np.isnan(to_plot)] = 0.5
    make_figure([to_plot], 'Mosaic GLT', 'figs/mosaic_glt.png')
    quit()

if args.rgb_file is not None:
    to_plot = np.transpose(rgb.copy(), (1,2,0))
    to_plot -= np.nanpercentile(to_plot, 2, axis=(0, 1))[np.newaxis, np.newaxis, :]
    to_plot /= np.nanpercentile(to_plot, 98, axis=(0, 1))[np.newaxis, np.newaxis, :]
    to_plot[np.isnan(to_plot)] = 0.5
    make_figure([to_plot], 'RGB', 'figs/rgb.png')
    quit()


to_plot = np.transpose(SA.copy(), (1,2,0))
scaler = np.array([np.nanpercentile(to_plot[to_plot[...,x]>0,x],98) for x in range(to_plot.shape[2])])
to_plot /= scaler[np.newaxis,np.newaxis,:]
to_plot[np.all(to_plot == 0,axis=-1),:] = 1
to_plot[np.isnan(to_plot)] = 0.5
#make_figure([to_plot], 'Mosaiced L2b Output Spectral Abundance Estimate','figs/l2b_mosaic.png',mineral_legend=True)


if args.emit_mineral_uncertainty_file is not None:
    to_plot = np.transpose(SA_uncert.copy(), (1,2,0))
    scaler = np.array([np.nanpercentile(to_plot[to_plot[...,x]>0,x],98) for x in range(to_plot.shape[2])])
    to_plot /= scaler[np.newaxis,np.newaxis,:]
    to_plot[np.all(to_plot == 0,axis=-1),:] = 1
    to_plot[np.isnan(to_plot)] = 0.5
    #make_figure([to_plot], 'Mosaiced L2b Output Spectral Abundance Uncertainty Estimate',
    #            'figs/l2b_mosaic_uncertainty_complete.png',mineral_legend=True )


    to_plot = SA_uncert.copy()
    mask = np.isnan(to_plot)
    to_plot[SA == 0] = 0
    to_plot[:, fractional_cover[args.earth_band, ...] < 0.5] = 0
    to_plot = np.transpose(to_plot, (1,2,0))
    scaler = np.array([np.nanpercentile(to_plot[to_plot[...,x]>0,x],98) for x in range(to_plot.shape[2])])
    to_plot /= scaler[np.newaxis,np.newaxis,:]
    to_plot[np.all(to_plot == 0,axis=-1),:] = 1
    to_plot[mask.transpose((1,2,0))] = 0.5
    #make_figure([to_plot], 'Mosaiced L2b Output Spectral Abundance Uncertainty Estimate',
    #            'figs/l2b_mosaic_uncertainty_mineral_present.png',mineral_legend=True )


nodata = np.all(np.isnan(SAcomplete),axis=0).astype(float)

to_plot = np.transpose(SAcomplete.copy(), (1,2,0))
mask = np.all(SAcomplete == 0,axis=0).astype(float)
nodata = np.all(np.isnan(SAcomplete),axis=0).astype(float)
to_plot = np.argmax(to_plot,axis=-1)
mask[mask == 0] = np.nan
nodata[nodata == 0] = np.nan

#make_figure([to_plot,mask,nodata], 'Mosaiced L2b Max-value Output Spectral Abundance Estimate',
#            'figs/l2b_mosaic_allmineral.png', cmaps=['tab10','binary_r','binary'],vbounds=[[0,1],[0,1],[0,2]])



# Do Aggregation
ws = int(round(0.5 / geotransform[1]))
edges_y = np.arange(0, SAp.shape[1], ws, dtype=int)
edges_x = np.arange(0, SAp.shape[2], ws, dtype=int)
ASA = np.zeros((SAp.shape[0],len(edges_y),len(edges_x)))
ASA_matchres = np.zeros((SAp.shape))
ASA_matchres[:,nodata == 1] = 0.5

if args.emit_mineral_uncertainty_file is not None:
    ASA_uncert = np.zeros((SAp.shape[0],len(edges_y),len(edges_x)))
    ASA_uncert_matchres = np.zeros((SAp.shape))
    ASA_uncert_matchres[:,nodata == 1] = 0.5
    SA_uncert[:, fractional_cover[args.earth_band, ...] < 0.5] = np.nan
    SA_uncert[SA == 0] = 0

good_fraction = np.zeros((len(edges_y),len(edges_x)))

# There's got to be a more efficient way to do this, but it involves rolling numpy axes, and this isn't actually
# expensive in the end
for _y, ypos in enumerate(edges_y):
    for _x, xpos in enumerate(edges_x):
        ASA[:,_y,_x] = np.nanmean(SAp[:,ypos:ypos + ws, xpos: xpos + ws],axis=(1,2))
        good_fraction[_y,_x] = np.nanmean(np.all(np.isnan(SA[:,ypos:ypos + ws, xpos: xpos + ws])==False,axis=0).astype(int))

        if (good_fraction[_y,_x] >= 0.5):
            ASA_matchres[:, ypos:ypos + ws, xpos: xpos + ws] = np.nanmean(SAp[:, ypos:ypos + ws, xpos: xpos + ws],axis=(1, 2))[:, np.newaxis, np.newaxis]

            # Estimationg, ignoring veg
            if args.emit_mineral_uncertainty_file is not None:
                goodcount = np.sum(np.isnan(SA_uncert[:, ypos:ypos + ws, xpos: xpos + ws]) == False,axis=(1,2))
                #uncert = np.sqrt(1/np.power(goodcount,2) * np.nansum(np.power( SA_uncert[:, ypos:ypos + ws, xpos: xpos + ws],2),axis=(1, 2)))
                uncert = np.sqrt(1/goodcount * np.nansum(np.power( SA_uncert[:, ypos:ypos + ws, xpos: xpos + ws],2),axis=(1, 2)))

                ASA_uncert_matchres[:, ypos:ypos + ws, xpos: xpos + ws] = uncert[:, np.newaxis, np.newaxis]
                ASA_uncert[:,_y,_x] = uncert

ASA[:,good_fraction < 0.5] = np.nan
if args.emit_mineral_uncertainty_file is not None:
    ASA_uncert[:,good_fraction < 0.5] = np.nan
    print(ASA_uncert)


if args.emit_mineral_uncertainty_file is not None:

    to_plot = np.transpose(ASA_uncert_matchres.copy(), (1,2,0))
    mask = np.all(to_plot == 0.5,axis=-1)
    zeros = np.all(to_plot == 0, axis=-1)
    to_plotASA = np.transpose(ASA_uncert.copy(), (1,2,0))
    scaler = np.array([np.nanpercentile(to_plotASA[to_plotASA[...,x]>0,x],98) for x in range(to_plotASA.shape[2])])
    to_plot /= scaler[np.newaxis,np.newaxis,:]
    to_plot[mask,:] = 0.5
    to_plot[zeros,:] = 1

    order = int(np.max(np.floor(-1*np.log10(scaler)))+1)


    single_color = to_plot.copy()
    single_color[np.logical_and(np.all(to_plot == 0.5, axis=-1) == False, np.all(to_plot == 1, axis=-1) == False),1:] = 0
    make_figure([single_color], 'L3 Aggregated Spectral Abundance Uncertainty Estimate',
                'figs/l3_mosaic_uncertainty_{}.png'.format(mineral_band_names[args.mineral_bands[0]]),order=order, mineral_legend=True)


    make_figure([to_plot], 'L3 Aggregated Spectral Abundance Uncertainty Estimate',
                'figs/l3_mosaic_uncertainty.png',order=order, mineral_legend=True)

quit()


to_plot = np.transpose(ASA_matchres.copy(), (1,2,0))
mask = np.all(to_plot == 0.5,axis=-1)
zeros = np.all(to_plot == 0, axis=-1)
to_plotASA = np.transpose(ASA.copy(), (1,2,0))
scaler = np.array([np.nanpercentile(to_plotASA[to_plotASA[...,x]>0,x],98) for x in range(to_plotASA.shape[2])])
to_plot /= scaler[np.newaxis,np.newaxis,:]
to_plot[mask,:] = 0.5
to_plot[zeros,:] = 1


fig = plt.figure(figsize=figsize)
plt_idx = 0
ax = fig.add_axes([buffer*(plt_idx+1) + plt_idx*plt_xsize,buffer + 0*plt_ysize, plt_xsize,plt_ysize], zorder=1)
single_color = to_plot.copy()
single_color[np.logical_and(np.all(to_plot == 0.5,axis=-1) == False, np.all(to_plot == 1,axis=-1)==False),1:] = 0
ax.imshow(single_color)
ax.set_title('L3 Aggregated Spectral Abundance Estimate')
ax.set_axis_off()

ax = fig.add_axes([buffer*3, buffer*3, buffer, buffer*2], zorder=2)
mineral_leg_handles = [Patch(facecolor='red', edgecolor='black',label=mineral_band_names[args.mineral_bands[0]] + ': Scaled 0-{}'.format(round(scaler[0],3))),
                       Patch(facecolor='green', edgecolor='black',label=mineral_band_names[args.mineral_bands[1]] + ': Scaled 0-{}'.format(round(scaler[1],3))),
                       Patch(facecolor='blue', edgecolor='black',label=mineral_band_names[args.mineral_bands[2]] + ': Scaled 0-{}'.format(round(scaler[2],3)))]
ax.legend(handles=mineral_leg_handles, loc='center left', ncol=1, frameon=False)
ax.set_axis_off()
plt.savefig('figs/l3_mosaic_{}.png'.format(mineral_band_names[args.mineral_bands[0]]),dpi=300,bbox_inches='tight')
plt.clf()
del fig


fig = plt.figure(figsize=figsize)
plt_idx = 0
ax = fig.add_axes([buffer*(plt_idx+1) + plt_idx*plt_xsize,buffer + 0*plt_ysize, plt_xsize,plt_ysize], zorder=1)

ax.imshow(to_plot)
ax.set_title('L3 Aggregated Spectral Abundance Estimate')
ax.set_axis_off()

ax = fig.add_axes([buffer*3, buffer*3, buffer, buffer*2], zorder=2)
mineral_leg_handles = [Patch(facecolor='red', edgecolor='black',label=mineral_band_names[args.mineral_bands[0]] + ': Scaled 0-{}'.format(round(scaler[0],3))),
                       Patch(facecolor='green', edgecolor='black',label=mineral_band_names[args.mineral_bands[1]] + ': Scaled 0-{}'.format(round(scaler[1],3))),
                       Patch(facecolor='blue', edgecolor='black',label=mineral_band_names[args.mineral_bands[2]] + ': Scaled 0-{}'.format(round(scaler[2],3)))]
ax.legend(handles=mineral_leg_handles, loc='center left', ncol=1, frameon=False)
ax.set_axis_off()
plt.savefig('figs/l3_mosaic.png',dpi=300,bbox_inches='tight')
plt.clf()
del fig

















