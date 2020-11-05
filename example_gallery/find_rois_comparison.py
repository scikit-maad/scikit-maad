#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 12:45:22 2020

@author: haupert
"""
from maad.sound import load, spectrogram
from maad.features import shape_features, plot_shape, centroid_features, overlay_centroid
from maad.util import read_audacity_annot, linear_scale, format_features
from maad.rois import overlay_rois, create_mask, select_rois, find_rois_cwt
import numpy as np
import pandas as pd


#s, fs = load('./data/jura_cold_forest_jour.wav')
#rois = read_audacity_annot('./data/jura_cold_forest_jour_label.txt')  ## annotations using Audacity

s, fs = load('./data/spinetail.wav')
rois = read_audacity_annot('./data/spinetail.txt')  ## annotations using Audacity

# Spectrogram
Sxx, tn, fn, ext = spectrogram(s, fs)
Sxx = 10*np.log10(Sxx)

###=============== from Audacity =================

### with all labels
rois = format_features(rois, tn, fn)
overlay_rois(Sxx, ext, rois, vmin=-120, vmax=20)
    
# Compute an visualize features
shape, params = shape_features(Sxx, resolution='med', rois=rois)
ax = plot_shape(shape.mean(), params)

###=============== Auto 2D =================
# create a mask
X = linear_scale(Sxx)
im_mask = create_mask(im=X, ext=ext, 
                      mode_bin = 'relative', bin_std=1.5, bin_per=0.1,
                      display=False)
# create rois from mask
im_rois, rois = select_rois(im_mask,min_roi=200, max_roi=im_mask.shape[1]*5, 
                            ext=ext, display= True)
# view bbox
rois = format_features(rois, tn, fn)
ax, fig = overlay_rois(Sxx, ext, rois, vmin=-120, vmax=20)

# Compute and visualize features
shape, params = shape_features(Sxx, resolution='med', rois=rois)
plot_shape(shape.mean(), params)

# Compute and visualize features
centroid = centroid_features(Sxx, rois=rois)
centroid = format_features(centroid, tn, fn)
overlay_centroid(Sxx, ext, centroid, savefig=None, vmin=-120, vmax=20, fig=fig, ax=ax)

###=============== Auto 1D =================
       
rois_cr = find_rois_cwt(s, fs, flims=[3000, 8000], tlen=3, th=0.003)
rois_sp = find_rois_cwt(s, fs, flims=[6000, 12000], tlen=0.2, th=0.001)

rois =pd.concat([rois_sp, rois_cr], ignore_index=True)

# view bbox
rois = format_features(rois, tn, fn)
ax, fig = overlay_rois(Sxx, ext, rois, vmin=-120, vmax=20)
    
# get features: shape, center frequency
shape, params = shape_features(Sxx, resolution='med', rois=rois)
plot_shape(shape.mean(), params)
centroid = centroid_features(Sxx, rois=rois)
centroid = format_features(centroid, tn, fn)
overlay_centroid(Sxx, ext, centroid, savefig=None, vmin=-120, vmax=20, fig=fig, ax=ax)

# final dataframe with all the features and coordinates
features = pd.merge(centroid, shape)


####
rois = read_audacity_annot('./data/spinetail.txt')  ## annotations using Audacity
#rois = rois.loc[rois.label=='SP',] 
rois = format_features(rois, tn, fn)
centroid = centroid_features(Sxx, rois=rois)
centroid = format_features(centroid, tn, fn)
ax, fig = overlay_rois(Sxx, ext, rois, vmin=-120, vmax=20)
ax, fig = overlay_centroid(Sxx, ext, centroid, vmin=-120, vmax=20, fig=fig, ax=ax)
   

