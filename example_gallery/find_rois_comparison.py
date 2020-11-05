#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 12:45:22 2020

@author: haupert
"""
from maad.sound import load, spectrogram
from maad.features import shape_features, plot_shape, centroid_features, overlay_centroid, rois_features
from maad.util import read_audacity_annot, linear_scale, format_features,get_unimode, running_mean 
from maad.rois import overlay_rois, create_mask, select_rois, find_rois_cwt, remove_background, median_equalizer
from skimage import  morphology
import numpy as np
import pandas as pd


###=============== load audio =================
s, fs = load('./data/spinetail.wav')
rois = read_audacity_annot('./data/spinetail.txt')  ## annotations using Audacity

###=============== compute spectrogram =================
Sxx, tn, fn, ext = spectrogram(s, fs)
Sxx = 10*np.log10(Sxx)

###=============== from Audacity =================

### with all labels
rois = format_features(rois, tn, fn)
ax, fig = overlay_rois(Sxx, ext, rois, vmin=-120, vmax=20)
    
# Compute an visualize features
shape, params = shape_features(Sxx, resolution='low', rois=rois)
plot_shape(shape.mean(), params)

# Compute and visualize centroids
centroid = centroid_features(Sxx, rois=rois)
centroid = format_features(centroid, tn, fn)
ax, fig = overlay_centroid(Sxx, ext, centroid, savefig=None, vmin=-120, vmax=20, fig=fig, ax=ax)

###=============== find ROI 2D =================
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
shape, params = shape_features(Sxx, resolution='low', rois=rois)
plot_shape(shape.mean(), params)

# Compute and visualize centroids
centroid = centroid_features(Sxx, rois=rois)
centroid = format_features(centroid, tn, fn)
overlay_centroid(Sxx, ext, centroid, savefig=None, vmin=-120, vmax=20, fig=fig, ax=ax)

###=============== Find ROI 1D =================
       
rois_cr = find_rois_cwt(s, fs, flims=[3000, 8000], tlen=3, th=0.003)
rois_sp = find_rois_cwt(s, fs, flims=[6000, 12000], tlen=0.2, th=0.001)

rois =pd.concat([rois_sp, rois_cr], ignore_index=True)

# view bbox
rois = format_features(rois, tn, fn)
ax, fig = overlay_rois(Sxx, ext, rois, vmin=-120, vmax=20)
    
# get features: shape, center frequency
shape, params = shape_features(Sxx, resolution='low', rois=rois)
plot_shape(shape.mean(), params)
centroid = centroid_features(Sxx, rois=rois)
centroid = format_features(centroid, tn, fn)
overlay_centroid(Sxx, ext, centroid, savefig=None, vmin=-120, vmax=20, fig=fig, ax=ax)

# final dataframe with all the features and coordinates
features = pd.merge(centroid, shape)

#=============================================================================
# other example
#=============================================================================

###=============== load audio =================
s, fs = load('./data/jura_cold_forest_jour.wav')
rois = read_audacity_annot('./data/jura_cold_forest_jour_label.txt')  ## annotations using Audacity
   
###=============== compute spectrogram =================
Sxx, tn, fn, ext = spectrogram(s, fs)
Sxx = 10*np.log10(Sxx)

###=============== from Audacity =================

### with all labels
rois = format_features(rois, tn, fn)
ax, fig = overlay_rois(Sxx, ext, rois, vmin=-120, vmax=20)

# Compute and visualize centroids
centroid = centroid_features(Sxx, rois=rois)
centroid = format_features(centroid, tn, fn)
ax, fig = overlay_centroid(Sxx, ext, centroid, savefig=None, vmin=-120, vmax=20, fig=fig, ax=ax)

###=============== find ROI 2D =================
# create a mask
X = linear_scale(Sxx)

# remove background
X = remove_background(X,ext)
X = median_equalizer(X)

im_mask = create_mask(im=X, ext=ext, 
                      mode_bin = 'relative', bin_std=3, bin_per=0.75,
                      display=False)

# create rois from mask
im_rois, rois = select_rois(im_mask,min_roi=25, max_roi=im_mask.shape[1]*5, 
                            ext=ext, display= False)
# view bbox
rois = format_features(rois, tn, fn)
ax, fig = overlay_rois(Sxx, ext, rois, vmin=-120, vmax=20)

# Compute and visualize centroids
centroid = centroid_features(Sxx, rois=rois)
centroid = format_features(centroid, tn, fn)
overlay_centroid(Sxx, ext, centroid, savefig=None, vmin=-120, vmax=20, fig=fig, ax=ax)

###=============== Find ROI 1D =================
       
rois_sh = find_rois_cwt(s, fs, flims=[7000, 8000], tlen=0.2, th=0.00001)
rois_sm = find_rois_cwt(s, fs, flims=[3500, 5500], tlen=0.2, th=0.00001)
rois_lm = find_rois_cwt(s, fs, flims=[2000, 7500], tlen=2, th=0.0001)
rois_sl = find_rois_cwt(s, fs, flims=[1800, 3000], tlen=0.2, th=0.00001)

# add label column
rois_sh['label'] = 'CR'
rois_sm['label'] = 'SM'
rois_lm['label'] = 'LM'
rois_sl['label'] = 'SL'

# concat df
rois =pd.concat([rois_sh, rois_sm, rois_lm, rois_sl], ignore_index=True)
# change position of label column to be the first column
l = rois['label'] # get the column label
rois=rois.drop(['label'],axis=1)  #drop the column
rois.insert(0,'label',l) #insert as the first column
    
# get features: centroid, 
rois = format_features(rois, tn, fn)
rois = centroid_features(Sxx, rois)
rois = rois_features(Sxx, rois)

rois = format_features(rois, tn, fn)
ax, fig = overlay_rois(Sxx, ext, rois, vmin=-120, vmax=20)
ax, fig = overlay_centroid(Sxx, ext, rois, savefig=None, vmin=-120, vmax=20, fig=fig, ax=ax)

