#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Find Regions of interest (ROIs) in a spectrogram
================================================

A spectrogram is a time-frequency (2d) representation of a audio recording. 
Each acoustic event nested in the audio recording is represented by an acoustic
signature. When sounds does not overlap in time and frequency, it is possible
to extract automatically the acoustic signature as a region of interest (ROI) 
by different image processing tools such as binarization, double thresholding,
mathematical morphology tools...

Dependencies: To execute this example you will need to have installed the 
scikit-image and pandas Python packages.

"""
# sphinx_gallery_thumbnail_path = '../_images/sphx_glr_find_rois.png'

from maad import sound, rois, features
from maad.util import power2dB, plot2D, format_features, read_audacity_annot

#%%
# First, load and audio file and compute the power spectrogram.
s, fs = sound.load('../data/cold_forest_daylight.wav')

t0 = 0
t1 = 20
f0 = 100
f1 = 10000
dB_max = 96

Sxx, tn, fn, ext = sound.spectrogram(s, fs, nperseg=1024, noverlap=512, 
                                     fcrop=(f0,f1), tcrop=(t0,t1))

# Convert the power spectrogram into dB, add dB_max which is the maximum decibel
# range when quantification bit is 16bits and display the result
Sxx_db = power2dB(Sxx) + dB_max
plot2D(Sxx_db, **{'vmin':0, 'vmax':dB_max, 'extent':ext})

#%% 
# Then, relevant acoustic events are extracted directly from the power 
# spectrogram based on a double thresholding technique. The result is binary
# image called a mask. Double thresholding technique is more sophisticated than
# basic thresholding based on a single value. First, a threshold selects pixels
# with high value (i.e. high acoustic energy). They should belong to an acoustic
# event. They are called seeds. From these seeds, we aggregate pixels connected
# to the seed with value higher than the second threslhold. These new pixels 
# become seed and the aggregating process continue until no more new pixels are
# aggregated, meaning that there is no more connected pixels with value upper 
# than the second threshold value.

# First we remove the stationary background in order to increase the contrast
Sxx_db_noNoise, _ = rois.remove_background_along_axis(Sxx_db, mode='ale', 
                                                 display=True, 
                                                 ext=ext)

# Then we smooth the spectrogram in order to facilitate the creation of masks as
# small sparse details are merged if they are close to each other
Sxx_db_noNoise_smooth = rois.smooth(Sxx_db_noNoise, ext, std=1, 
                         display=True, savefig=None)

# Then we create a mask (i.e. binarization of the spectrogram) by using the 
# double thresholding technique
im_mask = rois.create_mask(im=Sxx_db_noNoise_smooth, ext=ext, 
                           mode_bin ='relative', bin_std=6, bin_per=0.5,
                           verbose=False, display=False)

# Finaly, we put together pixels that belong to the same acoustic event, and 
# remove very small events (<=25 pixel²)
im_rois, df_rois = rois.select_rois(im_mask, min_roi=25, max_roi=None, 
                                 ext=ext, display= False,
                                 figsize=(4,(t1-t0)))
    
# format dataframe df_rois in order to convert pixels into time and frequency
df_rois = format_features(df_rois, tn, fn)

# overlay bounding box on the original spectrogram
ax, fig = rois.overlay_rois(Sxx_db, ext, df_rois, vmin=0, vmax=96)

# Compute and visualize centroids
df_centroid = features.centroid_features(Sxx_db, df_rois, im_rois)
df_centroid = format_features(df_centroid, tn, fn)
ax, fig = features.overlay_centroid(Sxx_db, ext, df_centroid, savefig=None,
                                    vmin=0, vmax=96, marker='+',ms=2,
                                    fig=fig, ax=ax)

#%% 
# Now, we can compare with manual annotation performed with Audacity software.
# Each acoustic signature is manually selected and labeled. All similar acoustic 
# signatures are labeled with the same name
df_rois = read_audacity_annot('../data/cold_forest_daylight_label.txt')  ## annotations using Audacity

# format dataframe df_rois in order to convert time and frequency into pixels
df_rois = format_features(df_rois, tn, fn)

# overlay bounding box on the original spectrogram
ax, fig = rois.overlay_rois(Sxx_db, ext, df_rois, vmin=0, vmax=96)
    
# Compute and visualize centroids
df_centroid = features.centroid_features(Sxx_db, df_rois)
df_centroid = format_features(df_centroid, tn, fn)
ax, fig = features.overlay_centroid(Sxx, ext, df_centroid, savefig=None, 
                                    vmin=-0, vmax=96, ms=2, color='blue',
                                    fig=fig, ax=ax)

#%% 
# References
# -----------
# 1. Sifre, L., & Mallat, S. (2013). Rotation, scaling and deformation invariant scattering for texture discrimination. Computer Vision and Pattern Recognition (CVPR), 2013 IEEE Conference On, 1233–1240. http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=6619007
