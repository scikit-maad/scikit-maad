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
scikit-image, scikit-learn and pandas Python packages.

"""
# sphinx_gallery_thumbnail_path = '../_images/sphx_glr_compare_auto_and_manual_rois_selection.png'

import numpy as np
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

# First we remove the stationary background in order to increase the contrast [1]
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
# Let's compare with the manual annotation (Ground Truth GT) obtained with 
# Audacity software.
# Each acoustic signature is manually selected and labeled. All similar acoustic 
# signatures are labeled with the same name
df_rois_GT = read_audacity_annot('../data/cold_forest_daylight_label.txt')  ## annotations using Audacity

# drop rows with frequency and time outside of tn and fn
df_rois_GT = df_rois_GT[(df_rois_GT.min_t >= tn.min()) & 
                        (df_rois_GT.max_t <= tn.max()) & 
                        (df_rois_GT.min_f >= fn.min()) & 
                        (df_rois_GT.max_f <= fn.max())]

# format dataframe df_rois in order to convert time and frequency into pixels
df_rois_GT = format_features(df_rois_GT, tn, fn)

# overlay bounding box on the original spectrogram
ax, fig = rois.overlay_rois(Sxx_db, ext, df_rois_GT, vmin=0, vmax=96)
    
# Compute and visualize centroids
df_centroid_GT = features.centroid_features(Sxx_db, df_rois_GT)
df_centroid_GT = format_features(df_centroid_GT, tn, fn)
ax, fig = features.overlay_centroid(Sxx, ext, df_centroid_GT, savefig=None, 
                                    vmin=-0, vmax=96, ms=2, color='blue',
                                    fig=fig, ax=ax)

# print informations about the rois
print ('Total number of ROIs : %2.0f' %len(df_rois_GT))
print ('Number of different ROIs : %2.0f' %len(np.unique(df_rois_GT['label'])))

#%%
# Now we cluster the ROIS depending on 3 ROIS features :
# - centroid_f : frequency position of the roi centroid 
# - duration_t : duration of the roi
# - bandwidth_f : frequency bandwidth of the roi
# The clustering is done by the so-called KMeans clustering algorithm.
# The number of attended clustering is the number of clusters found with 
# manual annotation.
# Finally, each rois is labeled with the corresponding cluster number predicted
# by KMeans
from sklearn.cluster import KMeans

# select features to perform KMeans clustering
FEATURES = [ 'centroid_f','duration_t','bandwidth_f']

# perform KMeans with the same number of clusters as with the manual annotation  
NN_CLUSTERS = len(np.unique(df_rois_GT['label'])) 
labels = KMeans(n_clusters=NN_CLUSTERS, random_state=0).fit_predict(df_centroid[FEATURES])

# Replace the unknow label by the cluster number predicted by KMeans
df_centroid['label'] = [str(i) for i in labels] 

# overlay color bounding box corresponding to the label, and centroids
# on the original spectrogram
ax, fig = rois.overlay_rois(Sxx_db, ext, df_centroid, vmin=0, vmax=96)
ax, fig = features.overlay_centroid(Sxx, ext, df_centroid, savefig=None, 
                                    vmin=-0, vmax=96, ms=2,
                                    fig=fig, ax=ax)

#%% 
# References
# -----------
# 1.Towsey, M., 2013b. Noise Removal from Wave-forms and Spectrograms Derived from
#   Natural Recordings of the Environment. Queensland University of Technology,
#   Brisbane