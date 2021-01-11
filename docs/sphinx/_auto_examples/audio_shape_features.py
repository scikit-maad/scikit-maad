#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute robust audio features using 2D wavelets
===============================================



Dependencies: To execute this example you will need to have instaled the Python packages
matplotlib, scikit-image and scikit-learn.

"""

import matplotlib.pyplot as plt
from maad import sound, features, rois
from maad.util import power2dB, plot2D, format_features

plt.close('all')

#%%
# First, load and audio file and compute the spectrogram.
s, fs = sound.load('../data/spinetail.wav')
#s, fs = sound.load('/Users/jsulloa/Downloads/usignolo.wav')
s = s[0:30*fs]
Sxx, tn, fn, ext = sound.spectrogram(s, fs, nperseg=1024, noverlap=512)

db_max=70
Sxx_db = power2dB(Sxx, db_range=db_max) + db_max
plot2D(Sxx_db, **{'figsize':(4,10),'extent':ext})

#%%
# Find Rois
Sxx_db_smooth = sound.smooth(Sxx_db, std=1, display=False,
                             **{'vmin':0, 'vmax':db_max, 'extent':ext})

im_mask = rois.create_mask(im=Sxx_db_smooth, mode_bin ='relative', bin_std=6, bin_per=0.5)
im_rois, df_rois = rois.select_rois(im_mask, min_roi=25, max_roi=None, display=True, **{'extent':ext})

df_rois = format_features(df_rois, tn, fn)

# overlay bounding box on the original spectrogram
ax0, fig0 = rois.overlay_rois(Sxx_db, df_rois, **{'vmin':0, 'vmax':db_max, 'extent':ext})

#%% Compute shape features
df_shape, params = features.shape_features(Sxx_db, resolution='low', rois=df_rois)

#%% Plot in a bidimensional space
features.plot_shape(df_shape.mean(), params)
