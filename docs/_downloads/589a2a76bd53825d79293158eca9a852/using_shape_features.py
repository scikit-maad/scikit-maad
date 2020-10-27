#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Using wavelets to characterize the time-frequency shape of sounds
=================================================================

@author:

"""
from maad.sound import load, spectrogram
from maad.features import shape_features, filter_bank_2d_nodc, opt_shape_presets, plot_shape
from maad.util import format_rois, read_audacity_annot
from maad.rois import overlay_rois
import numpy as np

s, fs = load('./data/spinetail.wav')
rois_tf = read_audacity_annot('./data/spinetail.txt')  ## annotations using Audacity
rois_cr = rois_tf.loc[rois_tf.label=='CRER',]  
rois_sp = rois_tf.loc[rois_tf.label=='SP',]

Sxx, ts, f, ext = spectrogram(s, fs)
Sxx = np.log10(Sxx)

# Visualize large vocalizations
rois_bbox = format_rois(rois_cr, ts, f, fmt='bbox')
overlay_rois(Sxx, ext, rois_bbox.values.tolist(), vmin=-16, vmax=0)

# Visualize short vocalizations
rois_bbox = format_rois(rois_sp, ts, f, fmt='bbox')
overlay_rois(Sxx, ext, rois_bbox.values.tolist(), vmin=-16, vmax=0)

# Compute an visualize features
shape_cr, params = shape_features(Sxx, ts, f, resolution='high', rois=rois_cr)
ax = plot_shape(shape_cr.mean(), params)

shape_sp, params = shape_features(Sxx, ts, f, resolution='high', rois=rois_sp)
ax = plot_shape(shape_sp.mean(), params)

# visualize using t-SNE
