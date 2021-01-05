#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Using wavelets to characterize the time-frequency shape of sounds
=================================================================

@author:

"""
from maad.sound import load, spectrogram
from maad.features import shape_features, plot_shape
from maad.util import format_features, read_audacity_annot, power2dB
from maad.rois import overlay_rois

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn import preprocessing

s, fs = load('../data/spinetail.wav')
rois_tf = read_audacity_annot('../data/spinetail.txt')  ## annotations using Audacity
rois_cr = rois_tf.loc[rois_tf.label=='CRER',]  
rois_sp = rois_tf.loc[rois_tf.label=='SP',]

Sxx_power, ts, f, ext = spectrogram(s, fs)
Sxx_dB = power2dB(Sxx_power, db_range=90) + 96

# Visualize large vocalizations
rois_cr = format_features(rois_cr, ts, f)
overlay_rois(Sxx_dB, rois_cr, **{'extent':ext, 'vmin':0, 'vmax':80})

# Visualize short vocalizations
rois_sp = format_features(rois_sp, ts, f)
overlay_rois(Sxx_dB, rois_sp, **{'extent':ext, 'vmin':0, 'vmax':80})

# Compute an visualize features
shape_cr, params = shape_features(Sxx_dB, resolution='med', rois=rois_cr)
ax = plot_shape(shape_cr.mean(), params)

shape_sp, params = shape_features(Sxx_dB, resolution='med', rois=rois_sp)
ax = plot_shape(shape_sp.mean(), params)

######## Simple clustering with PCA

# join both shapes dataframe
features = shape_cr.append(shape_sp)

# Standardizing the dataset
X = features.filter(regex='shp*',axis='columns')
X_shape = X.values.shape
X = X.values.flatten()
X = X.reshape(-1, 1)
X = preprocessing.StandardScaler().fit_transform(X)
X = X.reshape(X_shape)
Y = np.asarray(features['label'])
target_names = np.unique(Y)

# Calculate PCA
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

# Plot PCA result
plt.figure()
colors = ['navy', 'darkorange']
lw = 2
for color, i in zip(colors, np.unique(Y)):
    plt.scatter(X_r[Y == i, 0], X_r[Y == i, 1], 
                color=color, alpha=.8, lw=lw, label=i)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA')
