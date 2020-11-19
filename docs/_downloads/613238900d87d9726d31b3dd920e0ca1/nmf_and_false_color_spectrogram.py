#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Source separation and false-colour spectrograms
===============================================

Soundscapes result from a combination of multiple signals that are mixed-down
into a single time-series. Unmixing these signals can be regarded as an 
important preprocessing step for further analyses of individual components.
In this example we will combine the robust characterization capabilities of 
the ``shape_features`` with an advanced signal decomposition tool, the 
non-negative-matrix factorization (NMF). NMF is a widely used tool to analyse
high-dimensional that automatically extracts sparse and meaningfull components
of non-negative matrices. Audio spectrograms are in essence sparse and 
non-negative matrices, and hence well suited to be decomposed with NMF. This 
decomposition can be further used to generate false-colour spectrograms to 
rapidly identify patterns in soundscapes. This example shows how to use the
scikit-maad package to easily decompose audio signals and visualize 
false-colour spectrograms.

Dependencies: To execute this example you will need to have instaled the 
scikit-image and scikit-learn packages.

@author: jsulloa
"""

import numpy as np
import matplotlib.pyplot as plt
from maad import sound, features
from maad.util import linear2dB, plot2D
from skimage import transform
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF

#%%
# Load audio and compute a spectrogram
s, fs = sound.load('../data/spinetail.wav')
Sxx, tn, fn, ext = sound.spectrogram(s, fs, nperseg=1024, noverlap=512)

Sxx_db = linear2dB(Sxx, db_range=80)
Sxx_db = transform.rescale(Sxx_db, 0.5, anti_aliasing=True, multichannel=False)
plot2D(Sxx_db)

#%% 
# Compute feature with shape_features_raw to get the raw output of the 
# spectrogram filtered by the filterbank composed of 2D Gabor wavelets

params, shape_im = features.shape_features_raw(Sxx_db, resolution='low')

# Format the output as an array for decomposition
X = np.array(shape_im).reshape([len(shape_im), Sxx_db.size]).transpose()

# Decompose signal using non-negative matrix factorization
Y = NMF(n_components=3, init='random', random_state=0).fit_transform(X)

# Format plt_data matrix
Y = MinMaxScaler(feature_range=(0,1)).fit_transform(Y)
intensity = 1 - MinMaxScaler(feature_range=(0,0.99)).fit_transform(Sxx_db)
plt_data = Y.reshape([Sxx_db.shape[0], Sxx_db.shape[1], 3])
plt_data = np.dstack((plt_data, intensity))

#%% 
# Plot the resulting basis spectrogram as separate elements and combine them to 
# produce a false-colour spectrogram

# Plot each basis spectrogram
fig, axes = plt.subplots(3,1)
for idx, ax in enumerate(axes):
    ax.imshow(plt_data[:,:,idx], origin='lower', aspect='auto', 
              interpolation='bilinear')
    ax.set_axis_off()
    ax.set_title('Basis ' + str(idx+1))

plt.show()
#%% 
# The first basis spectrogram shows fine and rapid modulations that the signal
# has. Both signals have these features and hence both are delineated in this
# basis. The second basis highlights the short calls on the background. The 
# third component highlights the longer vocalizations of the spinetail. 
# The three components can be mixed up to compose a false-colour spectrogram
# where it can be easily distinguished the different sound sources by color.

# Plot a false-colour spectrogram
fig, ax = plt.subplots(2,1)
ax[0].imshow(Sxx_db, origin='lower', aspect='auto', interpolation='bilinear', cmap='gray')
ax[1].imshow(plt_data, origin='lower', aspect='auto', interpolation='bilinear')
plt.show()