#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Signal decomposition and false-color spectrograms
===============================================

Soundscapes result from a combination of multiple signals that are mixed-down
into a single time-series. Unmixing these signals can be regarded as an 
important preprocessing step for further analyses of individual components.
Here, we will combine the robust characterization capabilities of 
the bidimensional wavelets with an advanced signal decomposition tool, the 
non-negative-matrix factorization (NMF). NMF is a widely used tool to analyse
high-dimensional that automatically extracts sparse and meaningfull components
of non-negative matrices. Audio spectrograms are in essence sparse and 
non-negative matrices, and hence well suited to be decomposed with NMF. This 
decomposition can be further used to generate false-color spectrograms to 
rapidly identify patterns in soundscapes and increase the interpretability of 
the signal. This example shows how to use the scikit-maad package to easily 
decompose audio signals and visualize false-colour spectrograms.

Dependencies: To execute this example you will need to have instaled the 
scikit-image and scikit-learn Python packages.

"""
# sphinx_gallery_thumbnail_path = '../_images/sphx_glr_plot_nmf_and_false_color_spectrogram_003.png'
import numpy as np
import matplotlib.pyplot as plt
from maad import sound, features
from maad.util import linear2dB, plot2D
from skimage import transform
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF

#%%
# First, load and audio file and compute the spectrogram.
s, fs = sound.load('../data/spinetail.wav')
Sxx, tn, fn, ext = sound.spectrogram(s, fs, nperseg=1024, noverlap=512)

Sxx_db = linear2dB(Sxx, db_range=70)
Sxx_db = transform.rescale(Sxx_db, 0.5, anti_aliasing=True, multichannel=False)
plot2D(Sxx_db, **{'figsize':(4,10),'extent':(tn[0], tn[-1], fn[0], fn[-1])})

#%% 
# Then, compute feature with ``shape_features_raw`` to get the raw output of the 
# spectrogram filtered by the filterbank composed of 2D Gabor wavelets. This
# raw output can be fed to the NMF algorithm to decompose the spectrogram into
# elementary basis spectrograms.

params, shape_im = features.shape_features_raw(Sxx_db, resolution='low')

# Format the output as an array for decomposition
X = np.array(shape_im).reshape([len(shape_im), Sxx_db.size]).transpose()

# Decompose signal using non-negative matrix factorization
Y = NMF(n_components=3, init='random', random_state=0).fit_transform(X)

# Normalize the data and combine the three NMF basis spectrograms and the
# intensity spectrogram into a single array to fit the RGBA color model. RGBA
# stands for Red, Green, Blue and Alpha, where alpha indicates how opaque each
# pixel is.

Y = MinMaxScaler(feature_range=(0,1)).fit_transform(Y)
intensity = 1 - (Sxx_db - Sxx_db.min()) / (Sxx_db.max() - Sxx_db.min())
plt_data = Y.reshape([Sxx_db.shape[0], Sxx_db.shape[1], 3])
plt_data = np.dstack((plt_data, intensity))

#%% 
# Finally, plot the resulting basis spectrogram as separate elements and 
# combine them to produce a false-colour spectrogram using the RGBA color 
# model.

fig, axes = plt.subplots(3,1, figsize=(10,8))
for idx, ax in enumerate(axes):
    ax.imshow(plt_data[:,:,idx], origin='lower', aspect='auto', 
              interpolation='bilinear')
    ax.set_axis_off()
    ax.set_title('Basis ' + str(idx+1))

#%% 
# The first basis spectrogram shows fine and rapid modulations of the signal.
# Both signals have these features and hence both are delineated in this
# basis. The second basis highlights the short calls on the background, and the 
# third component highlights the longer vocalizations of the spinetail. 
# The three components can be mixed up to compose a false-colour spectrogram
# where it can be easily distinguished the different sound sources by color.

fig, ax = plt.subplots(2,1, figsize=(10,6))
ax[0].imshow(Sxx_db, origin='lower', aspect='auto', interpolation='bilinear', cmap='gray')
ax[0].set_axis_off()
ax[0].set_title('Spectrogram')
ax[1].imshow(plt_data, origin='lower', aspect='auto', interpolation='bilinear')
ax[1].set_axis_off()
ax[1].set_title('False-color spectrogram')
