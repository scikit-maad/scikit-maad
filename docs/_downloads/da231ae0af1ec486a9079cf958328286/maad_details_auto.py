# -*- coding: utf-8 -*-
"""
Details Auto - This script gives an example of how to use scikit-MAAD package
=============================================================================

Created on Mon Aug  6 17:59:44 2018
@author: haupert
"""

print(__doc__)

# Clear all the variables 
from IPython import get_ipython
get_ipython().magic('reset -sf')
 
# =============================================================================
# Load the modules
# =============================================================================
import matplotlib.pyplot as plt
import pandas as pd # for csv
import numpy as np
from math import ceil

# =============================================================================
# ############## Import MAAD module
from pathlib import Path # in order to be wind/linux/MacOS compatible
import os

# change the path to the current path where the script is located
# Get the current dir of the current file
dir_path = os.path.dirname(os.path.realpath('__file__'))
os.chdir(dir_path)

maad_path = Path(dir_path).parents[0]
os.sys.path.append(maad_path.as_posix())
import maad

# Close all the figures (like in Matlab)
plt.close("all")

"""****************************************************************************
# -------------------          options              ---------------------------
****************************************************************************"""
#list 
# demo.wav
# JURA_20180812_173000.wav
# MNHN_20180712_05300.wav
# S4A03902_20171124_065000.wav
# S4A03998_20180712_060000.wav
# S4A04430_20180713_103000.wav
# S4A04430_20180712_141500.wav
filename= str(maad_path / 'data/jura_cold_forest.wav')
filename_label= filename[0:-4] +'_label.txt'
                                          
"""****************************************************************************
# -------------------          end options          ---------------------------
****************************************************************************"""


"""****************************************************************************
# -------------- LOAD SOUND AND PREPROCESS SOUND    ---------------------------
****************************************************************************"""
# Load the original sound
#datadir = filename[0:-4]
s,fs = maad.sound.load(filename=filename, channel="left",
                            display=False, savefig=None)
# Filter the sound between Low frequency cut (lfc) and High frequency cut (hlc)
s_filt = maad.sound.select_bandwidth(s, fs, lfc=250, hfc=None, order=2, 
                                     display=False, savefig=None)
# Compute the spectrogram of the sound
im_ref,dt,df,ext = maad.sound.spectrogram(s_filt, fs, dt_df_res=[0.02, 20], 
                                          db_range=60, db_gain=40, rescale=True, 
                                          fcrop =[0,20000], tcrop = [0,60],
                                          display=True, savefig=None)

#im_ref,dt,df,ext = maad.sound.spectrogram(s_filt, fs, noverlap=0.5, nperseg=2048,
#                                          db_range=60, db_gain=30, rescale=True, 
#                                          fcrop =[100,10100], tcrop = [0,60],
#                                          display=True, savefig=savefig_root)

"""****************************************************************************
# --------------------------- FIND ROIs    ------------------------------------
****************************************************************************"""
# smooth
im_smooth_pre = maad.rois.smooth(im_ref, ext, std=1, display=True)

# Noise subtraction()
win_px=round(500/df)   # convert window width from Hz into pixels
std_px=round(250/df)    # convert std from im_blurr into pixels
im_denoized = maad.rois.remove_background(im_smooth_pre, ext, gauss_win=win_px, 
                                          gauss_std=std_px, beta1=0.8, beta2=1, 
                                          llambda=1.1, display=True, 
                                          savefig=None)

# smooth
im_smooth_post = maad.rois.smooth(im_denoized, ext, std=3, display=True)

# FAIRE UNE FONCTION QUI AUGMENTE LES CONTRASTES
# methode 1.exposure.equalize_adapthist(image, kernel_size=None, clip_limit=0.01, nbins=256)
# methode 2 Laplacian : https://stackoverflow.com/questions/39619222/laplacian-sharpening-grey-image-as-result


# Binarization
im_bin = maad.rois.create_mask(im_smooth_post, ext, bin_std=7, bin_per=0.5, 
                               mode='relative', display=True, savefig=None)

# Rois extraction
"""
#==== MANUAL ======
im_rois, rois_bbox, rois_label = maad.rois.select_rois(im_bin,ext,mode_roi='manual', 
                                 filename='.\data\S4A03998_20180712_060000_label.txt',
                                 mask=False, display=True, savefig=None)
"""
#==== AUTO ========
min_f = ceil(100/df) # 100Hz 
min_t = ceil(0.1/dt) # 100ms 
max_f = np.asarray([round(1000/df), im_ref.shape[0]])
max_t = np.asarray([im_ref.shape[1], round(1/dt)])
im_rois, rois_bbox, rois_label = maad.rois.select_rois(im_bin, ext,mode_roi='auto',
                                 min_roi=np.min(min_f*min_t), max_roi=np.max(max_f*max_t), 
                                 display=True,savefig=None)

# display overlay ROIs
maad.rois.overlay_rois(im_ref, ext, rois_bbox, rois_label, savefig=None)


"""****************************************************************************
# ---------------           GET FEATURES                 ----------------------
****************************************************************************"""
# Characterise ROIs
#freq = (2**-0.5, 2**-1, 2**-1.5, 2**-2) 
#freq = (2**-0.33, 2**-0.66, 2**-1)
freq = (0.75,0.5)
params, kernels = maad.features.filter_bank_2d_nodc(frequency=freq, 
                                                    ntheta=3, bandwidth=1,
                                                    gamma=0.25, display=True, 
                                                    savefig=None)

# multiresolution image filtering (Gaussian pyramids)
im_filtlist = maad.features.filter_multires(im_ref, ext, kernels, params,
                                            npyr=2,display=True, 
                                            savefig=None, dpi=48)

# Extract shape features for each roi
params_shape, shape_features = maad.features.shapes(im_filtlist = im_filtlist, 
                                                       params = params, 
                                                       im_rois=im_rois)
# Extract centroids features for each roi
centroid_features = maad.features.centroids(im=im_ref, ext=ext, 
                                            date=maad.util.date_from_filename(filename), 
                                            im_rois=im_rois)

# 
features = maad.features.save_csv(filename[:-4]+'.csv', shape_features, 
                                  centroid_features,label_features=rois_label, 
                                  mode='w')

print(72 * '_')

"""****************************************************************************
# ---------------   FEATURES VIZUALIZATION WITH PANDAS   ----------------------
****************************************************************************"""
features = pd.read_csv(filename[:-4]+'.csv')
 
# table with a summray of the features value
features.describe()
 
# histograpm for each features
features.hist(bins=40, figsize=(12,12))
plt.show()
 
# Find correlations. 
corr_matrix = features.corr()
corr_matrix["shp1"].sort_values(ascending=False)
 
print(72 * '_')

"""****************************************************************************
# ---------------           CLASSIFY FEATURES            ----------------------
****************************************************************************"""

# =============================================================================
# Machine learning :
# Clustering/classication :  PCA
# =============================================================================

from sklearn.decomposition import PCA
import numpy as np

X = []
nshp = len(params_shape)
nrow, ncol = features.shape
select_header = list(features.columns[ncol-nshp:ncol])
#select_header.append('cfreq')
# Get the relevant shapes values
X = features[select_header].values

Y = []
# Create a vector Y with colors corresponding to the label
unique_labelName = np.unique(np.array(features.labelName))
for label in features.labelName:
    for ii, name in enumerate(unique_labelName):   
        if label in name :
            Y.append(int(ii))

# Calcul the PCA and display th results
plt.figure()
pca = PCA(n_components=2)
Xp = pca.fit_transform(X)
plt.scatter(Xp[:, 0], Xp[:, 1], c=Y, s=40, cmap='hsv')

# =============================================================================
# Machine learning :
# Clustering/classication :  Gaussian Mixture Model (GMM)
# =============================================================================
from sklearn import mixture
C = 5 # Number of clusters
clf = mixture.GaussianMixture(n_components=C, covariance_type='full')
clf.fit(X)
yp=clf.predict(X)

plt.figure()
plt.scatter(Xp[:,0],Xp[:,1],c=yp,s=40)

# =============================================================================
# Machine learning :
# Clustering/classication :  HDDC (Bouveryon)
# =============================================================================
# =============================================================================
# # Parameters for HDDA
# MODEL = 'M6'
# C = 5 # Number of clusters
# th = 0.05 # The threshold for the Cattel test
# # Select the best model using BIC or ICL
# bic, icl = [], []
# for model_ in ['M1','M2','M3','M4','M5','M6','M7','M8']:
#     model = maad.cluster.HDDC(C=C, th=th,model=model_)
#     model.fit(X)
#     bic.append(model.bic)
#     icl.append(model.icl)
#     
# plt.figure()
# plt.plot(bic)
# plt.plot(icl)
# plt.legend(("BIC", "ICL"))
# plt.xticks(np.arange(8), ('M1','M2','M3','M4','M5','M6','M7','M8'))
# plt.grid()
# 
# model = maad.cluster.HDDC(C=C,th=th,model=MODEL)
# model.fit(X)
# model.bic
# yp=model.predict(X)
# 
# plt.figure()
# plt.scatter(Xp[:,0],Xp[:,1],c=yp,s=40)
# =============================================================================

