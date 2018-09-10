# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 17:55:47 2018

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

# Import MAAD modules   
import sys
sys.path.append('D:\\mes_projets\\2018\\_TOOLBOX\\Python\\scikit-maad') 
import maad

# change the path to the current path where the script is
import os
# Get the current dir of the current file
dir_path = os.path.dirname(os.path.realpath('__file__'))
os.chdir(dir_path)

# Close all the figures (like in Matlab)
plt.close("all")

"""****************************************************************************
# -------------------          options              ---------------------------
****************************************************************************"""
#filename = "data\demo.wav"
#filename = "data\JURA_JUILLET_2018.wav"
#filename="data\MNHN-SO-2018-128.wav'
#filename = "data\S4A03902_20171124_065000.wav"
#filename="data\S4A03998_20180712_060000.wav"
#filename='data\.wav"
#filename='data\S4A04430_20180713_103000.wav"
#filename='data\S4A04430_20180712_141500.wav"
filename="data\S4A04430_20180713_103000.wav"
savefig_root = filename[0:-4]
display=True

"""****************************************************************************
# -------------- LOAD SOUND AND PREPROCESS SOUND    ---------------------------
****************************************************************************"""
im_ref,fs,ext,date = maad.sound.preprocess(filename, display=False,
                                           dt=0.02, df=20,
                                           fcrop=[100,10100], tcrop=[0,60])

"""****************************************************************************
# --------------------------- FIND ROIs    ------------------------------------
****************************************************************************"""
im_rois, rois_bbox = maad.rois.find_rois(im_ref, ext, display=True, 
                                          mode='relative',bin_std=3, 
                                          bin_per=0.90)

"""****************************************************************************
# ---------------           GET FEATURES                 ----------------------
****************************************************************************"""
features, param_features = maad.features.get_features(im_ref, ext, date=date, 
                                            im_rois=im_rois,
                                            display=True,savefig =None,
                                            npyr=2, freq=(0.75, 0.5, 0.25), 
                                            ntheta = 2)


"""****************************************************************************
# ---------------   FEATURES VIZUALIZATION WITH PANDAS   ----------------------
****************************************************************************"""
features = pd.read_csv(filename[:-4]+'.csv')
 
# table with a summray of the features value
features.describe()
 
# histograpm for each features
features.hist(bins=50, figsize=(20,15))
plt.show()
 
# Find correlations. 
corr_matrix = features.corr()
corr_matrix["shp1"].sort_values(ascending=False)
 
print(82 * '_')

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
select_header = list(features.columns[ncol-nshp:ncol-nshp+6])
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
plt.scatter(Xp[:, 0], Xp[:, 1], c=Y, s=40)
