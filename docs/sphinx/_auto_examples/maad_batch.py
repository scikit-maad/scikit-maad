# -*- coding: utf-8 -*-
"""
Batch - This script gives an example of how to use scikit-MAAD package
======================================================================

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
# root directory of the files
rootdir = 'D:\\mes_projets_data\\2018\\FRANCE\\PNR_JURA\\Magneto4\\Data\\'
savedir = 'D:\\mes_projets_data\\2018\\FRANCE\\PNR_JURA\\Magneto4\\Results\\'
save_csv = 'magneto04'
         
"""****************************************************************************
# -------------------          end options          ---------------------------
****************************************************************************"""


"""****************************************************************************
# -------------- LOAD SOUND AND PREPROCESS SOUND    ---------------------------
****************************************************************************"""
# find a file in subdirectories
for root, subFolders, files in os.walk(rootdir):
    for file in files:
        if '.wav' in file:
            filename = os.path.join(root, file)
    
            # Save file basename
            savefile_base = filename[0:-4]
            savefile = savedir+savefile_base
            
            # Load the original sound
            s,fs = maad.sound.load(filename=filename, channel="left",
                                        display=False, savefig=None)
            # Filter the sound between Low frequency cut (lfc) and High frequency cut (hlc)
            s_filt = maad.sound.select_bandwidth(s, fs, lfc=1000, hfc=None, order=3, 
                                                 display=False, savefig=None)
            # Compute the spectrogram of the sound
            overlap, nperseg, dt, df = maad.sound.convert_dt_df_into_points(dt=0.025, 
                                                                            df=25, fs=fs)
            im_ref,tn,fn,ext = maad.sound.spectrogram(s_filt, fs, nperseg=nperseg, 
                                                   overlap=overlap, db_range=60, 
                                                   db_gain=30, db_norm_val=1, rescale=True, 
                                                   fcrop =[1100,11100], tcrop = None,
                                                   display=False, savefig=savefile_base)
            dt = tn[1]-tn[0]
            df = fn[1]-fn[0]
                       
            # Noise subtraction
            win_px=round(1000/df)   # convert window width from Hz into pixels
            std_px=win_px*3         # convert std from Hz into pixels
            im_denoized = maad.rois.noise_subtraction(im_ref, ext, gauss_win=win_px, 
                                                      gauss_std=std_px, beta1=0.8, beta2=1, 
                                                      llambda=1, display=False, 
                                                      savefig=None)
            # Blurring
            im_blurred = maad.rois.blurr(im_denoized, ext, std=2, display=False, 
                                         savefig=None)
            # Binarization
            im_bin = maad.rois.create_mask(im_blurred, ext, bin_c=4, bin_l=0.2, 
                                           display=False, savefig=None)
            # Rois extraction
            im_rois, rois_bbox  = maad.rois.select_rois(im_bin, ext, min_roi=25, 
                                                         max_roi=10**6, display=False, 
                                                         savefig=None)
            # display overlay ROIs
            maad.rois.overlay_rois(im_ref, ext, rois_bbox, savefig=savefile_base)
            
            # Characterise ROIs
            freq = (0.75, 0.5)
            ntheta=2
            
            params, kernels = maad.features.filter_bank_2d_nodc(frequency=freq, 
                                                                ntheta=ntheta, bandwidth=1,
                                                                gamma=1, display=False, 
                                                                savefig=savefile_base)
            
            # multiresolution image filtering (Gaussian pyramids)
            im_filtlist = maad.features.filter_multires(im_ref, ext, kernels, params,
                                                        npyr=4, display=False, savefig=None)
            
            # Extract shape features for each roi
            params_features, shape_features = maad.features.shapes(im_list = im_filtlist, 
                                                                   params = params, 
                                                                   im_rois=im_rois)
            
            # Extract centroids features for each roi
            centroid_features = maad.features.centroids(im=im_ref, df=df, dt=dt, 
                                                        date=maad.util.date_from_filename(filename), 
                                                        im_rois=im_rois)
            
            # table = maad.features.create_csv(shape_features, centroid_features)
            maad.features.save_csv(save_csv+'.csv', 
                                   shape_features, centroid_features,
                                   label_features = None)
            
            print(82 * '_')

# =============================================================================
# Data vizualization with pandas
# ============================================================================
features = pd.read_csv(filename[:-4]+'.csv')

# table with a summray of the features value
features.describe()

# histograpm for each features
# features.hist(bins=50, figsize=(20,15))
# plt.show()

# Find correlations. 
corr_matrix = features.corr()
corr_matrix["shp5"].sort_values(ascending=False)
corr_matrix["shp7"].sort_values(ascending=False)

