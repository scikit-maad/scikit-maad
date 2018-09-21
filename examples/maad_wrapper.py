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
print ("Current working directory %s" % dir_path)
# Go to the parent directory
parent_dir,_,_=dir_path.rpartition('\\')
os.chdir(parent_dir)
# Check current working directory.
print ("Directory changed successfully %s" % os.getcwd())

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
filename=".\\data\\S4A04430_20180713_103000.wav"

"""****************************************************************************
# -------------- LOAD SOUND AND PREPROCESS SOUND    ---------------------------
****************************************************************************"""
im_ref, fs, dt, df, ext, date = maad.sound.preprocess_wrapper(filename, display=True,
                                db_range=60, db_gain=40, dt_df_res=[0.02,20],
                                lfc=250, hfc=None, order=2,
                                fcrop=[0,10000], tcrop=[0,60])

"""****************************************************************************
# --------------------------- FIND ROIs    ------------------------------------
****************************************************************************"""
im_rois, rois_bbox, rois_label = maad.rois.find_rois_wrapper(im_ref, ext, display=True,
                                std_pre = 2, std_post=1, 
                                llambda=1.1, gauss_win = round(500/df),
                                mode_bin='relative', bin_std=5, bin_per=0.5,
                                mode_roi='auto')

"""****************************************************************************
# ---------------           GET FEATURES                 ----------------------
****************************************************************************"""
features, params_shape = maad.features.get_features_wrapper(im_ref, ext, 
                                            date=date,im_rois=im_rois,
                                            label_features=rois_label,
                                            display=True,savefig =None,
                                            npyr=4, freq=(0.75, 0.5, 0.25), 
                                            ntheta = 4, gamma=0.5)

"""****************************************************************************
# ---------------           CLASSIFY FEATURES            ----------------------
****************************************************************************"""

# =============================================================================
# Machine learning :
# Clustering/classication :  PCA
# =============================================================================

pca, Xp, YlabelID = maad.cluster.do_PCA(features,col_min=9)
