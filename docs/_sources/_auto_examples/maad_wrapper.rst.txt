.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download__auto_examples_maad_wrapper.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr__auto_examples_maad_wrapper.py:


Wrapper - This script gives an example of how to use scikit-MAAD package
========================================================================
Created on Mon Aug  6 17:59:44 2018
@author: haupert


.. code-block:: default


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
    # -------------- LOAD SOUND AND PREPROCESS SOUND    ---------------------------
    ****************************************************************************"""
    im_ref, fs, dt, df, ext = maad.sound.preprocess_wrapper(filename, display=True,
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
                                                date=maad.util.date_from_filename(filename),im_rois=im_rois,
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


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.000 seconds)


.. _sphx_glr_download__auto_examples_maad_wrapper.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: maad_wrapper.py <maad_wrapper.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: maad_wrapper.ipynb <maad_wrapper.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
