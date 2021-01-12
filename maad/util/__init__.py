# -*- coding: utf-8 -*-
"""
Utilities
=========

The module ``utils`` has a handfull of useful set of tools used in the audio analysis framework.

Visualization
-------------
.. autosummary::
    :toctree: generated/
    
    rand_cmap
    crop_image
    save_figlist
    plot1D
    plot2D
    plot_features_map
    plot_features
    plot_correlation_map
    false_Color_Spectro

Mathematical
------------
.. autosummary::
    :toctree: generated/
    
    running_mean
    get_unimode
    entropy
    psd
    rms
    kurtosis
    skewness
    moments

Parser
------
.. autosummary::
    :toctree: generated/
    
    read_audacity_annot
    write_audacity_annot
    date_parser

Miscellaneous
-------------
.. autosummary::
    :toctree: generated/
    
    index_bw
    intoBins
    shift_bit_length
    rle
    linear_scale
    amplitude2dB
    power2dB
    dB2amplitude
    dB2power
    mean_dB
    add_dB
    nearest_idx
    get_df_single_row
    format_features

"""

from .miscellaneous import (index_bw,
                           intoBins,
                           shift_bit_length,
                           rle,
                           linear_scale,
                           amplitude2dB,
                           power2dB,
                           dB2amplitude,
                           dB2power,
                           mean_dB,
                           add_dB,
                           nearest_idx,
                           get_df_single_row,
                           format_features)

from .visualization import (rand_cmap,
                           crop_image,
                           save_figlist,
                           plot1D,
                           plot2D,
                           plot_features_map,
                           plot_features,
                           plot_correlation_map,
                           false_Color_Spectro)

from .math_func import (running_mean,
                         get_unimode,
                         entropy,
                         psd,
                         rms,
                         kurtosis,
                         skewness,
                         moments)                     

from .parser import (read_audacity_annot,
                     write_audacity_annot,
                     date_parser)
__all__ = [
           # miscellaneous 
           'index_bw',
           'intoBins',
           'shift_bit_length',
           'rle',
           'linear_scale',
           'amplitude2dB',
           'power2dB',
           'dB2amplitude',
           'dB2power',
           'mean_dB',
           'add_dB',
           'nearest_idx',
           'get_df_single_row',
           'format_features',
           #  visualization      
           'rand_cmap',
           'crop_image',
           'save_figlist',
           'plot1D',
           'plot2D',
           'plot_features_map',
           'plot_features',
           'plot_correlation_map',
           'false_Color_Spectro',
           # math_func       
           'running_mean',
           'get_unimode',
           'entropy',
           'psd',
           'rms',
           'kurtosis',
           'skewness',
           'moments',
           # parser
           'read_audacity_annot',
           'write_audacity_annot',
           'date_parser'
            ]
