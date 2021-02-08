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
    plot1d
    plot_wave
    plot_spectrum
    plot2d
    plot_spectrogram
    overlay_rois
    overlay_centroid
    plot_features_map
    plot_features
    plot_correlation_map
    plot_shape
    false_Color_Spectro

Mathematical
------------
.. autosummary::
    :toctree: generated/
    
    running_mean
    get_unimode
    entropy
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
    into_bins
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
                           into_bins,
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
                           plot1d,
                           plot_wave,
                           plot_spectrum,
                           plot2d,
                           plot_spectrogram,
                           overlay_rois,
                           overlay_centroid,
                           plot_features_map,
                           plot_features,
                           plot_correlation_map,
                           plot_shape,
                           false_Color_Spectro)

from .math_func import (running_mean,
                         get_unimode,
                         entropy,
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
           'into_bins',
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
           'plot1d',
           'plot_wave',
           'plot_spectrum',
           'plot2d',
           'plot_spectrogram',
           'overlay_rois',
           'overlay_centroid',
           'plot_features_map',
           'plot_features',
           'plot_correlation_map',
           'plot_shape',
           'false_Color_Spectro',
           # math_func       
           'running_mean',
           'get_unimode',
           'entropy',
           'rms',
           'kurtosis',
           'skewness',
           'moments',
           # parser
           'read_audacity_annot',
           'write_audacity_annot',
           'date_parser'
            ]
