# -*- coding: utf-8 -*-
"""
Collection of functions to find regions of interest in audio and spectrograms.
"""

from .rois_2d import (load,
                     remove_background,
                     remove_background_morpho,
                     remove_background_along_axis,
                     median_equalizer,
                     select_bandwidth,
                     smooth,
                     create_mask,
                     select_rois,
                     overlay_rois,
                     rois_to_imblobs)

from .rois_1d import (sinc,
                      _corresp_onset_offset,
                      _energy_windowed,
                      find_rois_cwt)

__all__ = ['load', 
           'remove_background', 
           'remove_background_morpho',
           'remove_background_along_axis',
           'median_equalizer',
           'select_bandwidth',
           'smooth',
           'create_mask',
           'select_rois',
           'overlay_rois',
           'rois_to_imblobs',
           'sinc',
           '_corresp_onset_offset',
           '_energy_windowed',
           'find_rois_cwt']
