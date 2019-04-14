# -*- coding: utf-8 -*-
""" rois functions for scikit-maad
Collection of methods to find regions of interest in 1D and 2D signals.

"""

from .rois_2d import (load,
                     remove_background,
                     select_bandwidth,
                     smooth,
                     double_threshold_rel,
                     double_threshold_abs,
                     create_mask,
                     select_rois,
                     select_rois_auto,
                     select_rois_man,
                     overlay_rois,
                     find_rois_wrapper)

from .rois_1d import (sinc,
                      _corresp_onset_offset,
                      _energy_windowed,
                      find_rois_cwt)

__all__ = ['load', 
           'remove_background', 
           'select_bandwidth',
           'smooth',
           'double_threshold_rel',
           'double_threshold_abs',
           'create_mask',
           'select_rois',
           'select_rois_auto',
           'select_rois_man',
           'overlay_rois',
           'find_rois_wrapper',
           'sinc',
           '_corresp_onset_offset',
           '_energy_windowed',
           'find_rois_cwt']