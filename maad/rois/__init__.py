# -*- coding: utf-8 -*-
"""
@author: haupert
"""

# =============================================================================
# Load the py files with their functions
# =============================================================================

from ._rois_func import (load,
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
                      corresp_onset_offset,
                      energy_windowed,
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
           'corresp_onset_offset',
           'energy_windowed',
           'find_rois_cwt']