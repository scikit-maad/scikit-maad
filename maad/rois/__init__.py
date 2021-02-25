# -*- coding: utf-8 -*-
"""
Segmentation methods
====================

The module ``rois`` has a collection of functions to segment and find regions of interest in audio and spectrograms.

Temporal
--------
.. autosummary::
    :toctree: generated/

    find_rois_cwt
    
Spectro-temporal
----------------
.. autosummary::
    :toctree: generated/
    
    create_mask
    select_rois
    rois_to_imblobs

"""

from .rois_2d import (create_mask,
                     select_rois,
                     rois_to_imblobs)

from .rois_1d import (find_rois_cwt)

__all__ = [ # rois 2d
           'create_mask',
           'select_rois',
           'rois_to_imblobs',
           # rois 1d
           'find_rois_cwt']
