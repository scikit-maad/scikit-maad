# -*- coding: utf-8 -*-
"""
Segmentation methods
====================

The module ``rois`` has a collection of functions to segment and find regions of interest in audio and spectrograms.

Temporal (1D)
-------------
.. autosummary::
    :toctree: generated/

    find_rois_cwt
    
Spectro-temporal (2D)
---------------------
.. autosummary::
    :toctree: generated/
    
    create_mask
    select_rois
    rois_to_imblobs
    template_matching
    spectrogram_local_max

"""

from .rois_2d import (create_mask,
                     select_rois,
                     rois_to_imblobs,
                     spectrogram_local_max)

from .rois_1d import (find_rois_cwt)

from .template_matching_func import (template_matching)

__all__ = [ # rois 2d
           'create_mask',
           'select_rois',
           'rois_to_imblobs',
           'spectrogram_local_max',
           # rois 1d
           'find_rois_cwt',
           # template matching
           'template_matching']
