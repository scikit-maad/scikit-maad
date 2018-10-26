# -*- coding: utf-8 -*-
"""
@author: haupert
"""

# =============================================================================
# Load the py files with their functions
# =============================================================================

from .util import (linear_scale,
                   db_scale,
                   read_audacity_annot,
                   rand_cmap,
                   crop_image,
                   date_from_filename,
                   plot1D,
                   plot2D)

__all__ = ['linear_scale',
           'db_scale',
           'read_audacity_annot',
           'rand_cmap',
           'crop_image',
           'date_from_filename',
           'plot1D', 
           'plot2D']
