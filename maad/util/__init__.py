# -*- coding: utf-8 -*-
""" Utility functions for scikit-maad
Collection of miscelaneous functions that help to simplify the framework

"""

from .util import (index_bw,
                   running_mean,
                   shift_bit_length,
                   rle,
                   linear_scale,
                   linear2dB,
                   dB2linear,
                   rand_cmap,
                   crop_image,
                   get_unimode,
                   plot1D,
                   plot2D,
                   nearest_idx,
                   rois_to_audacity,
                   rois_to_imblobs,
                   format_rois,
                   normalize_2d)

from .parser_func import (read_audacity_annot,
                          date_from_filename,
                          date_parser)

from .wav2dBSPL import (wav2volt,
                        volt2SPL,
                        wav2SPL,
                        SPL2dBSPL,
                        wav2dBSPL,
                        wav2Leq,
                        wavSPL2Leq,
                        energy2dBSPL,
                        dBSPL2energy,
                        PSD2Leq)

__all__ = ['index_bw',
           'running_mean',
           'shift_bit_length',
           'rle',
           'linear_scale',
           'linear2dB',
           'dB2linear',
           'get_unimode',
           'rand_cmap',
           'crop_image',
           'read_audacity_annot',
           'date_from_filename',
           'date_parser',
           'plot1D', 
           'plot2D',
           'nearest_idx',
           'rois_to_audacity',
           'rois_to_imblobs',
           'format_rois',
           'normalize_2d',
           'wav2volt',
           'volt2SPL',
           'wav2SPL',
           'SPL2dBSPL',
           'wav2dBSPL',
           'wav2Leq',
           'wavSPL2Leq',
           'energy2dBSPL',
           'dBSPL2energy',
           'PSD2Leq']
