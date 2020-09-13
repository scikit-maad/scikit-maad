# -*- coding: utf-8 -*-
""" Utility functions for scikit-maad
Collection of miscellaneous functions that help to simplify the framework

"""

from .miscellaneous import (index_bw,
                   shift_bit_length,
                   rle,
                   linear_scale,
                   linear2dB,
                   dB2linear,
                   nearest_idx,
                   rois_to_audacity,
                   rois_to_imblobs,
                   format_rois,
                   normalize_2d)

from .visualization import (rand_cmap,
                   crop_image,
                   plot1D,
                   plot2D)

from .math_func import (running_mean,
               		get_unimode)

from .parser import (read_audacity_annot,
                     date_from_filename,
                     date_parser)

from .decibelSPL import (wav2volt,
                        volt2SPL,
                        wav2SPL,
                        SPL2dBSPL,
                        wav2dBSPL,
                        wav2Leq,
                        wavSPL2Leq,
                        energy2dBSPL,
                        dBSPL2energy,
                        PSD2Leq)

__all__ = [
           # miscellaneous 
           'index_bw',
           'shift_bit_length',
           'rle',
           'linear_scale',
           'linear2dB',
           'dB2linear',
           'nearest_idx',
           'rois_to_audacity',
           'rois_to_imblobs',
           'format_rois',
           'normalize_2d',
           #  visualization      
           'rand_cmap',
           'crop_image',
           'plot1D',
           'plot2D',
           # math       
           'running_mean',
           'get_unimode',
           # parser
           'read_audacity_annot',
           'date_from_filename',
           'date_parser',
           # decibelSPL        
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