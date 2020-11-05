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
                           get_df_single_row,
                           format_features)

from .visualization import (rand_cmap,
                           crop_image,
                           save_figlist,
                           plot1D,
                           plot2D)

from .math_tools import (running_mean,
                         get_unimode)

from .parser import (read_audacity_annot,
                     write_audacity_annot,
                     date_from_filename,
                     date_parser)

from .decibelSPL import (wav2volt,
                        volt2pressure,
                        wav2pressure,
                        pressure2dBSPL,
                        dBSPL2pressure,
                        power2dBSPL,
                        dBSPL2power,
                        mean_dBSPL,
                        add_dBSPL,
                        wav2dBSPL,
                        wav2Leq,
                        pressure2Leq,
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
           'get_df_single_row',
           'format_features',
           #  visualization      
           'rand_cmap',
           'crop_image',
           'save_figlist',
           'plot1D',
           'plot2D',
           # math       
           'running_mean',
           'get_unimode',
           # parser
           'read_audacity_annot',
           'write_audacity_annot',
           'date_from_filename',
           'date_parser',
           # decibelSPL        
           'wav2volt',
           'volt2pressure',
           'wav2pressure',
           'pressure2dBSPL',
           'dBSPL2pressure',
           'power2dBSPL',
           'dBSPL2power',
           'mean_dBSPL',
           'add_dBSPL',
           'wav2dBSPL',
           'wav2Leq',
           'pressure2Leq',
           'PSD2Leq']
