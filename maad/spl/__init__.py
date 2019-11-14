# -*- coding: utf-8 -*-
""" Utility functions for scikit-maad
Collection of miscelaneous functions that help to simplify the framework

"""

from .spl import (wav2volt,
                  volt2SPL,
                  wav2SPL,
                  SPL2dBSPL,
                  wav2dBSPL,
                  wav2Leq,
                  wavSPL2Leq,
                  energy2dBSPL,
                  dBSPL2energy,
                  PSD2Leq)

__all__ = ['wav2volt',
           'volt2SPL',
           'wav2SPL',
           'SPL2dBSPL',
           'wav2dBSPL',
           'wav2Leq',
           'wavSPL2Leq',
           'energy2dBSPL',
           'dBSPL2energy',
           'PSD2Leq']
