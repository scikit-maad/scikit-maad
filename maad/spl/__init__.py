# -*- coding: utf-8 -*-
""" SPL conversion functions for scikit-maad
Collection of miscelaneous functions that help to convert wav and volt date
to Sound Pressure Level (SPL in Pascal) and Leq (Continuous Equivalent SPL)

"""

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
