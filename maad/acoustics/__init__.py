# -*- coding: utf-8 -*-
"""
Collection of functions to describe the physics of acoustic waves
"""

from .decibelSPL import (wav2volt,
                        volt2pressure,
                        wav2pressure,
                        pressure2dBSPL,
                        dBSPL2pressure,
                        power2dBSPL,
                        amplitude2dBSPL,
                        wav2dBSPL,
                        wav2Leq,
                        pressure2Leq,
                        PSD2Leq)

__all__ = [
           # decibelSPL        
           'wav2volt',
           'volt2pressure',
           'wav2pressure',
           'pressure2dBSPL',
           'dBSPL2pressure',
           'power2dBSPL',
           'amplitude2dBSPL',
           'wav2dBSPL',
           'wav2Leq',
           'pressure2Leq',
           'PSD2Leq']
