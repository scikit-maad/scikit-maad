# -*- coding: utf-8 -*-
"""
Sound pressure level
====================

The module ``spl`` is a collection of functions used to describe the physics of acoustic waves.


Conversion to sound pressure level (SPL)
----------------------------------------
.. autosummary::
    :toctree: generated/
    
    wav2volt
    volt2pressure
    wav2pressure
    pressure2dBSPL
    dBSPL2pressure
    power2dBSPL
    amplitude2dBSPL
    wav2dBSPL
    wav2leq
    pressure2leq
    psd2leq

"""

from .conversion_SPL import (wav2volt,
                        volt2pressure,
                        wav2pressure,
                        pressure2dBSPL,
                        dBSPL2pressure,
                        power2dBSPL,
                        amplitude2dBSPL,
                        wav2dBSPL,
                        wav2leq,
                        pressure2leq,
                        psd2leq)

__all__ = [
           # conversion_SPL        
           'wav2volt',
           'volt2pressure',
           'wav2pressure',
           'pressure2dBSPL',
           'dBSPL2pressure',
           'power2dBSPL',
           'amplitude2dBSPL',
           'wav2dBSPL',
           'wav2leq',
           'pressure2leq',
           'psd2leq'
           ]
