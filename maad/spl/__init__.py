# -*- coding: utf-8 -*-
"""
Sound pressure level
====================

The module ``spl`` is a collection of functions used to describe the physics of acoustic waves.


Conversion SPL
--------------
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
    wav2Leq
    pressure2Leq
    PSD2Leq
    
Attenuation and active distance
-------------------------------
.. autosummary::
    :toctree: generated/
    
    attenuation_dB,
    dBSPL_per_bin,
    active_distance,
    pressure_at_r0,
    L0_at_r0,
    apply_attenuation

"""

from .conversion_SPL import (wav2volt,
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

from .active_space import (attenuation_dB,
                           dBSPL_per_bin,
                           active_distance,
                           pressure_at_r0,
                           dBSPL_at_r0,
                           apply_attenuation)

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
           'wav2Leq',
           'pressure2Leq',
           'PSD2Leq'
           #active_space
           'attenuation_dB',
           'dBSPL_per_bin',
           'active_distance',
           'pressure_at_r0',
           'dBSPL_at_r0',
           'apply_attenuation'
           ]
