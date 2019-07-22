#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 High level functions for signal characterization from 1D signals
 Code licensed under both GPL and BSD licenses
 Authors:  Juan Sebastian ULLOA <jseb.ulloa@gmail.com>
           Sylvain HAUPERT <sylvain.haupert@mnhn.fr>

"""

from scipy.signal import periodogram, welch
import pandas as pd

def psd(s, fs, nperseg=256, method='welch', window='hanning', nfft=None, tlims=None):
    """ 
    Estimates power spectral density of 1D signal using multiple methods 
    Note: this is a wrapper function that uses functions from scipy.signal module
    
    Parameters
    ----------
        s: 1D array
            Input signal to process 
        fs: float, optional
            sampling frequency of audio signal
        nperseg: int, optional
            lenght of segment for 'welch' method
        nfft: int, optional
            length of FFT
        method: {'welch', 'periodogram'}
            method used to compute the power spectral density
        tlims: 2D array
            temporal limits to compute the power spectral density
    Returns
    -------
        psd: 1D array
    
    """
    if tlims==None and method=='welch':
        f_idx, psd_s = welch(s, fs, window, nperseg, nfft)
    
    elif tlims==None and method=='periodogram':
        f_idx, psd_s = periodogram(s, fs, window, nfft, scaling='spectrum')
        
    else:
        print('TODO: Computation of power spectral density in windows will be available soon')
    
    cols = ['psd_' + str(idx).zfill(3) for idx in range(1,len(psd_s)+1)]
    psd_s = pd.DataFrame(data=[psd_s],columns=cols)
    
    return psd_s, f_idx