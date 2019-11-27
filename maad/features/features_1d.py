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
    Estimates power spectral density of 1D signal using Welch's or periodogram methods. 
    Note: this is a wrapper function that uses functions from scipy.signal module
    
    Parameters
    ----------
        s: 1D array 
            Input signal to process 
        fs: float, optional
            Sampling frequency of audio signal
        nperseg: int, optional
            Lenght of segment for 'welch' method, default is 256
        nfft: int, optional
            Length of FFT for periodogram method. If None, length of signal will be used.
            Length of FFT for welch method if zero padding is desired. If None, length of nperseg will be used.
        method: {'welch', 'periodogram'}
            Method used to estimate the power spectral density of the signal
        tlims: tuple of ints or floats
            Temporal limits to compute the power spectral density in seconds (s)
    Returns
    -------
        psd: pandas Series
            Estimate of power spectral density
        f_idx: pandas Series
            Index of sample frequencies
    Example
    -------
        s, fs = sound.load('spinetail.wav')
        psd, f_idx = psd(s, fs, nperseg=512)
    """
    
    if tlims is not None:
    # trim audio signal
        try:
            s = s[int(tlims[0]*fs): int(tlims[1]*fs)]
        except:
            raise Exception('length of tlims tuple should be 2')
    
    
    if method=='welch':
        f_idx, psd_s = welch(s, fs, window, nperseg, nfft)
    
    elif method=='periodogram':
        f_idx, psd_s = periodogram(s, fs, window, nfft, scaling='spectrum')
        
    else:
        raise Exception("Invalid method. Method should be 'welch' or 'periodogram' ")
        

    index_names = ['psd_' + str(idx).zfill(3) for idx in range(1,len(psd_s)+1)]
    psd_s = pd.Series(psd_s, index=index_names)
    f_idx = pd.Series(f_idx, index=index_names)
    return psd_s, f_idx