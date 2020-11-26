#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Ensemble of functions to compute acoustic descriptors from 1D signals

"""

from scipy.signal import periodogram, welch
import pandas as pd
import numpy as np

#### Importation from internal modules
from maad.sound import envelope
from maad.util import wav2pressure, mean_dB, pressure2Leq, wav2Leq, get_unimode

#=============================================================================
def psd(s, fs, nperseg=256, method='welch', window='hanning', nfft=None, tlims=None):
    """ 
    Estimates power spectral density of 1D signal using Welch's or periodogram methods. 
    
    Parameters
    ----------
    s: 1D array 
        Input signal to process 
    fs: float, optional
        Sampling frequency of audio signal
    nperseg: int, optional
        Length of segment for 'welch' method, default is 256
    window : string, default is 'hanning
        Name of the window used for the short fourier transform.
    nfft: int, optional
        Length of FFT for periodogram method. If None, length of signal will be used.
        Length of FFT for welch method if zero padding is desired. If None, length of nperseg will be used.
    method: {'welch', 'periodogram'}
        Method used to estimate the power spectral density of the signal
    tlims: tuple of ints or floats
        Temporal limits to compute the power spectral density in seconds (s)
        If None, estimates for the complete signal will be computed.
        Default is 'None'
    
    Returns
    -------
    psd: pandas Series
        Estimate of power spectral density
    f_idx: pandas Series
        Index of sample frequencies
    
    Notes
    -----
    This is a wrapper that uses functions from Scipy. In particular the scipy.signal module
    
    Examples
    --------
    >>> s, fs = sound.load('spinetail.wav')
    >>> psd, f_idx = features.psd(s, fs, nperseg=512)
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

#=============================================================================
def rms(s):
    """
    Computes the root-mean-square (RMS) level of an input signal

    Parameters
    ----------
    s : 1D array
        Input signal to process

    Returns
    -------
    rms: float
        Root mean square of input signal
    
    Examples
    --------
    >>> s, fs = sound.load('spinetail.wav')
    >>> rms = features.rms(s)
    
    """
    return np.sqrt(np.mean(s**2))

#=============================================================================
def skewness (x, axis=None):
    """
    Calcul the skewness (asymetry) of a signal x
    
    Parameters
    ----------
    x : ndarray of floats 
        1d signal or 2d matrix
        
    axis : integer, optional, default is None
        select the axis to compute the kurtosis
        The default is to compute the mean of the flattened array.
                            
    Returns
    -------    
    ku : float or ndarray of floats
        skewness of x 
        
        if x is a 1d vector => single value
        
        if x is a 2d matrix => array of values corresponding to the number of
        points in the other axis
        
    """
    if isinstance(x, (np.ndarray)) == True:
        if axis is None:
            # flatten the array
            Nf = len(np.ndarray.flatten((x)))
        else:
            Nf = x.shape[axis]
        mean_x =  np.mean(x, axis=axis)
        std_x = np.std(x, axis=axis)
        z = x - mean_x[..., np.newaxis]
        sk = (np.sum(z**3, axis=axis)/(Nf-1))/std_x**3
    else:
        print ("WARNING: type of x must be ndarray") 
        sk = None

    # test if ku is an array with a single value
    if (isinstance(sk, (np.ndarray)) == True) and (len(sk) == 1):
        sk = float(sk)

    return sk

#=============================================================================
def kurtosis (x, axis=None):
    """
    Calcul the kurtosis (tailedness or curved or arching) of a signal x
    
    Parameters
    ----------
    x : ndarray of floats 
        1d signal or 2d matrix       
    axis : integer, optional, default is None
        select the axis to compute the kurtosis
        The default is to compute the mean of the flattened array.
                            
    Returns
    -------    
    ku : float or ndarray of floats
        kurtosis of x 
        
        if x is a 1d vector => single value
        
        if x is a 2d matrix => array of values corresponding to the number of
        points in the other axis
    """
    if isinstance(x, (np.ndarray)) == True:
        if axis is None:
            # flatten the array
            Nf = len(np.ndarray.flatten((x)))
        else:
            Nf = x.shape[axis]
        mean_x =  np.mean(x, axis=axis)
        std_x = np.std(x, axis=axis)
        z = x - mean_x[..., np.newaxis]
        ku = (np.sum(z**4, axis=axis)/(Nf-1))/std_x**4
    else:
        print ("WARNING: type of x must be ndarray") 
        ku = None
        
    # test if ku is an array with a single value
    if (isinstance(ku, (np.ndarray)) == True) and (len(ku) == 1):
        ku = float(ku)
       
    return ku

#=============================================================================
def zero_crossing_rate(s, fs):
    """
    Compute the Zero Crossing Rate of an audio signal.
    
    Parameters
    ----------
    s : 1D array
        Audio to process (wav)
    fs : float
        Sampling frequency of the audio (Hz)

    Returns
    ------- 
    zcr : float   
        number of zero crossing /s

    Note
    ----
    From wikipedia :
    The zero-crossing rate is the rate of sign-changes along a signal, i.e., 
    the rate at which the signal changes from positive to zero to negative or 
    from negative to zero to positive.[1] This feature has been used heavily 
    in both speech recognition and music information retrieval, 
    being a key feature to classify percussive sounds
    
    """
    zero_crosses = np.nonzero(np.diff(s > 0))[0]
    duration = len(s) / fs
    zcr = 1/duration * len(zero_crosses)
    
    return zcr


    