#!/usr/bin/env python
""" 
Collection of functions to transform audio signals : Take the envelope
"""
#
# Authors:  Juan Sebastian ULLOA <lisofomia@gmail.com>
#           Sylvain HAUPERT <sylvain.haupert@mnhn.fr>
#
# License: New BSD License

# =============================================================================
# Load the modules
# =============================================================================
# Import external modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.signal import periodogram, welch

# import internal modules
from maad.sound import wave2frames

#%%
# =============================================================================
# public functions
# =============================================================================
def envelope (s, mode='fast', Nt=32):
    """
    Calcul the envelope of a sound waveform (1d)
    
    Parameters
    ----------
    s : 1d ndarray of floats 
        Vector containing sound waveform 
    mode : str, optional, default is `fast`
        - `fast` : The sound is first divided into frames (2d) using the 
            function wave2timeframes(s), then the max of each frame gives a 
            good approximation of the envelope.
        - `Hilbert` : estimation of the envelope from the Hilbert transform. 
            The method is slow
    Nt : integer, optional, default is `32`
        Size of each frame. The largest, the highest is the approximation.
                  
    Returns
    -------    
    env : 1d ndarray of floats
        Envelope of the sound
        
    References
    ----------
    .. [1] Towsey, Michael (2013), Noise Removal from Waveforms and Spectrograms Derived from Natural Recordings of the Environment. Queensland University of Technology, Brisbane.
    .. [2] Towsey, Michael (2017),The calculation of acoustic indices derived from long-duration recordings of the natural environment. Queensland University of Technology, Brisbane.
    
    Examples
    --------
    >>> s,fs = maad.sound.load("../data/guyana_tropical_forest.wav")
    >>> env_fast = maad.sound.envelope(s, mode='fast', Nt=32)
    >>> env_fast
    array([0.2300415 , 0.28643799, 0.24285889, ..., 0.3059082 , 0.20040894,
       0.26074219])
    
    >>> env_hilbert = maad.sound.envelope(s, mode='hilbert')
    >>> env_hilbert
    array([0.06588196, 0.11301711, 0.09201435, ..., 0.18053983, 0.18351906,
       0.10258595])
    
    compute the time vector for the vector wave
    
    >>> import numpy as np
    >>> t = np.arange(0,len(s),1)/fs
    
    compute the time vector for the vector env_fast
    >>> t_env_fast = np.arange(0,len(env_fast),1)*len(s)/fs/len(env_fast)
    
    plot 0.1s of the envelope and 0.1s of the abs(s)
    
    >>> import matplotlib.pyplot as plt
    >>> fig1, ax1 = plt.subplots()
    >>> ax1.plot(t[t<0.1], abs(s[t<0.1]), label='abs(s)')
    >>> ax1.plot(t[t<0.1], env_hilbert[t<0.1], label='env(s) - hilbert option')
    >>> ax1.plot(t_env_fast[t_env_fast<0.1], env_fast[t_env_fast<0.1], label='env(s) - fast option')
    >>> ax1.set_xlabel('Time [sec]')
    >>> ax1.legend()
    """
    if mode == 'fast' :
        # Envelope : take the max (see M. Towsey) of each frame
        frames = wave2frames(s, Nt)
        env = np.max(abs(frames),0) 
    elif mode =='hilbert' :
        # Compute the hilbert transform of the waveform and take the norm 
        # (magnitude) 
        env = np.abs(hilbert(s))  
    else:
        print ("WARNING : choose a mode between 'fast' and 'hilbert'")
        
    return env

#%%
def psd(s, fs, nperseg=256, method='welch', window='hanning', nfft=None, tlims=None,
        display=False):
    """ 
    Estimate the power spectral density of 1D signal using Welch's or periodogram methods. 
    
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
    This is a wrapper that uses functions from Scipy (scipy.org), in particular from the scipy.signal module.
    
    Examples
    --------
    >>> from maad import sound
    >>> s, fs = sound.load('../data/spinetail.wav')
    >>> psd, f_idx = sound.psd(s, fs, nperseg=512, display=True)
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
    
    if display:
        fig, ax = plt.subplots(figsize=(9,5))
        ax.plot(f_idx, psd_s)
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Amplitude')
        
    return psd_s, f_idx

