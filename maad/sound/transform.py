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
import scipy
import resampy

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
    
    Compute the time vector for the vector wave.
    
    >>> import numpy as np
    >>> t = np.arange(0,len(s),1)/fs
    
    Compute the time vector for the vector env_fast.
    
    >>> t_env_fast = np.arange(0,len(env_fast),1)*len(s)/fs/len(env_fast)
    
    Plot 0.1s of the envelope and 0.1s of the abs(s).
    
    >>> import matplotlib.pyplot as plt
    >>> fig1, ax1 = plt.subplots(figsize=(10,4))
    >>> ax1.plot(t[t<0.1], abs(s[t<0.1]), label='abs(s)', lw=0.7)
    >>> ax1.plot(t[t<0.1], env_hilbert[t<0.1], label='env(s) - hilbert option', lw=0.7)
    >>> ax1.plot(t_env_fast[t_env_fast<0.1], env_fast[t_env_fast<0.1], label='env(s) - fast option', lw=0.7)
    >>> ax1.set_xlabel('Time [sec]')
    >>> ax1.set_ylabel('Amplitude')
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
        flims=None, display=False):
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
    flims: tuple of ints or floats
        Spectral limits to compute the power spectral density in Hertz (Hz)
        If None, estimates from 0 to fs/2 will be computed.
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
    
    Specify temporal and spectral limits.
    
    >>> psd, f_idx = sound.psd(s, fs, nperseg=2056, tlims=(5.3, 7.9), flims=(2000, 12000), display=True)
    
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
    if flims is not None:
        psd_s = psd_s[(f_idx>=flims[0]) & (f_idx<=flims[1])]
        f_idx = f_idx[(f_idx>=flims[0]) & (f_idx<=flims[1])]
    
    if display:
        fig, ax = plt.subplots(figsize=(9,5))
        ax.plot(f_idx, psd_s)
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Amplitude')
        
    return psd_s, f_idx
#%%
def resample(data, sr, target_sr, res_type = 'kaiser_best', **kwargs):
    """
    Changes the sample rate of a time series.

    Parameters
    ----------
    data : np.ndarray 
        Mono or stereo signal as NumPy array.

    sr : int
        Time series sampling rate. 

    target_sr : int
        Target sampling rate.

    res_type : str, optional
        Resample method. By default 'kaiser_best', is a high-quality method.
        `res_type='kaiser_fast'` is a faster method.
        `res_type='sinc_window'` is an advanced and custom method of resampling. 
        `res_type='scipy'` is a Fourier transforms method based.

    kwargs : additional keyword arguments
        If `res_type='sinc_window'`, additional keyword arguments to pass to
        `resampy.resample`.

    Returns
    -------
    res_data : np.ndarray 
        Resampling time series.

    See Also
    --------
    resampy.resample
    scipy.signal.resample
    
    Examples
    --------
    Resample a time series from sample rate of 44100 to 55550.
    >>> import numpy as np
    >>> sr = 44100; T = 2.0; target_sr = 2*sr
    >>> t = np.linspace(0, T, int(T*sr))
    >>> data = np.sin(2. * np.pi * 440. *t)
    >>> data_resample = maad.sound.resample(data, sr, target_sr, 'kaiser_fast')
    >>> data.shape, data_resample.shape
    ((88200,), (111100,))
    """
    if sr == target_sr:
        return data
    
    ratio = float(target_sr) / sr
    n_samples = int(np.ceil(data.shape[-1] * ratio))

    if res_type == 'scipy':
        res_data = scipy.signal.resample(data, n_samples, axis = -1)
    
    else:
        res_data = resampy.resample(data, sr, target_sr, filter=res_type, axis=-1, **kwargs)
    return np.ascontiguousarray(res_data, dtype=data.dtype)

#%%
def slice_audio(data, sr, min_t, max_t, pad=False, pad_constant=0):
    """
    Slices a time series, from a initial time `min_t` to an ending time `max_t`.  
    If the target duration `duration =` is larger than the original duration and `pad = TRue`, 
    the time series is padded with a constant value `pad_constant = 0`.

    Parameters
    ----------
    data : np.ndarray 
        Mono or stereo signal as NumPy array.

    sr : int
        Time series sampling rate.

    min_t : float
        Initial time. If initial time `min_t < 0` and `pad=True`, this time is added to the 
        beginning of the data slice.

    max_t : float
        Ending time of the data slice.
        
    pad : bool, optional
        If true, the time series is padded with a constant value `pad_constant`. Default is False.
        
    pad_constant : 
        It is the constant with which the time series is padded. 
        
    Returns
    -------
    data_slice : np.ndarray 
        Time series with duration `duration = max_t - min_t`.

    See Also
    --------
    numpy.pad
    
    Examples
    --------
    Pad a time series from 2 seconds to 5 seconds.
    >>> import numpy as np
    >>> sr = 44100; T = 2.0
    >>> t = np.linspace(0, T, int(T*sr))
    >>> data = np.sin(2. * np.pi * 440. *t)
    >>> data_slice = maad.sound.slice_wav(data, sr, min_t = 0., max_t = 5., pad=True)
    >>> data_slice.shape[-1]/sr
    5.0
    """