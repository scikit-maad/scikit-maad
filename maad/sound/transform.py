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
from maad.util import plot_spectrum

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
        raise Exception("Invalid mode. Mode should be 'fast' or 'hilbert' ")
        
    return env

#%%
def spectrum(s, fs, nperseg=256, noverlap=None, nfft=None, window='hanning', method='welch', 
        tlims=None, flims=None, scaling='spectrum', as_pandas_series=False, display=False):
    """ 
    Estimate the power spectral density or power spectrum of 1D signal.
    
    The estimates can be computed using two methods: Welch or periodogram. Welch's method
    divides the signal into segments, computes the power spectral density for each 
    segment and then takes the average between segments. The periodogram method computes
    the power spectral density of the input signal using a defined window 
    (Hanning window by default).
    
    Parameters
    ----------
    s: 1D array 
        Input signal to process 
    
    fs: float, optional
        Sampling frequency of audio signal
    
    nperseg: int, optional
        Length of segment for 'welch' method, default is 256
    
    noverlap: int, optional
        Overlap between segments for Welch's method. If None, noverlap = nperseg/2.
    
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
        
    scaling: {'spectrum', 'density'}, optional
        Choose between power spectrum (units V**2) or power spectral density (units V**2/Hz) scaling.
        Defaults to 'spectrum'.
    
    as_pandas_series: bool
        Return data as a pandas.Series. This is usefull when computing multiple features
        over a signal. Default is False.
        

    
    Returns
    -------
    pxx: pandas Series
        Power spectral density estimate.
    
    f_idx: pandas Series
        Index of sample frequencies.
    
    Notes
    -----
    This is a wrapper that uses functions from Scipy (scipy.org), in particular from the scipy.signal module.
    
    Examples
    --------
    >>> from maad import sound
    >>> s, fs = sound.load('../data/spinetail.wav')
    >>> spec, f_idx = sound.spectrum(s, fs, nperseg=512, display=True)
    
    Specify temporal and spectral limits to get spectral characteristics of the 
    spinetail's song.
    
    >>> spec, f_idx = sound.spectrum(s, fs, nperseg=2056, tlims=(5.3, 7.9), flims=(2000, 12000), display=True)
    
    """
    
    if tlims is not None:
    # trim audio signal
        try:
            s = s[int(tlims[0]*fs): int(tlims[1]*fs)]
        except:
            raise Exception('length of tlims tuple should be 2')
    
    if method=='welch':
        f_idx, pxx = welch(s, fs, window, nperseg, noverlap, nfft, scaling=scaling)
    
    elif method=='periodogram':
        f_idx, pxx = periodogram(s, fs, window, nfft, scaling=scaling)
        
    else:
        raise Exception("Invalid method. Method should be 'welch' or 'periodogram' ")
        
    if flims is not None:
        pxx = pxx[(f_idx>=flims[0]) & (f_idx<=flims[1])]
        f_idx = f_idx[(f_idx>=flims[0]) & (f_idx<=flims[1])]

    if as_pandas_series==True:
        index_names = ['psd_' + str(idx).zfill(3) for idx in range(1,len(pxx)+1)]
        pxx = pd.Series(pxx, index=index_names)
        f_idx = pd.Series(f_idx, index=index_names)
        
    if display:
        plot_spectrum(pxx, f_idx)

    return pxx, f_idx

#%%
def resample(s, fs, target_fs, res_type = 'kaiser_best', **kwargs):
    """
    Changes the sample rate of an audio file or any time series.

    Parameters
    ----------
    s : np.ndarray 
        Mono or stereo signal as NumPy array.

    fs : int
        Time series sampling rate. 

    target_fs : int
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
    
    Resample an audio file from sample rate of 44100 kHz to 22050.
    
    >>> s, fs = sound.load('../data/spinetail.wav')
    >>> s_resamp = sound.resample(s, fs, target_fs=fs/2)
    >>> print('Number of samples - original audio:', s.shape[0], '\n'
    ...       'Number of samples - resampled audio:', s_resamp.shape[0])
    >>> _ = sound.spectrogram(s, fs, display=True)
    >>> _ = sound.spectrogram(s_resamp, fs/2, display=True)

    """
    if fs == target_fs:
        return s
    
    ratio = float(target_fs) / fs
    n_samples = int(np.ceil(s.shape[-1] * ratio))

    if res_type == 'scipy':
        res_data = scipy.signal.resample(s, n_samples, axis = -1)
    
    else:
        res_data = resampy.resample(s, fs, target_fs, filter=res_type, axis=-1, **kwargs)
    return np.ascontiguousarray(res_data, dtype=s.dtype)

#%%
def trim(s, fs, min_t, max_t, pad=False, pad_constant=0):
    """
    Slices a time series, from a initial time `min_t` to an ending time `max_t`.  
    If the target duration `duration =` is larger than the original duration and `pad = True`, 
    the time series is padded with a constant value `pad_constant = 0`.

    Parameters
    ----------
    s : np.ndarray 
        Mono or stereo signal as NumPy array.

    fs : int
        Time series sampling rate.

    min_t : float
        Initial time. If initial time `min_t < 0` and `pad=True`, this time is added to the 
        beginning of the audio slice.

    max_t : float
        Ending time of the audio slice.
        
    pad : bool, optional
        If true, the time series is padded with a constant value `pad_constant`. Default is False.
        
    pad_constant : 
        It is the constant with which the time series is padded. 
        
    Returns
    -------
    s_slice : np.ndarray 
        Time series with duration `duration = max_t - min_t`.

    See Also
    --------
    numpy.pad
    
    Examples
    --------
    
    Slice an audio file from 5 to 8 seconds.
    
    >>> from maad import sound
    >>> s, fs = sound.load('../data/spinetail.wav') 
    >>> s_slice = sound.trim(s, fs, min_t = 5, max_t = 8)
    >>> _ = sound.spectrogram(s_slice, fs, display=True, figsize=(4,6))
    >>> s_slice.shape[0]/fs
    3.0
    """

    min_lim = min_t * fs
    max_lim = max_t * fs
    duration = max_t - min_t
    
    if pad:
        
        if s.ndim == 1:
            
            if (duration * fs) > s.shape[-1]:
                up_lim = int(max_t * fs - s.shape[-1])
                if min_t <=0:
                    low_lim = int(abs(min_t) * fs) 
                    s_slice = np.pad(s, (low_lim, up_lim), 'constant', constant_values=(pad_constant))
                else:
                    low_lim = int(0.0) 
                    s_slice = np.pad(s, (low_lim, up_lim), 'constant', constant_values=(pad_constant))
                    s_slice = s_slice[int(min_lim):]
            else:
                s_slice = s[int(min_lim): int(max_lim)]
        
        else:
            
            s_slice = np.empty((s.shape[0], int(duration * fs)))
            for i, chanel in enumerate(s):
                if (duration * fs) > chanel.shape[-1]:
                    up_lim = int(max_t * fs - chanel.shape[-1])
                    
                    if min_t <=0:
                        low_lim = int(abs(min_t) * fs) 
                        s_slice = np.pad(chanel, (low_lim, up_lim), 'constant', constant_values=(pad_constant))
                    else:
                        low_lim = int(0.0) 
                        s_slice = np.pad(chanel, (low_lim, up_lim), 'constant', constant_values=(pad_constant))
                        s_slice = s_slice[int(min_lim):]
                else:
                    s_slice[i] = s[i][int(min_lim): int(max_lim)]
                    
        return s_slice
    
    else:
        
        if min_t < 0:
            raise ValueError("t_min must be >= 0.")
        
        if s.ndim == 1:
            if (duration * fs) > s.shape[-1]:
                raise ValueError("Target duration is longer than original duration.")
            else:
                s_slice = s[int(min_lim): int(max_lim)]
        else:   
            
            s_slice = np.empty((s.shape[0], int(duration * fs)))
            for i, chanel in enumerate(s):
                if (duration * fs) > chanel.shape[-1]:
                    raise ValueError("Target duration is longer than original duration.")
                else:
                    s_slice[i] = s[i][int(min_lim): int(max_lim)]                
        
        return s_slice