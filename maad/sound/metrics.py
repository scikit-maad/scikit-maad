#!/usr/bin/env python
""" 
Collection of metrics functions to evaluate audio preprocessing methods
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

# Import external modules
from maad.sound import envelope, avg_power_spectro, remove_background_along_axis
from maad.util import power2dB, mean_dB, get_unimode, running_mean

#%%
# =============================================================================
# public functions
# =============================================================================
def temporal_snr (s, mode ='fast', Nt=512) :
    """
    Compute the signal to noise ratio (SNR) of an audio signal in the time domain.

    Parameters
    ----------
    s : 1D array
        Audio to process
    mode : str, optional, default is `fast`
        Select the mode to compute the envelope of the audio waveform
        `fast` : The sound is first divided into frames (2d) using the 
        function _wave2timeframes(s), then the max of each frame gives a 
        good approximation of the envelope.
        `Hilbert` : estimation of the envelope from the Hilbert transform. 
        The method is slow
    Nt : integer, optional, default is 512
        Size of each frame. The largest, the highest is the approximation.
    
    Returns
    -------
    ENRt : float
        Total energy in dB computed in the time domain
    BGNt : float
        Estimation of the background energy (dB) computed in the time domain
    SNRt: float
        Signal to noise ratio (dB) computed in the time domain 
        SNRt = ENRt - BGNt

    References
    ----------
    .. [1] Towsey, Michael (2013), Noise Removal from Waveforms and Spectrograms Derived from Natural Recordings of the Environment. Queensland University of Technology, Brisbane.
    .. [2] Towsey, Michael (2017),The calculation of acoustic indices derived from long-duration recordings of the naturalenvironment. Queensland University of Technology, Brisbane.

    Examples
    --------
    >>> s, fs = maad.sound.load('../data/guyana_tropical_forest.wav')
    >>> _,_,snr = maad.sound.temporal_snr(s)
    >>> snr
    1.5744987447774665

    """
    # Envelope
    env = envelope(s, mode=mode, Nt=Nt)
    # linear to power dB
    envdB = power2dB(env**2)
    # total energy estimation. 
    ENRt = mean_dB(envdB, axis=1)
    # Background noise estimation
    BGNt = get_unimode (envdB, mode ='median')
    # Signal to Noise ratio estimation
    SNRt = ENRt - BGNt

    return ENRt, BGNt, SNRt

#%%
def spectral_snr (Sxx_power) :
    """
    Compute the signal to noise ratio (SNR) of an audio from its spectrogram
    in the time-frequency domain.

    Parameters
    ----------
    Sxx_power : 2D array
        Power spectrogram density [Matrix] to process.
        
    Returns
    -------
    ENRf : float
        Total energy in dB computed in the frequency domain which corresponds
        to the average of the power spectrogram then the sum of the average
    BGNf : float
        Estimation of the background energy (dB) computed based on the estimation
        of the noise profile of the power spectrogram (2d)
    SNRf: float
        Signal to noise ratio (dB) computed in the frequency domain 
        SNRf = ENRf - BGNf
    ENRf_per_bin : vector of floats
        Energy in dB per frequency bin
    BGNf_per_bin : vector of floats
        Background (noise profile) energy in dB per frequency bin
    SNRf_per_bin : vector of floats  
        Signal to noise ratio per frequency bin
        
    References
    ----------
    ..[1] Towsey, Michael (2013), Noise Removal from Waveforms and Spectrograms 
          Derived from Natural Recordings of the Environment. 
          Queensland University of Technology, Brisbane.
    ..[2] Towsey, Michael (2017),The calculation of acoustic indices derived 
          from long-duration recordings of the naturalenvironment.
          Queensland University of Technology, Brisbane.

    Examples
    --------
    >>> s, fs = maad.sound.load('../data/guyana_tropical_forest.wav')
    >>> Sxx_power,_,_,_ = maad.sound.spectrogram (s, fs)  
    >>> _, _, snr, _, _, _ = maad.sound.spectral_snr(Sxx_power)
    >>> snr
    4.084065436435541
    
    """
    # average Sxx_power along time axis
    ENRf_per_bin = avg_power_spectro(Sxx_power)
    # compute total energy in dB
    ENRf = power2dB(sum(ENRf_per_bin))
    # Extract the noise profile (BGNf_per_bin) from the spectrogram Sxx_power
    _, noise_profile = remove_background_along_axis(Sxx_power, mode='median',axis=1) 
    # smooth the profile by removing spurious thin peaks (less than 5 pixels wide)
    noise_profile = running_mean(noise_profile,N=5)
    # noise_profile (energy) into dB
    BGNf_per_bin= power2dB(noise_profile)
    # compute noise/background energy in dB
    BGNf = power2dB(sum(noise_profile))
    # compute signal to noise ratio
    SNRf = ENRf - BGNf 
    # compute SNR_per_bin
    SNRf_per_bin = ENRf_per_bin - ENRf_per_bin

    return ENRf, BGNf, SNRf, ENRf_per_bin, BGNf_per_bin,SNRf_per_bin 

#%%
def sharpness (Sxx) :
    """ 
    Compute the sharpness of a spectrogram
     
    Parameters 
    ---------- 
    Sxx : 2d ndarray of scalars 
        Spectrogram (or image) 
        
    Returns
    -------
    sharpness : scalar
        sharpness of the spectrogram (or image)
        
    Examples
    --------
    >>> s, fs = maad.sound.load('../data/guyana_tropical_forest.wav')
    >>> Sxx_power,_,_,_ = maad.sound.spectrogram (s, fs)  
    >>> sharp = maad.sound.sharpness(Sxx_power)
    >>> sharp
    1.1930709869950632e-05
    
    """
    
    Gt = np.gradient(Sxx, edge_order=1, axis=1)
    Gf = np.gradient(Sxx, edge_order=1, axis=0)
    S = np.sqrt(Gt**2+ Gf**2)
    sharpness=sum(sum(S))/(Gt.shape[0]*Gt.shape[1])
    return sharpness   
 
