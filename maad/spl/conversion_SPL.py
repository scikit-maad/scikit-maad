#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
Collection of functions to convert audio and voltage data to Sound Pressure 
Level (SPL in Pascal) and Leq (Continuous Equivalent SPL).
Functions are adapted from the algorithm proposed in 
Merchant, N.D., Fristrup, K.M., Johnson, M.P., Tyack, P.L., Witt, M.J., Blondel, P. and Parks, S.E. (2015), 
Measuring acoustic habitats. Methods Ecol Evol, 6: 257-265. `DOI: 10.1111/2041-210X.12330 <https://doi.org/10.1111/2041-210X.12330>`_
"""   
#
# Authors:  Juan Sebastian ULLOA <lisofomia@gmail.com>
#           Sylvain HAUPERT <sylvain.haupert@mnhn.fr>
#
# License: New BSD License 


#****************************************************************************
# -------------------       Load modules           ---------------------------
#****************************************************************************

# Import external modules

import numpy as np 
from numpy import sum, log10, abs, mean, sqrt

# min value
import sys
_MIN_ = sys.float_info.min

#****************************************************************************
# -------------------       Functions             ---------------------------
#****************************************************************************

#%%
def wav2volt (wave, Vadc=2): 
    """
    Convert an audio signal amplitude to Volts.

    .. warning::
       **Important:** Previous versions of this function incorrectly calculated the
        voltage conversion by omitting the division by 2 for the `Vadc` parameter.
        This resulted in voltage values that were twice as large as they should.
        If you want to correct the voltage values obtained with the previous version,
        you can simply divide the results by 2. 
        In dB SPL, this lead to a 6 dB increase in the calculated sound pressure level.
        If you want to correct the SPL values obtained with the previous version,
        you can simply subtract 6 dB from the old results.
        This has been corrected in the version 1.5.1. Ensure you are using the
        latest version (>=1.5.1) to obtain accurate voltage values.
    
    Parameters
    ----------
    wave : ndarray-like or scalar 
        wave should already be normalized between -1 to 1 (depending on the number of bits)
        take the output of the function sound.load of maad module
        ndarray-like or scalar containing the raw sound waveform 
        
    Vadc : scalar, optional, default is 2Vpp (=>+/-1V)
        Maximal voltage (peak to peak) converted by the analog to digital convertor ADC  
                
    Returns
    -------
    volt : ndarray-like or scalar
        ndarray-like or scalar containing the sound waveform in volt
        
    Examples
    --------
    >>> w, fs = maad.sound.load('../data/cold_forest_daylight.wav') 
    >>> maad.spl.wav2volt(wave=w, Vadc=2)
        array([ 0.02155849,  0.02570888,  0.02583096, ..., -0.0082877 ,
        -0.00219072, -0.00377764])
    """
    # force to be ndarray
    wave = np.asarray(wave)
    
    volt =wave * Vadc/2
    return volt

#%%
def volt2pressure(volt, gain, sensitivity=-35, dBref=94):  
    """
    Convert Volts to instantaneous sound pressure (p [Pa]).
    
    Parameters
    ----------
    volt : ndarray-like or scalar
        ndarray-like or scalar containing the sound waveform in volt
        
    gain : integer
        Total gain applied to the sound (preamplifer + amplifier)
    
    sensitivity : float, optional, default is -35 (dB/V)
        Sensitivity of the microphone
    
    dBref : integer, optional, default is 94 (dBSPL)
        Pressure sound level used for the calibration of the microphone 
        (usually 94dB, sometimes 114dB)
                
    Returns
    -------
    p : ndarray-like or scalar
        ndarray-like or scalar containing the sound waveform in pressure (Pa)
        
    Examples
    --------
    >>> w, fs = maad.sound.load('../data/cold_forest_daylight.wav') 
    >>> v = maad.spl.wav2volt(wave=w)
    >>> maad.spl.volt2pressure(volt=v, gain=42) 
        array([ 0.00481491,  0.00574187,  0.00576913, ..., -0.00185099,
        -0.00097856, -0.00168741])   
    
    Same result with the function wav2pressure
    
    >>> maad.spl.wav2pressure(wave=w, gain=42)
        array([ 0.00481491,  0.00574187,  0.00576913, ..., -0.00185099,
        -0.00097856, -0.00168741])
    
    """
    # force to be ndarray
    volt = np.asarray(volt)
    
    # wav to instantaneous sound pressure level (SPL)
    # coefficient to convert Volt into pressure (Pa)
    coeff = 1/10**(sensitivity/20) 
    p = volt * coeff / 10**(gain/20)
    return p

#%%
def wav2pressure (wave, gain, Vadc = 2, sensitivity=-35, dBref=94): 
    """
    Convert wave amplitude to instantaneous sound pressure (p [Pa]).

    .. warning::
       **Important:** Previous versions of this function incorrectly calculated the
        pressure conversion by omitting the division by 2 for the `Vadc` parameter.
        This resulted in pressure values that were twice as large as they should.
        If you want to correct the pressure values obtained with the previous version,
        you can simply divide the results by 2. 
        In dB SPL, this would lead to a 6 dB increase in the calculated sound pressure level.
        If you want to correct the SPL values obtained with the previous version,
        you can simply subtract 6 dB from the old results.
        This has been corrected in the version 1.5.1. Ensure you are using the
        latest version (>=1.5.1) to obtain accurate pressure values.
    
    Parameters
    ----------
    wave : ndarray-like or scalar 
        wave should already be normalized between -1 to 1 (depending on the number of bits)
        take the output of the function sound.load of maad module
        ndarray-like or scalar containing the raw sound waveform 
        
    gain : integer
        Total gain applied to the sound (preamplifer + amplifier)
        
    Vadc : scalar, optional, default is 2Vpp (=>+/-1V)
        Maximal voltage (peak to peak) converted by the analog to digital convertor ADC    

    sensitivity : float, optional, default is -35 (dB/V)
        Sensitivity of the microphone
    
    dBref : integer, optional, default is 94 (dBSPL)
        Pressure sound level used for the calibration of the microphone 
        (usually 94dB, sometimes 114dB)
                
    Returns
    -------
    p : ndarray-like or scalar
        ndarray-like or scalar containing the sound waveform in pressure (Pa)
        

    Examples
    --------
    >>> w, fs = maad.sound.load('../data/cold_forest_daylight.wav') 
    >>> maad.spl.wav2pressure(wave=w, gain=42)
        array([ 0.00481491,  0.00574187,  0.00576913, ..., -0.00185099,
        -0.00097856, -0.00168741])
    
    Same result with 2 functions
    
    >>> v = maad.spl.wav2volt(wave=w)
    >>> maad.spl.volt2pressure(volt=v, gain=42) 
        array([ 0.00481491,  0.00574187,  0.00576913, ..., -0.00185099,
        -0.00097856, -0.00168741])        
    
    """
    # force to be ndarray
    wave = np.asarray(wave)

    v = wav2volt(wave, Vadc)
    p = volt2pressure(v, gain, sensitivity, dBref)
    return p

#%%
def pressure2dBSPL (p, pRef=20e-6): 
    """
    Convert sound pressure (p [Pa]) to sound pressure level (L [dB]).
    
    Parameters
    ----------
    p : ndarray-like or scalar
        Array or scalar containing the sound pressure in Pa 
                
    pRef : Sound pressure reference in the medium (air:20e-6 Pa, water:1e-6 Pa)
                
    Returns
    -------
    L : ndarray-like or scalar
        Array or scalar containing the sound pressure level (L [dB])
        

    Examples
    --------
    >>> w, fs = maad.sound.load('../data/cold_forest_daylight.wav') 
    >>> p = maad.spl.wav2pressure(wave=w, gain=42)
    
    Get instantaneous sound pressure level (L)
    
    >>> maad.spl.pressure2dBSPL(abs(p))
        array([47.63116868, 49.16046498, 49.20160951, ..., 39.32747759,
        33.79115165, 38.52380598])
    
    Get equivalent sound pressure level (Leq) from the RMS of the pressure signal
    
    >>> p_rms = maad.util.rms(p)
    >>> maad.spl.pressure2dBSPL(p_rms)  
        48.51429086346293
        
    """    
    # force to be ndarray
    p = np.asarray(p)
    
    # if p ==0 set to MIN
    p[p==0] = _MIN_
    
    # Take the log of the ratio pressure/pRef
    L = 20*log10(p/pRef) 
    return L

#%%
def dBSPL2pressure (L, pRef=20e-6): 
    """
    Convert sound pressure level (L [dB]) to sound pressure (p [Pa]).
    
    Parameters
    ----------
    L : ndarray-like or scalar
        Array or scalar containing the sound pressure level (L [dB])
                
    pRef : Sound pressure reference in the medium (air:20e-6 Pa, water:1e-6 Pa)
                
    Returns
    -------
    p : ndarray-like or scalar
        Array or scalar containing the sound pressure in Pa 
        
    Examples
    --------
    >>> w, fs = maad.sound.load('../data/cold_forest_daylight.wav') 
    >>> p = maad.spl.wav2pressure(wave=w, gain=42)
    >>> p_rms = maad.util.rms(p)
    >>> print(p_rms)
        0.0053302126891706676
    >>> L = maad.spl.pressure2dBSPL(p_rms)  
    >>> maad.spl.dBSPL2pressure(L)
        spl.dBSPL2pressure(L)
    """    
    # force to be ndarray
    L = np.asarray(L)
    
    # dB SPL to pressure
    p = 10**(L/20)*pRef
    return p

#%%
def wav2dBSPL (wave, gain, Vadc=2, sensitivity=-35, dBref=94, pRef=20e-6): 
    """
    Convert wave amplitude to instantaneous sound pressure level (L [dB SPL]).

    .. warning::
       **Important:** Previous versions of this function incorrectly calculated the
        pressure conversion by omitting the division by 2 for the `Vadc` parameter.
        In dB SPL, this would lead to a 6 dB increase in the calculated sound pressure level.
        If you want to correct the SPL values obtained with the previous version,
        you can simply subtract 6 dB from the old results.
        This has been corrected in the version 1.5.1. Ensure you are using the
        latest version (>=1.5.1) to obtain accurate dB SPL values.
    
    Parameters
    ----------
    wave : ndarray-like or scalar 
        wave should already be normalized between -1 to 1 (depending on the number of bits)
        take the output of the function sound.load of maad module
        ndarray-like or scalar containing the raw sound waveform 
        
    gain : integer
        Total gain applied to the sound (preamplifer + amplifier)
        
    Vadc : scalar, optional, default is 2Vpp (=>+/-1V)
        Maximal voltage (peak to peak) converted by the analog to digital convertor ADC     

    sensitivity : float, optional, default is -35 (dB/V)
        Sensitivity of the microphone
    
    dBref : integer, optional, default is 94 (dBSPL)
        Pressure sound level used for the calibration of the microphone 
        (usually 94dB, sometimes 114dB)
        
    pRef : Sound pressure reference in the medium (air:20e-6 Pa, water:1e-6 Pa)
                
    Returns
    -------
    L : ndarray-like or scalar
        ndarray-like or scalar containing the sound waveform in dB SPL (Sound Pressure level in dB)
        

    Examples
    --------
    >>> w, fs = maad.sound.load('../data/cold_forest_daylight.wav') 
    >>> L = maad.spl.wav2dBSPL(wave=w, gain=42)
    
    Get an approximate of the equivalent sound pressure level (Leq)
        
    >>> maad.util.mean_dB(L)
        48.51429086346294
        
    Get equivalent sound pressure level (Leq) from the dedicated function
        
    >>> Leq = maad.spl.wav2leq(w, fs, gain=42, dt=1)
    >>> Leq_mean = maad.util.mean_dB(Leq)
    >>> Leq_mean
        48.55488267086038
    
    """       
    # force to be ndarray
    wave = np.asarray(wave)
    
    p = wav2pressure(wave, gain, Vadc, sensitivity, dBref)
    p_abs = abs(p)
    L = pressure2dBSPL(p_abs, pRef)
    return L

#%%
def amplitude2dBSPL (s, gain, Vadc=2, sensitivity=-35, dBref=94, pRef=20e-6): 
    """
    Convert signal (amplitude) to instantaneous sound pressure level (L [dB SPL]).

    .. warning::
       **Important:** Previous versions of this function incorrectly calculated the
        volt conversion by omitting the division by 2 for the `Vadc` parameter.
        In dB SPL, this would lead to a 6 dB increase in the calculated sound pressure level.
        If you want to correct the SPL values obtained with the previous version,
        you can simply subtract 6 dB from the old results.
        This has been corrected in the version 1.5.1. Ensure you are using the
        latest version (>=1.5.1) to obtain accurate dB SPL values.
    
    Parameters
    ----------
    s : ndarray-like or scalar 
        s is an amplitude signal (not energy signal : s² )
        
    gain : integer
        Total gain applied to the sound (preamplifer + amplifier)
        
    Vadc : scalar, optional, default is 2Vpp (=>+/-1V)
        Maximal voltage (peak to peak) converted by the analog to digital convertor ADC     

    sensitivity : float, optional, default is -35 (dB/V)
        Sensitivity of the microphone
    
    dBref : integer, optional, default is 94 (dBSPL)
        Pressure sound level used for the calibration of the microphone 
        (usually 94dB, sometimes 114dB)
        
    pRef : Sound pressure reference in the medium (air:20e-6 Pa, water:1e-6 Pa)
                
    Returns
    -------
    L : ndarray-like or scalar
        ndarray-like or scalar containing the sound waveform in dB SPL (Sound Pressure level in dB)
    
    See also
    --------
    power2dBSPL

    Examples
    --------
    >>> import numpy as np
    >>> w, fs = maad.sound.load('../data/cold_forest_daylight.wav') 
    >>> Sxx_amplitude,tn,fn,_ = maad.sound.spectrogram (w, fs, nperseg=1024, mode='amplitude')  
    >>> S_amplitude_mean = np.mean(Sxx_amplitude, axis=1)
    
    Get instantaneous sound pressure level (L).
    
    >>> maad.spl.amplitude2dBSPL(S_amplitude_mean, gain=42)    
        array([ 33.54362056,  38.35057445,  35.93645686,  33.5848702 ,
                30.41177332,  27.22906075,  25.42079962,  24.42677998,
                22.89529647,  21.24902085,  20.75161999,  20.25268302,
                ...
                -12.18064124, -12.32827014, -12.48541472, -12.51967103,
                -12.54814367, -12.54536435, -12.75677825, -12.77745939])
    """       
    # force to be ndarray
    s = np.asarray(s)
    
    L = wav2dBSPL (s, gain, Vadc, sensitivity, dBref, pRef)

    return L


#%%
def power2dBSPL (P, gain, Vadc=2, sensitivity=-35, dBref=94, pRef=20e-6): 
    """
    Convert power (amplitude²) to sound pressure level (L [dB]).

    .. warning::
       **Important:** Previous versions of this function incorrectly calculated the
        volt conversion by omitting the division by 2 for the `Vadc` parameter.
        In dB SPL, this would lead to a 6 dB increase in the calculated sound pressure level.
        If you want to correct the SPL values obtained with the previous version,
        you can simply subtract 6 dB from the old results.
        This has been corrected in the version 1.5.1. Ensure you are using the
        latest version (>=1.5.1) to obtain accurate dB SPL values.    
    
    Parameters
    ----------
    P : ndarray-like or scalar
        ndarray-like or scalar containing the power signal (P), for instance 
        Sxx_power, the power spectral density (PSD)

    gain : integer
        Total gain applied to the sound (preamplifer + amplifier)
        
    Vadc : scalar, optional, default is 2Vpp (=>+/-1V)
        Maximal voltage (peak to peak) converted by the analog to digital convertor ADC     

    sensitivity : float, optional, default is -35 (dB/V)
        Sensitivity of the microphone
    
    dBref : integer, optional, default is 94 (dBSPL)
        Pressure sound level used for the calibration of the microphone 
        (usually 94dB, sometimes 114dB)
                
    pRef : Sound pressure reference in the medium (air:20e-6 Pa, water:1e-6 Pa)
                
    Returns
    -------
    L : ndarray-like or scalar
        ndarray-like or scalar containing the sound pressure level (L [dB])
        
    See also
    --------
    amplitude2dBSPL
        
    Examples
    --------
    >>> import numpy as np
    >>> w, fs = maad.sound.load('../data/cold_forest_daylight.wav') 
    >>> Sxx_power,tn,fn,_ = maad.sound.spectrogram (w, fs, nperseg=1024, mode='psd')  
    >>> S_power_mean = np.mean(Sxx_power, axis=1)
    
    Get instantaneous sound pressure level (L).
    
    >>> maad.spl.power2dBSPL(S_power_mean, gain=42)    
        array([ 3.55439604e+01,  3.94219755e+01,  3.71509454e+01,  3.54860553e+01,
                3.20633292e+01,  2.85071055e+01,  2.65508217e+01,  2.56607733e+01,
                2.43080132e+01,  2.24405108e+01,  2.18647044e+01,  2.14653511e+01,
                ...
                -1.11242771e+01, -1.12703297e+01, -1.14085678e+01, -1.14487227e+01,
                -1.14906343e+01, -1.14980091e+01, -1.16907591e+01, -1.17027481e+01])
    """    
    # force to be ndarray
    P = np.asarray(P)
    # convert power (energy) to amplitude
    w = sqrt(P)
    # convert amplitude to dB sPL
    L = wav2dBSPL (w, gain, Vadc, sensitivity, dBref, pRef)
    
    return L

################################## Leq ########################################
#%%
def wav2leq (wave, f, gain, Vadc=2, dt=1, sensitivity=-35, dBref = 94, pRef = 20e-6): 
    """
    Convert wave to Equivalent Continuous Sound Pressure level (Leq [dB SPL]).

    .. warning::
       **Important:** Previous versions of this function incorrectly calculated the
        volt conversion by omitting the division by 2 for the `Vadc` parameter.
        In dB SPL, this would lead to a 6 dB increase in the calculated sound pressure level.
        If you want to correct the SPL values obtained with the previous version,
        you can simply subtract 6 dB from the old results.
        This has been corrected in the version 1.5.1. Ensure you are using the
        latest version (>=1.5.1) to obtain accurate dB SPL values. 
    
    Parameters
    ----------
    wave : ndarray-like or scalar 
        wave should already be normalized between -1 to 1 (depending on the number of bits)
        take the output of the function sound.load of maad module
        ndarray-like or scalar containing the raw sound waveform 
        
    f : integer
        Sampling frequency in Hz
        
    gain : integer
        Total gain applied to the sound (preamplifer + amplifier)
        
    Vadc : scalar, optional, default is 2Vpp (=>+/-1V)
        Maximal voltage (peak to peak) converted by the analog to digital convertor ADC  
        
    dt : float, optional, default is 1 (second)
        Integration step to compute the Leq (Equivalent Continuous Sound level)

    sensitivity : float, optional, default is -35 (dB/V)
        Sensitivity of the microphone
    
    dBref : integer, optional, default is 94 (dBSPL)
        Pressure sound level used for the calibration of the microphone 
        (usually 94dB, sometimes 114dB)
    
    pRef : Sound pressure reference in the medium (air:20e-6, water:1e-6 Pa)

    Returns
    -------
    Leq : float
        Equivalent Continuous Sound pressure level (Leq [dB SPL])

    Examples
    --------
    >>> w, fs = maad.sound.load('../data/cold_forest_daylight.wav') 
    >>> Leq = maad.spl.wav2leq (w, fs, gain=42)  
    >>> maad.util.mean_dB(Leq)
        48.55488267086038
    """    
    # force to be ndarray
    wave = np.asarray(wave)
    
    # convert in Volt
    volt = wav2volt(wave, Vadc)
    ##### volt amplitude to Leq 
    # wav to RMS
    dN = int(np.floor(dt*f)) # RMS period in number of points
    N_RMS = int(np.floor(len(volt)/dN)) # number of RMS periods
    # reduction of volt vector length
    volt_1D = volt[0:dN*N_RMS]
    # reshape into 2D (the duration of each row is dt)
    volt_2D = volt_1D.reshape(N_RMS,dN)
    # RMS
    volt_RMS = sqrt(mean(volt_2D**2,axis=1))
    # if volt_RMS ==0 set to MIN
    volt_RMS[volt_RMS==0] = _MIN_
    # RMS to Leq (Equivalent Continuous Sound level)
    Leq = 20*log10(volt_RMS) - sensitivity - gain + dBref - 20*np.log10(pRef/20e-6)
    return Leq

#%%
def pressure2leq (p, fs, dt=1, pRef = 20e-6): 
    """
    Convert pressure vector (p [Pa]) to Equivalent Continuous Sound Pressure 
    level (Leq [dB SPL]).
    
    Parameters
    ----------
    p : ndarray-like
        Array containing the sound pressure in Pa 
        
    fs : integer
        Sampling frequency in Hz
        
    dt : float, optional, default is 1 (second)
        Integration step to compute the Leq (Equivalent Continuous Sound level)

    pRef : Sound pressure reference in the medium (air:20e-6, water:1e-6 Pa)
    
    Returns
    -------
    Leq : float
        Equivalent Continuous Sound pressure level (Leq [dB SPL])
        
    Examples
    --------
    >>> w, fs = maad.sound.load('../data/cold_forest_daylight.wav') 
    >>> p = maad.spl.wav2pressure(wave=w, gain=42)
    >>> Leq = maad.spl.pressure2leq (p, fs)
    >>> maad.util.mean_dB(Leq)
        48.534282757580755
        
    """     
    # be sure they are ndarray
    p = np.asarray(p)
    
    ##### wav SPLto Leq 
    # wav to RMS
    dN = int(np.floor(dt*fs)) # RMS period in number of points
    N_RMS = int(np.floor(len(p)/dN)) # number of RMS periods
    # reduction of volt vector length
    p_1D = p[0:dN*N_RMS]
    # reshape into 2D (the duration of each row is dt)
    p_2D = p_1D.reshape(N_RMS,dN)
    # RMS
    p_RMS = sqrt(mean(p_2D**2,axis=1))
    # if p_RMS ==0 set to MIN
    p_RMS[p_RMS==0] = _MIN_
    # RMS to Leq (Equivalent Continuous Sound level)
    Leq = 20*log10(p_RMS/pRef)
    return Leq

#%%
def psd2leq (P, gain, Vadc=2, sensitivity=-35, dBref = 94, pRef = 20e-6):
    """
    Convert Power spectral density (PSD, amplitude²) into 
    Equivalent Continuous Sound pressure level (Leq [dB SPL])

    .. warning::
       **Important:** Previous versions of this function incorrectly calculated the
        volt conversion by omitting the division by 2 for the `Vadc` parameter.
        In dB SPL, this would lead to a 6 dB increase in the calculated sound pressure level.
        If you want to correct the SPL values obtained with the previous version,
        you can simply subtract 6 dB from the old results.
        This has been corrected in the version 1.5.1. Ensure you are using the
        latest version (>=1.5.1) to obtain accurate dB SPL values. 
    
    Parameters
    ----------
    P : ndarray-like (1d)
        ndarray-like containing the Power spectral density (PSD=amplitude²) 
        
    gain : integer
        Total gain applied to the sound (preamplifer + amplifier)
        
    Vadc : scalar, optional, default is 2Vpp (=>+/-1V)
        Maximal voltage (peak to peak) converted by the analog to digital convertor ADC  

    sensitivity : float, optional, default is -35 (dB/V)
        Sensitivity of the microphone
    
    dBref : integer, optional, default is 94 (dBSPL)
        Pressure sound level used for the calibration of the microphone 
        (usually 94dB, sometimes 114dB)

    pRef : Sound pressure reference in the medium (air:20e-6, water:1e-6 Pa)

    Returns
    -------
    Leq : float
        Equivalent Continuous Sound pressure level (Leq [dB SPL])
        
    Examples
    --------
    >>> w, fs = maad.sound.load('../data/cold_forest_daylight.wav') 
    >>> Sxx_power,tn,fn,_ = maad.sound.spectrogram (w, fs)
    >>> S_power_mean = maad.sound.avg_power_spectro(Sxx_power) 
    >>> maad.spl.psd2leq(S_power_mean, gain=42)
        47.537824826354665
        
    """  
    # be sure they are ndarray
    P = np.asarray(P)
    
    # convert P (amplitude²) to pressure²
    P = wav2pressure (sqrt(P), gain, Vadc, sensitivity, dBref)**2
    
    # if P ==0 set to MIN
    P[P==0] = _MIN_
    
    # Energy (pressure^2) to Leq (=> Leq (Equivalent Continuous Sound level) 
    # if the sum is performed on the whole PSD)
    Leq = 10*log10(sum(P)/pRef**2)
    return Leq
    
################################ TEST ########################################
    
## Test the current operating system
if __name__ == "__main__":
    # ############## Import MAAD module
    from pathlib import Path # in order to be Windows/Linux/MacOS compatible
    import os
    
    # change the path to the current path where the script is located
    # Get the current dir of the current file
    dir_path = os.path.dirname(os.path.realpath('__file__'))
    os.chdir(dir_path)
    
    # data directory 
    datadir = (Path(dir_path).parents[1] / "data/").resolve()
    filename = datadir/"cold_forest_daylight.wav"
    
    # Go to the root directory where is the module maad and import maad
    maad_path = Path(dir_path).parents[1]
    os.sys.path.append(maad_path.as_posix())
    from maad import sound, util, spl
    
    ####### Variables
    SENSITIVITY     = -35       # Sensbility microphone (for SM4 it is -35dBV)
    GAIN_TOTAL      = 42        # total amplification gain : preamplifier gain = 26dB and gain = 16dB
    PREF            = 20e-6     # Reference pressure (in the air : 20µPa)
    DELTA_T         = 1         # Leq integration time dt in s
    NFFT            = 1024      # length of the fft
    VADC            = 2         # Maximal voltage (peak to peak) converted by the analog to digital convertor ADC (i.e : +/- 1V peak to peak => Vadc=2V)
    
    # Load the sound
    wav,fs = sound.load(filename=filename, channel='right', detrend=False, verbose=False)
    
    ## conversion into dB SPL 

    print('-------------------------------------------------------------------------')
    print('Leq calculation directly from the sound file (in time or frequency domain)')

    # convert sounds (wave) into SPL and subtract DC offset
    wav = wav - np.mean(wav)
    # wav -> pressure -> Leq
    p = spl.wav2pressure(wav, gain=GAIN_TOTAL, Vadc=VADC, sensitivity=SENSITIVITY, dBref=94)
    Leq = spl.pressure2leq(p, fs, dt=DELTA_T, pRef=PREF)
    print('Leq from volt', util.mean_dB(Leq))

    # wav -> Leq
    wav_Leq2 = spl.wav2leq(wav, f=fs, gain=GAIN_TOTAL, Vadc=VADC, dt=DELTA_T, 
                    sensitivity=SENSITIVITY, dBref=94, pRef=PREF) 
    print('Leq from wav', util.mean_dB(wav_Leq2))

    # Power Density Spectrum : PSD
    # with wav
    from numpy import fft
    P = abs(fft.fft(wav)/len(wav))**2
    # average Leq from PSD
    P_Leq3  = spl.psd2leq(P, gain=GAIN_TOTAL, Vadc=VADC, 
                    sensitivity=SENSITIVITY, dBref=94, pRef=PREF)
    print('Leq from PSD',P_Leq3)

    ##
    # Leq from spectrogram Sxx_power : 
    print('-------------------------------------------------------------------------')
    print('Leq calculation from Spectrogram (time-frequency representation of a sound)')

    # Power Density Spectrogram : Sxx_power
    # with wav
    Sxx_power,tn,fn,_ = sound.spectrogram (wav, fs=fs, nperseg=NFFT, mode='psd')  
    # Sxx_power to S_power
    S_power = np.mean(Sxx_power,axis=1)
    # average Leq from S_power
    P_Leq5  = spl.psd2leq(S_power, gain=GAIN_TOTAL, Vadc=VADC, 
                    sensitivity=SENSITIVITY, dBref=94, pRef=PREF) 
    print('Leq from Sxx_power spectrogram',P_Leq5)

    # total energy from S_power
    energy  = np.sum(S_power)
    P_Leq6  = spl.wav2dBSPL(np.sqrt(energy), gain=GAIN_TOTAL, Vadc=VADC, 
                    sensitivity=SENSITIVITY, dBref=94, pRef=PREF)
    print('Leq from S_power energy',P_Leq6)

    print('')
    print('> The difference with previous Leq calculation is due to the average of '+ 
            'the Sxx_power along the time axis which reduces the noise contribution into the total energy.')
    print('> By increasing the NFFT length (i.e. NFFT=4096), Leq value converges towards previous Leq values') 



    
    
