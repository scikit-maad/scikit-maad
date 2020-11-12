#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
SPL conversion functions for scikit-maad
Collection of miscelaneous functions that help to convert wav and volt data
to Sound Pressure Level (SPL in Pascal) and Leq (Continuous Equivalent SPL)
"""   
#
# Authors:  Juan Sebastian ULLOA <lisofomia@gmail.com>
#           Sylvain HAUPERT <sylvain.haupert@mnhn.fr>
#
# License: New BSD License 


"""****************************************************************************
# -------------------       Load modules            ---------------------------
****************************************************************************"""

# Import external modules

import numpy as np 
from numpy import sum, log10, abs, mean, sqrt

# min value
import sys
_MIN_ = sys.float_info.min

"""****************************************************************************
# -------------------       Functions               ---------------------------
****************************************************************************"""

##### wav2volt
def wav2volt (wave, Vadc=2): 
    """
    convert in Volt
    
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
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> maad.util.wav2volt(wave=w, Vadc=2)
        array([ 0.02155849,  0.02570888,  0.02583096, ..., -0.0082877 ,
       -0.00438145, -0.00755528])
    """
    # force to be ndarray
    wave = np.asarray(wave)
    
    volt =wave * Vadc
    return volt

##### volt2pressure
def volt2pressure(volt, gain, sensitivity=-35, dBref=94):  
    """
    convert volt to instantaneous sound pressure (p [Pa])
    
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
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> v = maad.util.wav2volt(wave=w)
    >>> maad.util.volt2pressure(volt=v, gain=42) 
        array([ 0.00962983,  0.01148374,  0.01153826, ..., -0.00370198,
       -0.00195712, -0.00337482])      
    
    Same result with the function wav2pressure
    
    >>> maad.util.wav2pressure(wave=w, gain=42)
        array([ 0.00962983,  0.01148374,  0.01153826, ..., -0.00370198,
       -0.00195712, -0.00337482])
    
    """
    # force to be ndarray
    volt = np.asarray(volt)
    
    # wav to instantaneous sound pressure level (SPL)
    # coefficient to convert Volt into pressure (Pa)
    coeff = 1/10**(sensitivity/20) 
    p = volt * coeff / 10**(gain/20)
    return (p)

#####  wav2pressure 
def wav2pressure (wave, gain, Vadc = 2, sensitivity=-35, dBref=94): 
    """
    convert wave to instantaneous sound pressure (p [Pa])
    
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
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> maad.util.wav2pressure(wave=w, gain=42)
        array([ 0.00962983,  0.01148374,  0.01153826, ..., -0.00370198,
       -0.00195712, -0.00337482])
    
    Same result with 2 functions
    
    >>> v = maad.util.wav2volt(wave=w)
    >>> maad.util.volt2pressure(volt=v, gain=42) 
        array([ 0.00962983,  0.01148374,  0.01153826, ..., -0.00370198,
       -0.00195712, -0.00337482]) 
    
    """
    # force to be ndarray
    wave = np.asarray(wave)

    v = wav2volt(wave, Vadc)
    p = volt2pressure(v, gain, sensitivity, dBref)
    return (p)

#####  pressure2dBSPL
def pressure2dBSPL (p, pRef=20e-6): 
    """
    convert sound pressure (p [Pa]) to sound pressure level (L [dB])
    
    Parameters
    ----------
    p : ndarray-like or scalar
        Array or scalar containing the sound pressure in Pa 
                
    pRef : Sound pressure reference in the medium (air : 20e-6 Pa, water : ?)
                
    Returns
    -------
    L : ndarray-like or scalar
        Array or scalar containing the sound pressure level (L [dB])
        

    Examples
    --------
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> p = maad.util.wav2pressure(wave=w, gain=42)
    
    Get instantaneous sound pressure level (L)
    
    >>> maad.util.pressure2dBSPL(abs(p))
        array([53.65176859, 55.18106489, 55.22220942, ..., 45.3480775 ,
        39.81175156, 44.54440589])
    
    Get equivalent sound pressure level (Leq) from the RMS of the pressure signal
    
    >>> p_rms = maad.features.rms(p)
    >>> maad.util.pressure2dBSPL(p_rms)  
        54.53489077674256      
    """    
    # force to be ndarray
    p = np.asarray(p)
    
    # if p ==0 set to MIN
    p[p==0] = _MIN_
    
    # Take the log of the ratio pressure/pRef
    L = 20*log10(p/pRef) 
    return (L)

#####  dBSPL2pressure
def dBSPL2pressure (L, pRef=20e-6): 
    """
    convert sound pressure level (L [dB]) to sound pressure (p [Pa])
    
    Parameters
    ----------
    L : ndarray-like or scalar
        Array or scalar containing the sound pressure level (L [dB])
                
    pRef : Sound pressure reference in the medium (air : 20e-6 Pa, water : ?)
                
    Returns
    -------
    p : ndarray-like or scalar
        Array or scalar containing the sound pressure in Pa 
        
 
    Examples
    --------
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> p = maad.util.wav2pressure(wave=w, gain=42)
    >>> p_rms = maad.features.rms(p)
    >>> print(p_rms)
        0.010660425378341332
    >>> L = maad.util.pressure2dBSPL(p_rms)  
    >>> maad.util.dBSPL2pressure(L)
        0.010660425378341332 
    """    
    # force to be ndarray
    L = np.asarray(L)
    
    # dB SPL to pressure
    p = 10**(L/20)*pRef
    return (p)

#####  power2dBSPL
def power2dBSPL (P, pRef=20e-6): 
    """
    convert sound pressure power (P [Pa²]) to sound pressure level (L [dB])
    
    Parameters
    ----------
    P : ndarray-like or scalar
        ndarray-like or scalar containing the sound pressure power (P [Pa²])
                
    pRef : Sound pressure reference in the medium (air : 20e-6 Pa, water : ?)
                
    Returns
    -------
    L : ndarray-like or scalar
        ndarray-like or scalar containing the sound pressure level (L [dB])
        

    Examples
    --------
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> p = maad.util.wav2pressure(wave=w, gain=42)
    >>> Pxx,tn,fn,_ = maad.sound.spectrogram (p, fs, nperseg=1024, mode='psd')  
    >>> P = mean(Pxx, axis=1)
    
    Get instantaneous sound pressure level (L)
    
    >>> maad.util.power2dBSPL(P)    
    array([41.56456034, 45.44257539, 43.17154534, 41.50665519, 38.08392914,
           34.52770543, 32.57142163, 31.68137318, 30.32861314, 28.46111069,
           27.88530431, 27.48595098, 26.96673216, 25.88241843, 24.93524547,
           ...
           -5.24972979, -5.38796789, -5.42812278, -5.47003443, -5.47740917,
           -5.67015921, -5.68214822])
    
    Get equivalent sound pressure level (Leq)
        
    >>> maad.util.add_dBSPL(L)
        54.53489077674256
    """    
    # force to be ndarray
    P= np.asarray(P)
    
    # if p ==0 set to MIN
    P[P==0] = _MIN_
    
    # Take the log of the ratio power/pRef²
    L = 10*log10(P/pRef**2) 
    return (L)

#####  dBSPL2power
def dBSPL2power (L, pRef=20e-6): 
    """
    convert sound pressure level (L [dB])to sound pressure power (P [Pa²]) 
    
    Parameters
    ----------
    L : ndarray-like or scalar
        ndarray-like or scalar containing the sound pressure level (L [dB])
                
    pRef : Sound pressure reference in the medium (air : 20e-6 Pa, water : ?)
                
    Returns
    -------
    P : ndarray-like or scalar
        ndarray-like or scalar containing the pressure in Pa (!! amplitude not energy !!)
        

    Examples
    --------
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> L = maad.util.wav2dBSPL(wave=w, gain=42)
    >>> P = maad.util.dBSPL2power(L) 

    >>> Sxx,tn,fn,_ = maad.sound.spectrogram (w, fs, nperseg=4096, mode='amplitude')   
    >>> Lxx = maad.util.wav2dBSPL(wave=Sxx, gain=42)
    >>> Pxx = maad.util.dBSPL2power(Lxx) 
    
    Parceval's theorem of energy conservation
    
    Energy in time domain from power P
    >>> sum(P)
    Energy in frequency domain from power spectrogram Pxx
    >>> sum(Pxx/Pxx.shape[1]*len(w))
    """    
    # force to be ndarray
    L = np.asarray(L)
    
    # dB SPL to power (pressure²)
    P = 10**(L/10)*pRef**2
    return (P)

#####  wav2dBSPL
def wav2dBSPL (wave, gain, Vadc=2, sensitivity=-35, dBref=94, pRef=20e-6): 
    """
    convert wave to instantaneous sound pressure level (L [dB SPL])
    
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
        
    pRef : Sound pressure reference in the medium (air : 20e-6, water : ?)
                
    Returns
    -------
    L : ndarray-like or scalar
        ndarray-like or scalar containing the sound waveform in dB SPL (Sound Pressure level in dB)
        

    Examples
    --------
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> L = maad.util.wav2dBSPL(wave=w, gain=42)
    
    Get equivalent sound pressure level (Leq)
        
    >>> maad.util.mean_dBSPL(L)
        54.53489077674256

    """       
    # force to be ndarray
    wave = np.asarray(wave)
    
    p = wav2pressure(wave, gain, Vadc, sensitivity, dBref)
    p_abs = abs(p)
    L = pressure2dBSPL(p_abs, pRef)
    return (L)

##### add_dBSPL
def add_dBSPL(*argv, axis=1): 
    """
    add Sound Pressure Level (L [dB  SPL])
        
    Parameters
    ----------
    *argv : ndarray-like of floats
        Arrays containing the sound waveform in dB SPL (Sound Pressure level in dB)
                
    axis : integer, optional, default is 0
        if addition of multiple arrays, select the axis on which the sum is done
                
    Returns
    -------
    L_sum : ndarray-like of floats
        Array containing the sum of the Sound Pressure Level L in [dB SPL]
        

    Examples
    --------
    
    Example with an audio file
    
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> p = maad.util.wav2pressure(w, gain=42)
    >>> PSDxx,tn,fn,_ = maad.sound.spectrogram(p,fs)
    >>> L = maad.util.energy2dBSPL(PSDxx)
    >>> L_sum = maad.util.add_dBSPL(L, axis=2)  
    >>> fig_kwargs = {'figtitle':'Spectrum (PSD)',
                      'xlabel':'Frequency [Hz]',
                      'ylabel':'Power [dB]',
                      }
    >>> fig, ax = maad.util.plot1D(fn, L_sum.transpose(), **fig_kwargs)

    Example with single values
    
    >>> L1 = 90 # 90dB
    >>> maad.util.add_dBSPL(L1,L1)
        93.01029995663981
        
    Example with arrays
    
    >>> L1 = [90,80,70]
    >>> maad.util.add_dBSPL(L1,L1)
        array([90.45322979, 90.45322979])
    >>> maad.util.add_dBSPL(L1,L1, axis=0)
        array([93.01029996, 83.01029996, 73.01029996])
    """    
    # force to be ndarray
    L = np.asarray(argv)
    
    # Verify the adequation between axis number and number of dimensions of L
    if axis >= L.ndim:
        axis= L.ndim -1
    
    # dB SPL to energy as sum has to be done with energy
    e = dBSPL2pressure(L)**2   
    e_sum = e.sum(axis)
    
    # energy=>pressure to dB SPL
    L_sum = pressure2dBSPL(sqrt(e_sum))
    
    return (L_sum)

##### mean_dBSPL
def mean_dBSPL(*argv, axis=1): 
    """
    Compute the average of Sound Pressure Level (L [dB  SPL])
        
    Parameters
    ----------
    *argv : ndarray-like of floats
        Arrays containing the sound waveform in dB SPL (Sound Pressure level in dB)
                
    axis : integer, optional, default is 0
        if addition of multiple arrays, select the axis on which the sum is done
                
    Returns
    -------
    L_mean : ndarray-like of floats
        Array containing the mean of the Sound Pressure Level L in [dB SPL]
        

    Examples
    --------
    
    Example with an audio file
    
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> p = maad.util.wav2pressure(w, gain=42)
    >>> Pxx,tn,fn,_ = maad.sound.spectrogram(p,fs)
    >>> L = maad.util.power2dBSPL(Pxx)
    >>> L_mean = maad.util.mean_dBSPL(L, axis=2)  
    >>> fig_kwargs = {'figtitle':'Power spectrum (PSD)',
                      'xlabel':'Frequency [Hz]',
                      'ylabel':'Power [dB]',
                      }
    >>> fig, ax = maad.util.plot1D(fn, L_mean.transpose(), **fig_kwargs)

    Example with single values
    
    >>> L1 = 90 # 90dB
    >>> maad.util.mean_dBSPL(L1,L1)
        93.01029995663981
        
    Example with arrays
    
    >>> L1 = [90,80,70]
    >>> maad.util.mean_dBSPL(L1,L1)
        array([85.68201724, 85.68201724])
    >>> maad.util.mean_dBSPL(L1,L1, axis=0)  
        array([90., 80., 70.]) 
        
    """    
    # force to be ndarray
    L = np.asarray(argv)
    
    # Verify the adequation between axis number and number of dimensions of L
    if axis >= L.ndim:
        axis= L.ndim -1
    
    # dB SPL to energy as sum has to be done with energy
    e = dBSPL2pressure(L)**2   
    e_sum = e.mean(axis)
    
    # energy=>pressure to dB SPL
    L_mean = pressure2dBSPL(sqrt(e_sum))
    
    return (L_mean)


################################## Leq ########################################
# wav2Leq
def wav2Leq (wave, f, gain, Vadc=2, dt=1, sensitivity=-35, dBref = 94): 
    """
    convert wave to Equivalent Continuous Sound Pressure level (Leq [dB SPL])
    
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
                
    Returns
    -------
    Leq : float
        Equivalent Continuous Sound pressure level (Leq [dB SPL])
       
 
    Examples
    --------
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> Leq = maad.util.wav2Leq (w, fs, gain=42)  
    >>> maad.util.mean_dBSPL(Leq)
        array([54.57548258])
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
    Leq = 20*log10(volt_RMS) - sensitivity + dBref - gain
    return(Leq)

#####  pressure2Leq
def pressure2Leq (p, f, dt=1, pRef = 20e-6): 
    """
    convert pressure (p [Pa]) to Equivalent Continuous Sound Pressure level (Leq [dB SPL])
    
    Parameters
    ----------
    p : ndarray-like
        Array containing the sound pressure in Pa 
        
    f : integer
        Sampling frequency in Hz
        
    dt : float, optional, default is 1 (second)
        Integration step to compute the Leq (Equivalent Continuous Sound level)

    pRef : Sound pressure reference in the medium (air : 20e-6, water : ?)
           
    Returns
    -------
    Leq : float
        Equivalent Continuous Sound pressure level (Leq [dB SPL])
        
  
    Examples
    --------
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> p = maad.util.wav2pressure(wave=w, gain=42)
    >>> Leq = maad.util.pressure2Leq (p, f=fs)
    >>> maad.util.mean_dBSPL(Leq)
        array([54.57548258])
    """     
    # be sure they are ndarray
    p = np.asarray(p)
    
    ##### wav SPLto Leq 
    # wav to RMS
    dN = int(np.floor(dt*f)) # RMS period in number of points
    N_RMS = int(np.floor(len(p)/dN)) # number of RMS periods
    # reduction of volt vector length
    p_1D = p[0:dN*N_RMS]
    # reshape into 2D (the duration of each row is dt)
    p_2D = p_1D.reshape(N_RMS,dN)
    # RMS
    p_RMS = sqrt(mean(p_2D**2,axis=1))# Test the current operating system
    # if p_RMS ==0 set to MIN
    p_RMS[p_RMS==0] = _MIN_
    # RMS to Leq (Equivalent Continuous Sound level)
    Leq = 20*log10(p_RMS/pRef)
    return (Leq)

#####  PSD2Leq
def PSD2Leq (P, pRef = 20e-6):
    """
    convert Power spectral density (PSD in [Pa²]) into 
    Equivalent Continuous Sound pressure level (Leq [dB SPL])
    
    Parameters
    ----------
    P : ndarray-like
        ndarray-like containing the Power spectral density (PSD) already in Pa²

    pRef : Sound pressure reference in the medium (air : 20e-6, water : ?)
           
    Returns
    -------
    Leq : float
        Equivalent Continuous Sound pressure level (Leq [dB SPL])
        
    Examples
    --------
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> p = maad.util.wav2pressure(w, gain=42)
    >>> from numpy import fft, abs
    >>> S = abs(fft.fft(p)/len(p))
    >>> maad.util.PSD2Leq(P=S**2)
        54.53489077674256
    """  
    # be sure they are ndarray
    P = np.asarray(P)
    
    # if P ==0 set to MIN
    P[P==0] = _MIN_
    
    # Energy (pressure^2) to Leq (=> Leq (Equivalent Continuous Sound level) 
    # if the sum is performed on the whole PSD)
    Leq = 10*log10(sum(P)/pRef**2)
    return(Leq)
   
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
    filename = datadir/"jura_cold_forest_jour.wav"
    
    # Go to the root directory where is the module maad and import maad
    maad_path = Path(dir_path).parents[1]
    os.sys.path.append(maad_path.as_posix())
    import maad
    
    ####### Variables
    S = -35     #  Sensbility microphone (for SM4 it is -35dBV)
    G = 42      # total amplification gain : preamplifier gain = 26dB and gain = 16dB
    P_REF = 20e-6   # Reference pressure (in the air : 20µPa)
    deltaT = 1  # Leq integration time dt in s
    NFFT = 1024  # length of the fft
    VADC = 2    # Maximal voltage (peak to peak) converted by the analog to digital convertor ADC (i.e : +/- 1V peak to peak => Vadc=2V)
    
    # Load the sound
    wav,fs = maad.sound.load(filename=filename, channel='right', detrend=False, verbose=False)
    
    print('-------------------------------------------------------------------------')
    print('Leq calculation directly from the sound file (in time or frequency domain')
    
    # convert sounds (wave) into SPL and subtract DC offset
    wav = wav - mean(wav)
    # wav -> pressure -> Leq
    p = wav2pressure(wav, gain=G, Vadc=VADC, sensitivity=S, dBref=94)
    Leq = pressure2Leq(p, f=fs, dt=deltaT)
    print('Leq from volt', maad.util.linear2dB(mean(maad.util.dB2linear(Leq, mode="power")),mode="power"))
    
    # wav -> Leq
    wav_Leq2 = wav2Leq(wav, f=fs, gain=G, Vadc=VADC, dt=deltaT, sensitivity=S, dBref = 94) 
    print('Leq from wav', maad.util.linear2dB(mean(maad.util.dB2linear(wav_Leq2, mode="power")),mode="power"))
        
    # Power Density Spectrum : PSD
    # with p
    from numpy import fft
    P = abs(fft.fft(p)/len(p))**2
    # average Leq from PSD
    P_Leq3  = PSD2Leq(P)
    print('Leq from PSD',P_Leq3)
       
    ########################################################################
    # Leq from spectrogram PSDxx : 
    print('-------------------------------------------------------------------------')
    print('Leq calculation from Spectrogram (time-frequency representation of a sound ')
    
    # Power Density Spectrogram : PSDxx
    # with p
    Pxx,tn,fn,_ = maad.sound.spectrogram (p, fs=fs, nperseg=NFFT, mode='psd')  
    # Pxx to P
    P = mean(Pxx,axis=1)
    # average Leq from P
    P_Leq5  = PSD2Leq(P)
    print('Leq from Pxx spectrogram',P_Leq5)

    # total energy from PSD_mean
    energy  =sum(P)
    P_Leq6  = pressure2dBSPL(sqrt(energy))
    print('Leq from Pxx energy',P_Leq6)
    
    print('')
    print('> The difference with previous Leq calculation is due to the average of the PSDxx along the time axis which reduces the noise contribution into the total energy.')
    print('> By increasing the NFFT length (i.e. NFFT=4096), Leq value converges towards previous Leq values') 
 


    
    
