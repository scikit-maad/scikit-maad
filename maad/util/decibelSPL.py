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

"""****************************************************************************
# -------------------       Functions               ---------------------------
****************************************************************************"""

##### wav2volt
def wav2volt (wave, Vadc=2): 
    """
    convert in Volt
    
    Parameters
    ----------
    wave : 1d ndarray of floats 
        wave should already be normalized between -1 to 1 (depending on the number of bits)
        take the output of the function sound.load of maad module
        Vector containing the raw sound waveform 
        
    Vadc : scalar, optional, default is 2Vpp (=>+/-1V)
        Maximal voltage (peak to peak) converted by the analog to digital convertor ADC  
                
    Returns
    -------
    volt : 1d ndarray of floats
        Vector containing the sound waveform in volt
            
    Examples
    --------
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> v = maad.util.wav2volt(wave=w, Vadc=2)
    """
    # be sure they are ndarray
    wave = np.asarray(wave)
    
    volt =wave * Vadc
    return volt

##### volt2SPL
def volt2SPL(volt, gain, sensitivity=-35, dBref=94):  
    """
    convert volt amplitude to instantaneous sound pressure level SPL
    
    Parameters
    ----------
    volt : 1d ndarray of integers floats
        Vector containing the sound waveform in volt
        
    gain : integer
        Total gain applied to the sound (preamplifer + amplifier)
    
    sensitivity : float, optional, default is -35 (dB/V)
        Sensitivity of the microphone
    
    dBref : integer, optional, default is 94 (dBSPL)
        Pressure sound level used for the calibration of the microphone 
        (usually 94dB, sometimes 114dB)
                
    Returns
    -------
    wave_SPL : 1d ndarray of floats
        Vector containing the sound waveform in SPL (Sound Pressure level : Pa)
        
    Examples
    --------
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> v = maad.util.wav2volt(wave=w)
    >>> s = maad.util.volt2SPL(volt=v, gain=42)        
    """
    # be sure they are ndarray
    volt = np.asarray(volt)
    
    # wav to instantaneous sound pressure level (SPL)
    # coefficient to convert Volt into pressure (Pa)
    Volt2Pa = 1/10**(sensitivity/20) 
    wave_SPL = volt * Volt2Pa / 10**(gain/20)
    return (wave_SPL)

#####  wav2SPL 
def wav2SPL (wave, gain, Vadc = 2, sensitivity=-35, dBref=94): 
    """
    convert wave to instantaneous sound pressure level SPL
    
    Parameters
    ----------
    wave : 1d ndarray of integers (directly from the wav file)
        Vector containing the raw sound waveform 
        
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
    wave_SPL : 1d ndarray of floats
        Vector containing the sound waveform in SPL (Sound Pressure level : Pa)
        
    Examples
    --------
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> s = maad.util.wav2SPL(wave=w, gain=42)
    """
    wave_SPL = volt2SPL(wav2volt(wave, Vadc), sensitivity, gain, dBref) 
    return (wave_SPL)

#####  SPL2dBSPL
def SPL2dBSPL (waveSPL, pRef=20e-6): 
    """
    convert instantaneous sound pressure level SPL to 
    instantaneous sound pressure level SPL in dB
    
    Parameters
    ----------
    wave : 1d ndarray of integers or scalar
        Vector or scalar containing the SPL value (!! amplitude not energy !!)
                
    pRef : Sound pressure reference in the medium (air : 20e-6, water : ?)
                
    Returns
    -------
    wave_dBSPL : 1d ndarray of floats
        Vector containing the sound waveform in dB SPL (Sound Pressure level in dB)
        
    Examples
    --------
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> s = maad.util.wav2SPL(wave=w, gain=42)
    >>> s_dB = maad.util.SPL2dBSPL(waveSPL=s)        
    """    
    wave_dBSPL = energy2dBSPL(waveSPL**2, pRef)
    return (wave_dBSPL)

#####  wav2dBSPL
def wav2dBSPL (wave, gain, Vadc=2, sensitivity=-35, dBref=94, pRef=20e-6): 
    """
    convert wave to instantaneous sound pressure level SPL in dB
    
    Parameters
    ----------
    wave : 1d ndarray of integers (directly from the wav file)
        Vector containing the raw sound waveform 
        
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
    wave_dBSPL : 1d ndarray of floats
        Vector containing the sound waveform in dB SPL (Sound Pressure level in dB)

    Examples
    --------
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> s_dB = maad.util.wav2dBSPL (wave=w, gain=42)   
    """        
    wave_SPL   = wav2SPL(wave, Vadc, sensitivity, gain, dBref) 
    wave_dBSPL = energy2dBSPL(wave_SPL**2, pRef)
    return (wave_dBSPL)

# wav2Leq
def wav2Leq (wave, f, gain, Vadc=2, dt=1, sensitivity=-35, dBref = 94): 
    """
    convert wave to Equivalent Continuous Sound level (Leq)
    
    Parameters
    ----------
    wave : 1d ndarray of integers (directly from the wav file)
        Vector containing the raw sound waveform 
        
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
    sig_Leq : float
        Equivalent Continuous Sound level (Leq)
        
    Examples
    --------
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> leq_dB = maad.util.wav2Leq (wave=w, f=fs, gain=42)  
    """    
     
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
    # RMS to Leq (Equivalent Continuous Sound level)
    sig_Leq = 20*log10(volt_RMS) - sensitivity + dBref - gain
    return(sig_Leq)

#####  wavSPL2Leq
def wavSPL2Leq (wave_SPL, f, dt=1, pRef = 20e-6): 
    """
    convert wave SPL to Equivalent Continuous Sound level (Leq)
    
    Parameters
    ----------
    wave : 1d ndarray of floats
        Vector containing the sound waveform in SPL (Sound Pressure level : Pa) 
        
    f : integer
        Sampling frequency in Hz
        
    dt : float, optional, default is 1 (second)
        Integration step to compute the Leq (Equivalent Continuous Sound level)

    pRef : Sound pressure reference in the medium (air : 20e-6, water : ?)
           
    Returns
    -------
    sig_Leq : float
        Equivalent Continuous Sound level (Leq)
        
    Examples
    --------
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> s = maad.util.wav2SPL(wave=w, gain=42)
    >>> leq_dB = maad.util.wavSPL2Leq (wave_SPL=s, f=fs)
    """     
    # be sure they are ndarray
    wave_SPL = np.asarray(wave_SPL)
    
    ##### wav SPLto Leq 
    # wav to RMS
    dN = int(np.floor(dt*f)) # RMS period in number of points
    N_RMS = int(np.floor(len(wave_SPL)/dN)) # number of RMS periods
    # reduction of volt vector length
    wave_SPL_1D = wave_SPL[0:dN*N_RMS]
    # reshape into 2D (the duration of each row is dt)
    wave_SPL_2D = wave_SPL_1D.reshape(N_RMS,dN)
    # RMS
    wave_SPL_RMS = sqrt(mean(wave_SPL_2D**2,axis=1))# Test the current operating system
    # RMS to Leq (Equivalent Continuous Sound level)
    sig_Leq = 20*log10(wave_SPL_RMS/pRef)
    return (sig_Leq)

#####  energy2dBSPL
def energy2dBSPL(energy_SPL, pRef = 20e-6): 
    """
    convert energy signal (e.g. Power spectral density (PSD) or wav²) 
    already in SPL² into dB SPL
    
    Parameters
    ----------
    energy_SPL : 1d ndarray of floats
        Vector containing the energy signal in SPL²

    pRef : Sound pressure reference in the medium (air : 20e-6, water : ?)
           
    Returns
    -------
    energy_dBSPL : 1d ndarray of floats
         Vector containing the energy signal in dB SPL
         
    Examples
    --------
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    
    Energy in time domain (s²)
    
    >>> s = maad.util.wav2SPL(wave=w, gain=42)
    >>> s_dB = maad.util.energy2dBSPL(energy_SPL=s^2)
    
    Energy in frequency domain (S²)

    >>> from numpy import fft, abs
    >>> S = abs(fft.fft(s)/len(s))
    >>> S_dB = maad.util.energy2dBSPL(energy_SPL=S^2)
    
    """      
    # be sure they are ndarray
    energy_SPL = np.asarray(energy_SPL)
        
    # Take the log of the ratio ENERGY/pRef
    energy_dBSPL = 10*log10(energy_SPL/pRef**2)
    
    return (energy_dBSPL)


#####  dBSPL2energy
def dBSPL2energy (e_dB, pRef = 20e-6):
    """
    convert energy in dB SPL (e.g. PSD) into energy in SPL²
    
    Parameters
    ----------
    e_dB : 1d ndarray of floats
         Vector containing the energy in dB SPL

    pRef : Sound pressure reference in the medium (air : 20e-6, water : ?)
           
    Returns
    -------
    e : 1d ndarray of floats
        Vector containing the energy in SPL²
        
    Examples
    --------
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    
    Energy in time domain (s²)
    
    >>> s = maad.util.wav2SPL(wave=w, gain=42)
    >>> s_dB = maad.util.energy2dBSPL(energy_SPL=s^2)
    >>> e = maad.util.dBSPL2energy(energy_dBSPL=s_dB)
    
    Energy in frequency domain (S²)

    >>> from numpy import fft, abs
    >>> S = abs(fft.fft(s)/len(s))
    >>> S_dB = maad.util.energy2dBSPL(energy_SPL=S^2)
    >>> E = maad.util.dBSPL2energy(energy_dBSPL=S_dB)
    """  
    # be sure they are ndarray
    e_dB = np.asarray(e_dB)
    
    # SPL to energy (pressure^2) (=> Leq (Equivalent Continuous Sound level)
    e = 10**(e_dB/10)*pRef**2
    return(e)

def PSD2Leq (PSD, pRef = 20e-6):
    """
    convert Power spectral density (PSD) in SPL² into 
    Equivalent Continuous Sound level (Leq)
    
    Parameters
    ----------
    PSD : 1d ndarray of floats
        Vector containing the Power spectral density (PSD) already in SPL²

    pRef : Sound pressure reference in the medium (air : 20e-6, water : ?)
           
    Returns
    -------
    Leq : float
        Equivalent Continuous Sound level (Leq)
        
    Examples
    --------
    >>> w, fs = maad.sound.load('jura_cold_forest_jour.wav') 
    >>> s = maad.util.wav2SPL(wave=w, gain=42)
    >>> from numpy import fft, abs
    >>> S = abs(fft.fft(s)/len(s))
    >>> Leq = maad.util.PSD2Leq(PSD=S^2)
    """  
    # be sure they are ndarray
    PSD = np.asarray(PSD)
    
    # Energy (pressure^2) to Leq (=> Leq (Equivalent Continuous Sound level) 
    # if the sum is performed on the whole PSD)
    Leq = 10*log10(sum(PSD)/pRef**2)
    return(Leq)
   
########### TEST
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
    NFFT = 512  # length of the fft
    Vadc = 2    # Maximal voltage (peak to peak) converted by the analog to digital convertor ADC (i.e : +/- 1V peak to peak => Vadc=2V)
    
    # Load the sound
    volt,fs = maad.sound.load(filename=filename, channel='right', detrend=False, verbose=False)
    
    print('-------------------------------------------------------------------------')
    print('Leq calculation directly from the sound file (in time or frequency domain')
    
    # convert sounds (wave) into SPL and subtract DC offset
    volt = volt - mean(volt)
    # wav -> SPL -> Leq
    wav_SPL = volt2SPL(volt, sensitivity=S, gain=G, dBref=94)
    wav_Leq = wavSPL2Leq(wav_SPL, f=fs, dt=deltaT)
    print('Leq from volt', maad.util.linear2dB(mean(maad.util.dB2linear(wav_Leq, mode="power")),mode="power"))
    
    # wav -> Leq
    wav = volt/Vadc
    wav_Leq2 = wav2Leq(wav, f=fs, dt=deltaT, sensitivity=S, gain=G, dBref = 94) 
    print('Leq from wav', maad.util.linear2dB(mean(maad.util.dB2linear(wav_Leq2, mode="power")),mode="power"))
    
    # Power Density Spectrum : PSD
    from numpy import fft
    PSD = abs(fft.fft(wav_SPL)/len(wav_SPL))**2
    # average Leq from PSD
    PSD_Leq3  = PSD2Leq(PSD)
    print('Leq from PSD',PSD_Leq3 )
    
    ########################################################################
    # Leq from spectrogram PSDxx : 
    print('-------------------------------------------------------------------------')
    print('Leq calculation from Spectrogram (time-frequency representation of a sound ')
    
    # Power Density Spectrogram : PSDxx
    PSDxx,tn,fn,_ = maad.sound.spectrogram (wav_SPL, fs=fs, nperseg=NFFT, 
                                            mode='psd')  
    PSD_mean = mean(PSDxx,axis=1)
    # average Leq from PSD
    PSD_Leq4  = PSD2Leq(PSD_mean)
    print('Leq from PSDxx spectrogram',PSD_Leq4 )

    # energy from PSD_mean
    energy  =sum(PSD_mean)
    PSD_Leq5  = energy2dBSPL(energy)
    print('Leq from energy',PSD_Leq5 )
    
    print('')
    print('> The difference with previous Leq calculation is due to the average of the PSDxx along the time axis which reduces the noise contribution into the total energy.')
    print('> By increasing the NFFT length (i.e. NFFT=4096), Leq value converges towards previous Leq values') 

 

