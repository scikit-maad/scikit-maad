#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Created on Mon Jun 24 10:28:51 2019
    Convert raw wav data into calibrated dB SPL"""
#
# Authors:  Sylvain HAUPERT <sylvain.haupert@mnhn.fr>             
#
# License: New BSD License


"""****************************************************************************
# -------------------       Load modules            ---------------------------
****************************************************************************"""

# Import external modules

import numpy as np 
from numpy import sum, log, log10, min, max, abs, mean, median, sqrt, diff

"""****************************************************************************
# -------------------       Functions               ---------------------------
****************************************************************************"""

##### wav2volt
def wav2volt (wave, bit=16, Vadc = sqrt(2)): 
    """
    convert in Volt
    
    Parameters
    ----------
    wave : 1d ndarray of integers (directly from the wav file)
        Vector containing the raw sound waveform 

    bit : int, optional, default is 16
        Number of bits of the analog to digital converter (ADC)
        
    Vadc : scalar, optional, default is 1Vrms = 1*sqrt(2)
        Maximal voltage converted by the analog to digital convertor ADC 
                
    Returns
    -------
    volt : 1d ndarray of floats
        Vector containing the sound waveform in volt
    """
    volt = (wave/(2**(bit-1))) * Vadc
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
    """
    # wav to instantaneous sound pressure level (SPL)
    # coefficient to convert Volt into pressure (Pa)
    Volt2Pa = 1/10**(sensitivity/20) 
    wave_SPL = volt * Volt2Pa / 10**(gain/20)
    return (wave_SPL)

#####  wav2SPL 
def wav2SPL (wave, gain, bit=16, Vadc = sqrt(2), sensitivity=-35, dBref=94): 
    """
    convert wave to instantaneous sound pressure level SPL
    
    Parameters
    ----------
    wave : 1d ndarray of integers (directly from the wav file)
        Vector containing the raw sound waveform 
        
    gain : integer
        Total gain applied to the sound (preamplifer + amplifier)
        
    bit : int, optional, default is 16
        Number of bits of the analog to digital converter (ADC)  
        
    Vadc : scalar, optional, default is 1Vrms = 1*sqrt(2)
        Maximal voltage converted by the analog to digital convertor ADC       

    sensitivity : float, optional, default is -35 (dB/V)
        Sensitivity of the microphone
    
    dBref : integer, optional, default is 94 (dBSPL)
        Pressure sound level used for the calibration of the microphone 
        (usually 94dB, sometimes 114dB)
                
    ReturnsdBSP
    -------
    wave_SPL : 1d ndarray of floats
        Vector containing the sound waveform in SPL (Sound Pressure level : Pa)
    """
    wave_SPL = volt2SPL(wav2volt(wave, bit, Vadc), sensitivity, gain, dBref) 
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
    """
    # if waveSPL is a list, convert waveSPL into ndarray 
    if isinstance(waveSPL, list):
        waveSPL = np.asarray(waveSPL)
    
    wave_dBSPL = energy2dBSPL(waveSPL**2, pRef)
    return (wave_dBSPL)

#####  wav2dBSPL
def wav2dBSPL (wave, gain, Vadc = sqrt(2), bit=16, sensitivity=-35, dBref=94, pRef=20e-6): 
    """
    convert wave to instantaneous sound pressure level SPL in dB
    
    Parameters
    ----------
    wave : 1d ndarray of integers (directly from the wav file)
        Vector containing the raw sound waveform 
        
    gain : integer
        Total gain applied to the sound (preamplifer + amplifier)
        
    bit : int, optional, default is 16
        Number of bits of the analog to digital converter (ADC)  
        
    Vadc : scalar, optional, default is 1Vrms = 1*sqrt(2)
        Maximal voltage converted by the analog to digital convertor ADC    

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
    """
    # if wave is a list, convert wave into ndarray 
    if isinstance(wave, list):
        wave = np.asarray(wave)
        
    wave_SPL   = wav2SPL(wave, bit, Vadc, sensitivity, gain, dBref) 
    wave_dBSPL = energy2dBSPL(wave_SPL**2, pRef)
    return (wave_dBSPL)



# wav2Leq
def wav2Leq (wave, f, gain, Vadc=sqrt(2), bit=16, dt=1, sensitivity=-35, dBref = 94): 
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
        
    Vadc : scalar, optional, default is 1Vrms = 1*sqrt(2)
        Maximal voltage converted by the analog to digital convertor ADC 
        
    bit : int, optional, default is 16
        Number of bits of the analog to digital converter (ADC)  
        
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
    """    
     
    # convert in Volt
    volt = wav2volt(wave, bit, Vadc)
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

    bit : int, optional, default is 16
        Number of bits of the analog to digital converter (ADC)  
        
    dt : float, optional, default is 1 (second)
        Integration step to compute the Leq (Equivalent Continuous Sound level)

    pRef : Sound pressure reference in the medium (air : 20e-6, water : ?)
           
    Returns
    -------
    sig_Leq : float
        Equivalent Continuous Sound level (Leq)
    """     
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
    PSD_dBSPL : 1d ndarray of floats
         Vector containing the energy signal in dB SPL
    """  
###### !!! DON'T know why this changes the value of the variable energy_SPL that is passed to the function !!!!
###### Is this a pointer ?
#    #Avoid negative value
#    #if it's a scalar
#    if hasattr(energy_SPL, "__len__") == False:
#        if (energy_SPL<=pRef**2): 
#            energy_SPL = pRef**2   
#    else : 
#        energy_SPL[energy_SPL<=pRef**2] = pRef**2 
    
    # be sure they are ndarray
    energy_SPL = np.asarray(energy_SPL)
        
    # Take the log of the ratio ENERGY/pRef
    energy_dBSPL = 10*log10(energy_SPL/pRef**2)
    
    return (energy_dBSPL)


#####  dBSPL2energy
def dBSPL2energy (energy_dBSPL, pRef = 20e-6):
    """
    convert energy in dB SPL (e.g. PSD) into energy in SPL²
    
    Parameters
    ----------
    PSD_dBSPL : 1d ndarray of floats
         Vector containing the energy in dB SPL

    pRef : Sound pressure reference in the medium (air : 20e-6, water : ?)
           
    Returns
    -------
    PSD_SPL : 1d ndarray of floats
        Vector containing the energy in SPL²
    """  
    # be sure they are ndarray
    energy_dBSPL = np.asarray(energy_dBSPL)
    
    # SPL to energy (pressure^2) (=> Leq (Equivalent Continuous Sound level)
    energy_SPL = 10**(energy_dBSPL/10)*pRef**2
    return(energy_SPL)

def PSD2Leq (PSD_SPL, pRef = 20e-6):
    """
    convert Power spectral density (PSD) in SPL² into 
    Equivalent Continuous Sound level (Leq)
    
    Parameters
    ----------
    PSD_SPL : 1d ndarray of floats
        Vector containing the Power spectral density (PSD) already in SPL²

    pRef : Sound pressure reference in the medium (air : 20e-6, water : ?)
           
    Returns
    -------
    PSD_Leq : float
        Equivalent Continuous Sound level (Leq)
    """  
###### !!! DON'T know why this changes the value of the variable energy_SPL that is passed to the function !!!!
###### Is this a pointer ?
#    # Avoid negative value
#    # if it's a scalar
#    if hasattr(PSD_SPL, "__len__") == False:
#        if PSD_SPL <=pRef**2 : 
#            PSD_SPL = pRef**2   
#    else :     
#        PSD_SPL[PSD_SPL<=pRef**2] = pRef**2 
    
    # Energy (pressure^2) to Leq (=> Leq (Equivalent Continuous Sound level) 
    # if the sum is performed on the whole PSD)
    PSD_Leq = 10*log10(sum(PSD_SPL)/pRef**2)
    return(PSD_Leq)
   
########### TEST
############### Import OWN modules
## Test the current operating system
#import os
#import scipy as sp
#from pathlib import Path # in order to be wind/linux/MacOS compatible
#if os.name =='nt':
#    # Path
#    root_module_path = Path("D:\\")
#elif os.name =='posix':
#    # LINUX or MACOS
#    root_module_path = Path("/home/haupert/DATA")
#else:
#    print ("WARNING : operating system unknow")   
#    
#maad_path = root_module_path/Path('mes_projets/_TOOLBOX/Python/maad_project/scikit-maad')
#os.sys.path.append(maad_path.as_posix())
#import maad
#
## Sensbility microphone
#S = -35   #dBV
## Amplification gain
#G = 30 #4  # total amplification gain : preamplifier gain = 26dB and gain = 4dB
## Reference pressure (in the air : 20µPa)
#P_REF = 20e-6
## Leq integration time dt in s
#deltaT = 1
##
#NFFT = 512
##
#bit = 16
#
#filename = "/home/haupert/DATA/mes_projets/02_GUYANE/dB@NOURAGUES/DATA_PROCESSING/Propagation_sonore/GUYANE/04_Propagation_100m_direction_sud_4dB_essai3/output/wn_1m.wav"
#
#volt,fs = maad.sound.load(filename=filename, channel='left', detrend=False, verbose=False)
#
## convert sounds (wave) into SPL and subtract DC offset
#volt = volt - mean(volt)
## wav -> SPL -> Leq
#wav_SPL = volt2SPL(volt, sensitivity=S, gain=G, dBref=94)
#wav_Leq = wavSPL2Leq(wav_SPL, f=fs, dt=deltaT)
#
## wav -> Leq
#wav = volt*2**15
#wav_Leq2 = wav2Leq(wav, f=fs, dt=deltaT, sensitivity=S, gain=G, dBref = 94) 
#
#
## Power Density Spetrum : PSD
#PSD,_,_,_  = maad.sound.spectrogram(wav_SPL, fs=fs, nperseg=NFFT, overlap=0, display=False)
#PSD_mean = mean(PSD,axis=1)
## average Leq from PSD
#PSD_Leq3  = PSD2Leq(PSD_mean)
#
###### OK OK OK OK
#PSD,_,_,_  = maad.sound.spectrogram2(wav_SPL, fs=fs, nperseg=NFFT, overlap=0, mode='psd', detrend=False)
#PSD_mean = mean(PSD,axis=1)
## average Leq from PSD
#PSD_Leq4  = PSD2Leq(PSD_mean)
#
##### !!!!! SOMETHING WRONG WITH THE CORRECTION OF THE AMPLITUDE WHEN USING A WINDOW OTHER THAN RECTANGULAR WIN
#amp,_,_,_  = maad.sound.spectrogram2(wav_SPL, fs=fs, nperseg=NFFT, overlap=0, mode='amplitude', detrend=False)
#PSD_mean = mean((amp**2),axis=1)
## average Leq from PSD
#PSD_Leq5  = PSD2Leq(PSD_mean)
 
