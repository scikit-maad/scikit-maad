#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" S
Segmentation methods for 1D signals.
This module gathers a collection of functions to detect regions of interest (ROIs)
in the temporal domain.
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
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

# import internal modules
from maad.sound import sinc

#%%
# =============================================================================
# Private functions
# =============================================================================
def _corresp_onset_offset(onset, offset, tmin, tmax):
    """ 
    Check that each onsets have a corresponding offset.

    Parameters
    ----------
    onset: ndarray
        array with onset from find_rois_1d
    offset: ndarray
        array with offset from find_rois_1d
    tmin: float
        Start time of wav file  (in s)
    tmax:
        End time of wav file  (in s)
    
    Return
    ------
    onset : ndarray
        onset with corresponding offset
    offset : ndarray
        offset with corresponding onset
    """
    if onset[0] > offset[0]:      # check start
        onset = np.insert(onset,0,tmin)
    else:
        pass
    if onset[-1] > offset[-1]:      # check end
        offset = np.append(offset,tmax)
    else:
        pass
    return onset, offset

#%%
def _energy_windowed(s, wl=512, fs=None):
    """ 
    Computse windowed energy on an audio signal.
    
    Computes the energy of the signals by windows of length wl. Used to amplify sectors where the density of energy is higher
    
    Parameters
    ----------
    s : ndarray
        input signal
    wl : float
        length of the window to summarize the rms value
    fs : float
        frequency sampling of the signal, used to keep track of temporal information of the signal

    Returns
    -------
    time : ndarray
        temporal index vector
    s_rms : ndarray
        windowed rms signal
    """
    
    s_aux = np.lib.pad(s, (0, wl-len(s)%wl), 'reflect')  # padding
    s_aux = s_aux**2 
    #  s_aux = np.abs(s_aux) # absolute value. alternative option
    s_aux = np.reshape(s_aux,(int(len(s_aux)/wl),wl))
    s_rms = np.mean(s_aux,1)
    time = np.arange(0,len(s_rms)) * wl / fs + wl*0.5/fs
    return time, s_rms

#%%
# =============================================================================
# Public functions
# =============================================================================
def find_rois_cwt(s, fs, flims, tlen, th=0, display=False, save_df=False, 
                  savefilename='rois.csv', **kwargs):
    """
    Find region of interest using known estimates of signal length and frequency limits.
    
    The general approach is based on continous wavelet transform following a three step process
    
    1. Filter the signal with a bandpass sinc filter
    
    2. Smoothing the signal by convolving it with a Mexican hat wavelet (Ricker wavelet) [1]
    
    3. Binarize the signal applying a linear threshold
        
    Parameters
    ----------
    s : ndarray
        input signal
    flims : int
        upper and lower frequencies (in Hz) 
    tlen : int 
        temporal length of signal searched (in s)
    th : float, optional
        threshold to binarize the output
    display: boolean, optional, default is False
        plot results if set to True, default is False
    save_df : boolean, optional
        save results to csv file
    savefilename : str, optional
        Name of the file to save the table as comma separatd values (csv)        
    
    Returns
    -------
    rois : pandas DataFrame
        an table with temporal and frequencial limits of regions of interest            
    
    References
    ----------
    .. [1] Bioinformatics (2006) 22 (17): 2059-2065. DOI:10.1093/bioinformatics/btl355 http://bioinformatics.oxfordjournals.org/content/22/17/2059.long
    
    Examples
    --------
    >>> from maad import sound, rois
    >>> s, fs = sound.load('./data/spinetail.wav')
    >>> rois.find_rois_cwt(s, fs, flims=(4500,8000), tlen=2, th=0, display=True)
        min_f     min_t   max_f     max_t
    0  4500.0   0.74304  8000.0   2.50776
    1  4500.0   5.10839  8000.0   7.33751
    2  4500.0  11.23846  8000.0  13.37469
    3  4500.0  16.16109  8000.0  18.29732
    """
    # filter signal
    s_filt = sinc(s, flims, fs, atten=80, transition_bw=0.8)
    # rms: calculate window of maximum 5% of tlen. improves speed of cwt
    wl = 2**np.floor(np.log2(tlen*fs*0.05)) 
    t, s_rms = _energy_windowed(s_filt, int(wl), fs)
    # find peaks
    cwt_width = [round(tlen*fs/wl/2)]
    npad = 5 ## seems to work with 3, but not sure
    s_rms = np.pad(s_rms, np.int(cwt_width[0]*npad), 'reflect')  ## add pad
    s_cwt = signal.cwt(s_rms, signal.ricker, cwt_width)
    s_cwt = s_cwt[0][np.int(cwt_width[0]*npad):len(s_cwt[0])-np.int(cwt_width[0]*npad)] ## rm pad
    # find onset and offset of sound
    segments_bin = np.array(s_cwt > th)
    onset = t[np.where(np.diff(segments_bin.astype(int)) > 0)]+t[0]  # there is delay because of the diff that needs to  be accounted
    offset = t[np.where(np.diff(segments_bin.astype(int)) < 0)]+t[0]
    # format for output
    if onset.size==0 or offset.size==0:
    # No detection found
        print('Warning: No detection found')
        df = pd.DataFrame(data=None)
        if save_df==True:
            df.to_csv(savefilename, sep=',',header=False, index=False)

    else:
    # A detection was found, save results to csv
        onset, offset = _corresp_onset_offset(onset, offset, tmin=0, tmax=len(s)/fs)
        rois_tf = np.transpose([np.repeat(flims[0],repeats=len(onset)),
                                np.round(onset,5),  
                                np.repeat(flims[1],repeats=len(onset)),
                                np.round(offset,5)])
        cols=['min_f', 'min_t','max_f', 'max_t']
        df = pd.DataFrame(data=rois_tf,columns=cols)
        if save_df==True:
            df.to_csv(savefilename, sep=',', header=True, index=False)

    # Display
    if display==True: 
        figsize = kwargs.pop('figsize',(12,6))
        cmap = kwargs.pop('cmap','gray')
        nfft = kwargs.pop('nfft',512)
        noverlap = kwargs.pop('noverlap',256)
        # plot
        fig,(ax1,ax2) = plt.subplots(2,1,figsize=figsize)
        ax1.margins(x=0)
        ax1.plot(s_cwt)
        ax1.set_xticks([])
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)
        ax1.hlines(th, 0, len(s_cwt), linestyles='dashed', colors='r')
        ax2.specgram(s, NFFT=nfft, Fs=fs, noverlap=noverlap, cmap=cmap)
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_xlabel('Time (s)')
        if not(df.empty):
            for idx, _ in df.iterrows():
                xy = (df.min_t[idx],df.min_f[idx])
                width = df.max_t[idx] - df.min_t[idx]
                height = df.max_f[idx] - df.min_f[idx]
                rect = patches.Rectangle(xy, width, height, lw=1, 
                                         edgecolor='r', facecolor='none')
                ax2.add_patch(rect)
        plt.show()
    return df
