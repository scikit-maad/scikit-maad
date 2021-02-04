#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
"""
Created on Wed Feb  3 19:50:51 2021

@author: juanfe
"""
import numpy as np
from scipy.io.wavfile import write as write_wav

#%%
def write(filename, sr, data):
    """
    Write a NumPy array as a WAV file with the Scipy method. [1]_ 

    Parameters
    ----------
    filename : string or open file handle
        Name of output wav file.
    sr : int
        Sample rate (samples/sec).
    data : ndarray
        Mono or stereo signal as NumPy array.
        
    See Also
    --------
    scipy.io.wavfile.write

    Notes
    -----
    The data-type determines the bits-per-sample and PCM/float.

    Common data types: [2]_

    =====================  ===========  ===========  =============
         WAV format            Min          Max       NumPy dtype
    =====================  ===========  ===========  =============
    32-bit floating-point  -1.0         +1.0         float32
    32-bit PCM             -2147483648  +2147483647  int32
    16-bit PCM             -32768       +32767       int16
    8-bit PCM              0            255          uint8
    =====================  ===========  ===========  =============

    References
    ----------
    .. [1] The SciPy community, "scipy.io.wavfile.write", v1.6.0.
       https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
       
    .. [2] IBM Corporation and Microsoft Corporation, "Multimedia Programming
       Interface and Data Specifications 1.0", section "Data Format of the
       Samples", August 1991
       http://www.tactilemedia.com/info/MCI_Control_Info.html

    Examples
    --------
    Write a 440Hz sine wave, sampled at 44100Hz.
    >>> import numpy as np
    >>> sr = 44100; T = 2.0
    >>> t = np.linspace(0, T, int(T*sr))
    >>> maad.sound.write('example.wav', sr, data)
    """
    if data.ndim > 1:
        data = data.T
    write_wav(filename, sr, np.asfortranarray(data))
#%%    


sr = 44100; T = 2.0
t = np.linspace(0, T, int(T*sr))
write('example.wav', sr, data)