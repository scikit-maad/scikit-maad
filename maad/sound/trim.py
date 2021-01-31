#!/usr/bin/env python
""" 
Collection of functions to trim audio signals.
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

#%%
# =============================================================================
# public functions
# =============================================================================
def wave2frames (s, Nt=512):
    """
    Reshape a sound waveform (ie vector) into a serie of frames (ie matrix) of
    length Nt.
    
    Parameters
    ----------
    s : 1d ndarray of floats (already divided by the number of bits)
        Vector containing the sound waveform 
    Nt : int, optional, default is 512
        Number of points per frame
                
    Returns
    -------
    timeframes : 2d ndarray of floats
        Matrix containing K frames (row) with Nt points (column), K*N <= length (s)
        
    Examples
    --------
    
    >>> import numpy as np  
    
    Fast method to estimate the envelope of a waveform
    
    >>> s, fs = maad.sound.load('../data/guyana_tropical_forest.wav')
    >>> frames = maad.sound.wave2frames(s, Nt=32)
    >>> env_fast = np.max(abs(frames),0) 
    
    Comparison with the classic method with Hilbert transform 
    
    >>> env_hilbert = maad.sound.envelope(s, mode='hilbert')
    
    Compute the time vector for the vector wave.

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
    # transform wave into array
    s = np.asarray(s)
    # compute the number of frames
    K = len(s)//Nt
    # Reshape the waveform (ie vector) into a serie of frames (ie 2D matrix)
    timeframes = s[0:K*Nt].reshape(-1,Nt).transpose()
    
    return timeframes

