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
from scipy.signal import hilbert

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
    
    compute the time vector for the vector wave
    
    >>> import numpy as np
    >>> t = np.arange(0,len(s),1)/fs
    
    compute the time vector for the vector env_fast
    >>> t_env_fast = np.arange(0,len(env_fast),1)*len(s)/fs/len(env_fast)
    
    plot 0.1s of the envelope and 0.1s of the abs(s)
    
    >>> import matplotlib.pyplot as plt
    >>> fig1, ax1 = plt.subplots()
    >>> ax1.plot(t[t<0.1], abs(s[t<0.1]), label='abs(s)')
    >>> ax1.plot(t[t<0.1], env_hilbert[t<0.1], label='env(s) - hilbert option')
    >>> ax1.plot(t_env_fast[t_env_fast<0.1], env_fast[t_env_fast<0.1], label='env(s) - fast option')
    >>> ax1.set_xlabel('Time [sec]')
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
