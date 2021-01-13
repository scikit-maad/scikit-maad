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
    """
    # transform wave into array
    s = np.asarray(s)
    # compute the number of frames
    K = len(s)//Nt
    # Reshape the waveform (ie vector) into a serie of frames (ie 2D matrix)
    timeframes = s[0:K*Nt].reshape(-1,Nt).transpose()
    
    return timeframes

