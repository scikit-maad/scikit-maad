#!/usr/bin/env python
""" Utilitary functions for scikit-MAAD """
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
from numpy import mean, median
import matplotlib.pyplot as plt

# min value
import sys
_MIN_ = sys.float_info.min


#=============================================================================

def running_mean(x, N, axis=0):
    """
    moving average of x over a window N
    """    
    cumsum = np.cumsum(np.insert(x, 0, 0), axis) 
    return (cumsum[N:] - cumsum[:-N]) / N

#=============================================================================
def get_unimode (X, mode ='ale',axis=1, verbose=False, display=False):
    """
    determine the statistical mode or modal value which is 
    the most common number in the dataset
    
    Parameters
    ----------
    X :  1d or 2d ndarray of scalar
        Vector or matrix 
                
    mode : str, optional, default is 'ale'
        Select the mode to remove the noise
        Possible values for mode are :
        - 'ale' : Adaptative Level Equalization algorithm [Lamel & al. 1981]
        - 'median' : subtract the median value
        - 'mean' : subtract the mean value (DC)
    
    axis : integer, default is 1
        if matrix, estimate the mode for each row (axis=0) or each column (axis=1)
            
    verbose : boolean, optional, default is False
        print messages into the consol or terminal if verbose is True
        
    display : boolean, optional, default is False
        Display the signal if True
              
    Returns
    -------
    unimode_value : float
        The most common number in the dataset
    """         
    if X.ndim ==2: 
        if axis == 0:
            X = X.transpose()
            axis = 1
    elif X.ndim ==1: 
        axis = 0
        
    if mode=='ale':
                
        if X.ndim ==2:
            unimode_value = []
            for i, x in enumerate(X):  
                # Min and Max of the envelope (without taking into account nan)
                x_min = np.nanmin(x)
                x_max = np.nanmax(x)
                # Compute a 50-bin histogram ranging between Min and Max values
                hist, bin_edges = np.histogram(x, bins=50, range=(x_min, x_max))
                                   
                if display:
                    # Plot only the first histogram
                    plt.figure()
                    plt.plot(bin_edges[0:-1],hist)
                    
                # find the maximum of the peak with quadratic interpolation
                # don't take into account the first 4 bins.
                imax = np.argmax(hist[4::]) + 4
               
                unimode_value.append(bin_edges[imax])
                
            # transpose the vector
            unimode_value = np.asarray(unimode_value)
            unimode_value = unimode_value.transpose()
        else:
            x = X
            # Min and Max of the envelope (without taking into account nan)
            x_min = np.nanmin(x)
            x_max = np.nanmax(x)
            
            # Compute a 50-bin histogram ranging between Min and Max values
            hist, bin_edges = np.histogram(x, bins=50, range=(x_min, x_max))
                        
            if display:
                #n, bins, patches = plt.hist(x=s, bins=20, color='#0504aa', alpha=0.7, rwidth=0.85)
                plt.figure()
                plt.plot(bin_edges[0:-1],hist)
                         
            # find the maximum of the peak with quadratic interpolation
            imax = np.argmax(hist)
                        
            # assuming an additive noise model : noise_bckg is the max of the histogram
            # as it is an histogram, the value is unimode_value = bin_edges_interp[np.argmax(hist_interp)]
            unimode_value = bin_edges[imax]

    elif mode=='median':
        unimode_value = median(X, axis=axis)
        
    elif mode=='mean':
        unimode_value = mean(X, axis=axis)
        
    return unimode_value



