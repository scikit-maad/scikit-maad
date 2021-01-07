#!/usr/bin/env python 
"""  
Segmentation methods to find regions of interest in the time and frequency domain.
""" 

# Import external modules 
import matplotlib.patches as mpatches 
import matplotlib.pyplot as plt 
import numpy as np  
from scipy import signal 
from scipy.ndimage import morphology 
from scipy.stats import iqr 
from skimage import measure, filters
from skimage.morphology import reconstruction
from skimage.io import imread 
import pandas as pd 
import sys 
_MIN_ = sys.float_info.min 
 
# Import internal modules 
from maad.util import (plot1D, plot2D, linear_scale, rand_cmap, running_mean, 
                       get_unimode, mean_dB, add_dB, power2dB)
 
 
#**************************************************************************** 
#************* Load an image and convert it in gray level if needed  *********** 
#**************************************************************************** 
 
def load(filename, fs, duration, flipud = True, display=False, **kwargs): 
    """ 
    Load an image from a file or an URL 
     
    Parameters 
    ----------   
    filename : string 
        Image file name, e.g. ``test.jpg`` or URL. 
     
    fs : scalar 
        Sampling frequency of the audiogram (in Hz) 
     
    duration : scalar 
        Duration of the audiogram (in s) 
     
    flipud : boolean, optional, default is True 
        Vertical flip of the matrix (image) 
     
    display : boolean, optional, default is False 
        if True, display the image 
         
    \*\*kwargs, optional. This parameter is used by plt.plot  
        - figsize : tuple of integers, optional, default: (4,10) 
            width, height in inches.   
         
        - title : string, optional, default : 'Spectrogram' 
            title of the figure 
         
        - xlabel : string, optional, default : 'Time [s]' 
            label of the horizontal axis 
         
        - ylabel : string, optional, default : 'Amplitude [AU]' 
            label of the vertical axis 
         
        - cmap : string or Colormap object, optional, default is 'gray' 
            See https://matplotlib.org/examples/color/colormaps_reference.html 
            in order to get all the  existing colormaps 
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic',  
            'viridis'... 
         
        - vmin, vmax : scalar, optional, default: None 
            `vmin` and `vmax` are used in conjunction with norm to normalize 
            luminance data.  Note if you pass a `norm` instance, your 
            settings for `vmin` and `vmax` will be ignored. 
         
        - extent : scalars (left, right, bottom, top), optional, default: None 
            The location, in data-coordinates, of the lower-left and 
            upper-right corners. If `None`, the image is positioned such that 
            the pixel centers fall on zero-based (row, column) indices. 
         
        - dpi : integer, optional, default is 96 
            Dot per inch.  
            For printed version, choose high dpi (i.e. dpi=300) => slow 
            For screen version, choose low dpi (i.e. dpi=96) => fast 
         
        - format : string, optional, default is 'png' 
            Format to save the figure  
         
        ... and more, see matplotlib  
         
    Returns 
    ------- 
    im : ndarray 
        The different color bands/channels are stored in the 
        third dimension, such that a gray-image is MxN, an 
        RGB-image MxNx3 and an RGBA-image MxNx4. 
         
    extent : list of scalars [left, right, bottom, top]
        The location, in data-coordinates, of the lower-left and 
        upper-right corners.
         
    dt : scalar 
        Time resolution of the spectrogram (horizontal x-axis) 
         
    df : scalar 
        Frequency resolution of the spectrogram (vertical y-axis) 
    """ 
     
    print(72 * '_' ) 
    print("loading %s..." %filename)  
     
    # Load image 
    im  = imread(filename, as_gray=True) 
     
    # if 3D, convert into 2D 
    if len(im.shape) == 3: 
        im = im[:,:,0] 
         
    # Rescale the image between 0 to 1 
    im = linear_scale(im, minval= 0.0, maxval=1.0) 
             
    # Get the resolution 
    df = fs/(im.shape[0]-1) 
    dt = duration/(im.shape[1]-1) 
     
    # Extent 
    extent = [0, duration, 0, fs/2] 
     
    # flip the image vertically 
    if flipud: im = np.flip(im, 0) 
     
    # Display 
    if display :  
        ylabel =kwargs.pop('ylabel','Frequency [Hz]') 
        xlabel =kwargs.pop('xlabel','Time [sec]')  
        title  =kwargs.pop('title','loaded spectrogram') 
        cmap   =kwargs.pop('cmap','gray')  
        figsize=kwargs.pop('figsize',(4, 0.33*(extent[1]-extent[0])))  
        vmin=kwargs.pop('vmin',np.min(im))  
        vmax=kwargs.pop('vmax',np.max(im))  
         
        _, fig = plot2D (im, extent=extent, figsize=figsize,title=title,  
                         ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax, 
                         cmap=cmap, **kwargs) 
     
    return im, extent, dt, df 
 
#**************************************************************************** 
#*************               noise_subtraction                    *********** 
#**************************************************************************** 
def remove_background (Sxx, gauss_win=50, gauss_std = 25, beta1=1, beta2=1,  
                      llambda=1, display = False, savefig=None, **kwargs): 
    """ 
    Remove the background noise using spectral subtraction 
 
    Based on the spectrum of the A posteriori noise profile.   
    It computes an atenuation map in the time-frequency domain.  
    See [1]_ or [2]_ for more detail about the algorithm. 
 
    Parameters 
    ---------- 
    Sxx : 2d ndarray of scalars 
        Spectrogram  
         
    gauss_win=50 : int, optional, default: 50 
        Number of points in the gaussian window  
         
    gauss_std = 25 
        The standard deviation, sigma used to create the gaussian window   
         
    beta1 : scaler, optional, default: 1       
        beta1 has to be >0 
        Should be close to 1 
 
    beta2: scaler, optional, default: 1        
        beta2 has to be >0  
        better to not change 
     
    llambda : int, optional, default: 1        
        over-subtraction factor to compensate variation of noise amplitude. 
        Should be close to 1 
    
    verbose : boolean, optional, default is False
        Print messages and speed
         
    display : boolean, optional, default is False 
        Display the signal if True 
         
    savefig : string, optional, default is None 
        Root filename (with full path) is required to save the figures. Postfix 
        is added to the root filename. 
         
    \*\*kwargs, optional. This parameter is used by plt.plot and savefig functions 
             
        - savefilename : str, optional, default :'_spectro_after_noise_subtraction.png' 
            Postfix of the figure filename 
         
        - figsize : tuple of integers, optional, default: (4,10) 
            width, height in inches.   
         
        - title : string, optional, default : 'Spectrogram' 
            title of the figure 
         
        - xlabel : string, optional, default : 'Time [s]' 
            label of the horizontal axis 
         
        - ylabel : string, optional, default : 'Amplitude [AU]' 
            label of the vertical axis 
         
        - cmap : string or Colormap object, optional, default is 'gray' 
            See https://matplotlib.org/examples/color/colormaps_reference.html 
            in order to get all the  existing colormaps 
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic',  
            'viridis'... 
         
        - vmin, vmax : scalar, optional, default: None 
            `vmin` and `vmax` are used in conjunction with norm to normalize 
            luminance data.  Note if you pass a `norm` instance, your 
            settings for `vmin` and `vmax` will be ignored. 
         
        - extent : scalars (left, right, bottom, top), optional, default: None 
            The location, in data-coordinates, of the lower-left and 
            upper-right corners. If `None`, the image is positioned such that 
            the pixel centers fall on zero-based (row, column) indices. 
         
        - dpi : integer, optional, default is 96 
            Dot per inch.  
            For printed version, choose high dpi (i.e. dpi=300) => slow 
            For screen version, choose low dpi (i.e. dpi=96) => fast 
         
        - format : string, optional, default is 'png' 
            Format to save the figure 
             
        ... and more, see matplotlib     
                       
    Returns 
    ------- 
    Sxx_out : 2d ndarray of scalar 
        Spectrogram after denoising   
    noise_profile : 1d darray of scalar
        noise_profile
    BGNxx : 2d ndarray of scalar 
        Noise map
 
    References 
    ---------- 
    .. [1] Steven F. Boll, "Suppression of Acoustic Noise in Speech Using Spectral 
       Subtraction", IEEE Transactions on Signal Processing, 27(2),pp 113-120, 
       1979 
 
    .. [2] Y. Ephraim and D. Malah, Speech enhancement using a minimum mean square  
       error short-time spectral amplitude estimator, IEEE. Transactions in 
       Acoust., Speech, Signal Process., vol. 32, no. 6, pp. 11091121, Dec. 1984.   

    Examples:
    ---------
    
    Load audio recording and convert it into spectrogram
    
    >>> s, fs = maad.sound.load('../data/guyana_tropical_forest.wav')
    >>> Sxx,tn,fn,ext = maad.sound.spectrogram (s, fs)   
    
    Convert linear spectrogram into dB and add 96dB (which is the maximum dB
    for 16 bits wav) in order to have positive values
    
    >>> Sxx_dB = maad.util.power2dB(Sxx) + 96
    
    Remove stationnary noise from the spectrogram in dB
    
    >>> Sxx_dB_noNoise, noise_profile, _ = maad.rois.remove_background(Sxx_dB)

    Plot both spectrograms
    
    >>> import matplotlib.pyplot as plt 
    >>> fig, (ax1, ax2) = plt.subplots(2, 1)
    >>> maad.util.plot2D(Sxx_dB, ax=ax1, extent=ext, title='original', vmin=np.median(Sxx_dB), vmax=np.median(Sxx_dB)+40)
    >>> maad.util.plot2D(Sxx_dB_noNoise, ax=ax2, extent=ext, title='Without stationary noise', vmin=np.median(Sxx_dB_noNoise), vmax=np.median(Sxx_dB_noNoise)+40)
    >>> fig.set_size_inches(15,8)
    >>> fig.tight_layout()
       
    """   
    
    print(72 * '_' ) 
    print('Determine the profile of the stochastic background noise...') 
     
    Nf, Nw = Sxx.shape 

    # average spectrum (assumed to be ergodic) 
    mean_profile=np.mean(Sxx,1) 
     
    # White Top Hat (to remove non uniform background) = i - opening(i) 
    selem = signal.gaussian(gauss_win, gauss_std) 
    noise_profile = morphology.grey_opening(mean_profile, structure=selem) 
 
    # Remove the artefact at the end of the spectrum (2 highest frequencies) 
    noise_profile[-2:] = mean_profile [-2:] 
    noise_profile[:2] = mean_profile [:2] 
     
    # Create a matrix with the noise profile 
    noise_spectro=np.kron(np.ones((Nw,1)),noise_profile) 
    noise_spectro=noise_spectro.transpose() 
         
    # snr estimate a posteriori 
    SNR_est=(Sxx/noise_spectro) -1   
    SNR_est=SNR_est*(SNR_est>0) # keep only positive values 
     
    # compute attenuation map 
    # if llambda, beta1 and beta 2 are equal to 1, it is (1 - noise_spectro) 
    an_lk=(1-llambda*((1./(SNR_est+1))**beta1))**beta2 
    an_lk=an_lk*(an_lk>0) # keep only positive values 
     
    print('Remove the stochastic background noise...') 
     
    # Apply the attenuation map to the STFT coefficients 
    Sxx_out=an_lk*Sxx 
    
    # noise map BGNxx
    BGNxx = Sxx - Sxx_out
     
    # if nan in the image, convert nan into 0 
    np.nan_to_num(Sxx_out,0) 
    
    # Set negative value to 0
    Sxx_out[Sxx_out<0] = 0 
    
    print(np.max(Sxx_out))
    
    # Display 
    if display :  
        ylabel =kwargs.pop('ylabel','Frequency [Hz]') 
        xlabel =kwargs.pop('xlabel','Time [sec]')  
        title  =kwargs.pop('title','Spectrogram without stationnary noise') 
        cmap   =kwargs.pop('cmap','gray')  
        vmin=kwargs.pop('vmin',np.min(Sxx_out))  
        vmax=kwargs.pop('vmax',np.max(Sxx_out)) 
        extent=kwargs.pop('extent',None)
            
        if extent is not None : 
            fn = np.arange(0, Nf)*(extent[3]-extent[2])/(Nf-1) + extent[2]  
            xlabel = 'frequency [Hz]' 
            figsize=kwargs.pop('figsize', (4, 0.33*(extent[1]-extent[0])))
        else: 
            fn = np.arange(Nf) 
            xlabel = 'pseudofrequency [points]'
            figsize=kwargs.pop('figsize',(4, 13))  
        
        _, fig = plot2D (Sxx_out, extent=extent, figsize=figsize,title=title,  
                         ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax, 
                         cmap=cmap, **kwargs) 
        
        fig2, (ax1, ax2) = plt.subplots(2, sharex=True)
        fig2.set_size_inches((5,4))
        ax1,_ = plot1D(fn, mean_profile, ax=ax1, legend='Original profile',
                       color = 'b',
                       xlabel = '', ylabel = 'Amplitude [dB]', figtitle='') 
        ax1,_ = plot1D(fn, np.mean(BGNxx, axis=1), ax =ax1, legend='Noise profile',
                       color = 'r',
                       xlabel = '', ylabel = 'Amplitude [dB]', figtitle='')
        ax2,_ = plot1D(fn, np.mean(Sxx_out,axis=1), ax=ax2, color = 'k', 
                       legend='Denoized profile', 
                       xlabel = xlabel, ylabel = 'Amplitude [dB]', figtitle='') 
        fig2.tight_layout()  
        
        # SAVE FIGURE 
        if savefig is not None :  
            dpi   =kwargs.pop('dpi',96)             
            dpi=kwargs.pop('dpi', 96)  
            bbox_inches=kwargs.pop('bbox_inches', 'tight')  
            format=kwargs.pop('format','png') 
            savefilename=kwargs.pop('savefilename', '_spectro_after_noise_subtraction')   
            filename = savefig+savefilename+'.'+format 
            print('\n''save figure : %s' %filename) 
            fig.savefig(fname=filename, dpi=dpi, bbox_inches=bbox_inches, 
                        format=format, **kwargs)  
             
    return Sxx_out, noise_profile, BGNxx
 
def median_equalizer (Sxx, display=False, savefig=None, **kwargs): 
    """ 
    Median equalizer : remove background noise in a spectrogram 
     
    Parameters 
    ---------- 
    Sxx : 2D numpy array  
        Original spectrogram (or image), !!! not in dB 
        
    display : boolean, optional, default is False 
        Display the signal if True 
         
    savefig : string, optional, default is None 
        Root filename (with full path) is required to save the figures. Postfix 
        is added to the root filename. 
         
    \*\*kwargs, optional. This parameter is used by plt.plot and savefig functions 
             
        - savefilename : str, optional, default :'_spectro_after_noise_subtraction.png' 
            Postfix of the figure filename 
         
        - figsize : tuple of integers, optional, default: (4,10) 
            width, height in inches.   
         
        - title : string, optional, default : 'Spectrogram' 
            title of the figure 
         
        - xlabel : string, optional, default : 'Time [s]' 
            label of the horizontal axis 
         
        - ylabel : string, optional, default : 'Amplitude [AU]' 
            label of the vertical axis 
         
        - cmap : string or Colormap object, optional, default is 'gray' 
            See https://matplotlib.org/examples/color/colormaps_reference.html 
            in order to get all the  existing colormaps 
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic',  
            'viridis'... 
         
        - vmin, vmax : scalar, optional, default: None 
            `vmin` and `vmax` are used in conjunction with norm to normalize 
            luminance data.  Note if you pass a `norm` instance, your 
            settings for `vmin` and `vmax` will be ignored. 
         
        - extent : scalars (left, right, bottom, top), optional, default: None 
            The location, in data-coordinates, of the lower-left and 
            upper-right corners. If `None`, the image is positioned such that 
            the pixel centers fall on zero-based (row, column) indices. 
         
        - dpi : integer, optional, default is 96 
            Dot per inch.  
            For printed version, choose high dpi (i.e. dpi=300) => slow 
            For screen version, choose low dpi (i.e. dpi=96) => fast 
         
        - format : string, optional, default is 'png' 
            Format to save the figure 
             
        ... and more, see matplotlib     
                       
    Returns 
    ------- 
    Sxx_out : 2d ndarray of scalar 
        Spectrogram after denoising   
     
    References 
    ---------- 
    .. [1] This function has been proposed first by Carol BEDOYA <carol.bedoya@pg.canterbury.ac.nz> 
       Adapted by S. Haupert Oct 9, 2018 for Python 
       
    Examples:
    ---------
    
    Load audio recording and convert it into spectrogram
    
    >>> s, fs = maad.sound.load('../data/guyana_tropical_forest.wav')
    >>> Sxx,tn,fn,ext = maad.sound.spectrogram (s, fs)   
    
    Convert linear spectrogram into dB
    
    >>> Sxx_dB = maad.util.power2dB(Sxx) +96
    
    Remove stationnary noise from the spectrogram 
    
    >>> Sxx_noNoise = maad.rois.median_equalizer(Sxx)
    >>> Sxx_dB_noNoise =  maad.util.power2dB(Sxx_noNoise) 

    Plot both spectrograms
    
    >>> import matplotlib.pyplot as plt 
    >>> fig, (ax1, ax2) = plt.subplots(2, 1)
    >>> maad.util.plot2D(Sxx_dB, ax=ax1, extent=ext, title='original', vmin=np.median(Sxx_dB), vmax=np.median(Sxx_dB)+40)
    >>> maad.util.plot2D(Sxx_dB_noNoise, ax=ax2, extent=ext, title='Without stationary noise',vmin=np.median(Sxx_dB_noNoise), vmax=np.median(Sxx_dB_noNoise)+40)
    >>> fig.set_size_inches(15,8)
    >>> fig.tight_layout() 
       
    """  
     
    Sxx_out = (Sxx-np.median(Sxx,axis=1)[..., np.newaxis])
    
    # Numerator for normalization. Test if values of norm are <=0 and set them
    # to the highest value in Sxx. This will ensure that the result of the 
    # normalization will be lower than 1
    norm = (np.median(Sxx,axis=1)-np.min(Sxx,axis=1))
    norm[norm<=0] = Sxx.max()
    
    # normalization. Test if the numerator is 0
    Sxx_out = Sxx_out/norm[..., np.newaxis]
    
    # if the ratio is < 1, set the value to 1. 
    # Values < 1 are noise and should not be less than 1.
    # When Sxx_out is converted into dB => log10(1) => 0
    Sxx_out[Sxx_out<1] = 1
    
    # Display 
    if display :  
        ylabel =kwargs.pop('ylabel','Frequency [Hz]') 
        xlabel =kwargs.pop('xlabel','Time [sec]')  
        title  =kwargs.pop('title','Spectrogram without stationnary noise') 
        cmap   =kwargs.pop('cmap','gray')  
        extent=kwargs.pop('extent',None) 
        
        if extent is not None :
            figsize=kwargs.pop('figsize',(4, 0.33*(extent[1]-extent[0])))  
        else:
            figsize=kwargs.pop('figsize',(4, 13)) 
            
        # convert into dB
        Sxx_out_dB = power2dB(Sxx_out)
        
        vmin=kwargs.pop('vmin',0)  
        vmax=kwargs.pop('vmax',np.max(Sxx_out_dB)) 
         
        _, fig = plot2D (Sxx_out_dB, extent=extent, figsize=figsize,title=title,  
                         ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax, 
                         cmap=cmap, **kwargs) 
        # SAVE FIGURE 
        if savefig is not None :  
            dpi   =kwargs.pop('dpi',96)             
            dpi=kwargs.pop('dpi', 96)  
            bbox_inches=kwargs.pop('bbox_inches', 'tight')  
            format=kwargs.pop('format','png') 
            savefilename=kwargs.pop('savefilename', '_spectro_after_noise_subtraction')   
            filename = savefig+savefilename+'.'+format 
            print('\n''save figure : %s' %filename) 
            fig.savefig(fname=filename, dpi=dpi, bbox_inches=bbox_inches, 
                        format=format, **kwargs)  
    
    return Sxx_out 

def remove_background_morpho (Sxx, q =0.1, display=False, savefig=None, **kwargs): 
    """ 
    Remove background noise in a spectrogram using mathematical morphology tool.
     
    Parameters 
    ---------- 
    Sxx : 2D numpy array  
        Original spectrogram (or image) 
        
    q : float
        Quantile which must be between  0 and 1 inclusive. The closest to one, 
        the finest details are kept
     
    display : boolean, optional, default is False 
        Display the signal if True 
         
    savefig : string, optional, default is None 
        Root filename (with full path) is required to save the figures. Postfix 
        is added to the root filename. 
         
    \*\*kwargs, optional. This parameter is used by plt.plot and savefig functions 
             
        - savefilename : str, optional, default :'_spectro_after_noise_subtraction.png' 
            Postfix of the figure filename 
         
        - figsize : tuple of integers, optional, default: (4,10) 
            width, height in inches.   
         
        - title : string, optional, default : 'Spectrogram' 
            title of the figure 
         
        - xlabel : string, optional, default : 'Time [s]' 
            label of the horizontal axis 
         
        - ylabel : string, optional, default : 'Amplitude [AU]' 
            label of the vertical axis 
         
        - cmap : string or Colormap object, optional, default is 'gray' 
            See https://matplotlib.org/examples/color/colormaps_reference.html 
            in order to get all the  existing colormaps 
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic',  
            'viridis'... 
         
        - vmin, vmax : scalar, optional, default: None 
            `vmin` and `vmax` are used in conjunction with norm to normalize 
            luminance data.  Note if you pass a `norm` instance, your 
            settings for `vmin` and `vmax` will be ignored. 
         
        - extent : scalars (left, right, bottom, top), optional, default: None 
            The location, in data-coordinates, of the lower-left and 
            upper-right corners. If `None`, the image is positioned such that 
            the pixel centers fall on zero-based (row, column) indices. 
         
        - dpi : integer, optional, default is 96 
            Dot per inch.  
            For printed version, choose high dpi (i.e. dpi=300) => slow 
            For screen version, choose low dpi (i.e. dpi=96) => fast 
         
        - format : string, optional, default is 'png' 
            Format to save the figure 
             
        ... and more, see matplotlib     
                       
    Returns 
    ------- 
    Sxx_out : 2d ndarray of scalar 
        Spectrogram after denoising   
    noise_profile : 1d ndarray of scalar
        Noise profile
    BGNxx : 2d ndarray of scalar 
        Noise map
    
    Examples:
    ---------
    
    Load audio recording and convert it into spectrogram
    
    >>> s, fs = maad.sound.load('../data/guyana_tropical_forest.wav')
    >>> Sxx,tn,fn,ext = maad.sound.spectrogram (s, fs)   
    
    Convert linear spectrogram into dB
    
    >>> Sxx_dB = maad.util.power2dB(Sxx) +96
    
    Remove stationnary noise from the spectrogram 
    
    >>> Sxx_dB_noNoise,_,_ = maad.rois.remove_background_morpho(Sxx_dB, q=0.5)

    Plot both spectrograms
    
    >>> import matplotlib.pyplot as plt 
    >>> fig, (ax1, ax2) = plt.subplots(2, 1)
    >>> maad.util.plot2D(Sxx_dB, ax=ax1, extent=ext, title='original', vmin=np.median(Sxx_dB), vmax=np.median(Sxx_dB)+40)
    >>> maad.util.plot2D(Sxx_dB_noNoise, ax=ax2, extent=ext, title='Without stationary noise',vmin=np.median(Sxx_dB_noNoise), vmax=np.median(Sxx_dB_noNoise)+40)
    >>> fig.set_size_inches(15,8)
    >>> fig.tight_layout()     
    
    Load audio recording and convert it into spectrogram
    
    >>> s, fs = maad.sound.load('../data/guyana_tropical_forest.wav')
    >>> Sxx,tn,fn,ext = maad.sound.spectrogram (s, fs, tcrop=(0,20))   
    >>> Sxx_dB = maad.util.power2dB(Sxx) +96
    
    Remove stationnary noise from the spectrogram with different q
    
    >>> Sxx_dB_noNoise_q25,_,_ = maad.rois.remove_background_morpho(Sxx_dB, q=0.25)
    >>> Sxx_dB_noNoise_q50,_,_ = maad.rois.remove_background_morpho(Sxx_dB, q=0.5)
    >>> Sxx_dB_noNoise_q75,_,_ = maad.rois.remove_background_morpho(Sxx_dB, q=0.75)
    
    Plot 3 spectrograms
    
    >>> import matplotlib.pyplot as plt 
    >>> fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    >>> maad.util.plot2D(Sxx_dB_noNoise_q25, ax=ax1, extent=ext, title='Without stationary noise (q=0.25)',vmin=np.median(Sxx_dB_noNoise_q25), vmax=np.median(Sxx_dB_noNoise_q25)+40)
    >>> maad.util.plot2D(Sxx_dB_noNoise_q50, ax=ax2, extent=ext, title='Without stationary noise (q=0.50)',vmin=np.median(Sxx_dB_noNoise_q50), vmax=np.median(Sxx_dB_noNoise_q50)+40)
    >>> maad.util.plot2D(Sxx_dB_noNoise_q75, ax=ax3, extent=ext, title='Without stationary noise (q=0.75)',vmin=np.median(Sxx_dB_noNoise_q75), vmax=np.median(Sxx_dB_noNoise_q75)+40)
    >>> fig.set_size_inches(15,9)
    >>> fig.tight_layout()     
        
    """  
    
    # Use morpho math tools to estimate the background noise
    BGNxx = reconstruction(seed=Sxx-(np.quantile(Sxx, q)), mask=Sxx, method='dilation')
    Sxx_out = Sxx - BGNxx 
    
    # noise profile along time axis
    noise_profile = np.mean(BGNxx,1)
    
    # Set negative value to 0
    Sxx_out[Sxx_out<0] = 0 
    
    # Display 
    if display :  
        ylabel =kwargs.pop('ylabel','Frequency [Hz]') 
        xlabel =kwargs.pop('xlabel','Time [sec]')  
        title  =kwargs.pop('title','Spectrogram without stationnary noise') 
        cmap   =kwargs.pop('cmap','gray')  
        vmin=kwargs.pop('vmin',np.min(Sxx_out))  
        vmax=kwargs.pop('vmax',np.max(Sxx_out)) 
        extent=kwargs.pop('extent',None) 
        
        Nf, Nw = Sxx.shape 

        if extent is not None : 
            fn = np.arange(0, Nf)*(extent[3]-extent[2])/(Nf-1) + extent[2]  
            xlabel = 'frequency [Hz]' 
            figsize=kwargs.pop('figsize', (4, 0.33*(extent[1]-extent[0])))
        else: 
            fn = np.arange(Nf) 
            xlabel = 'pseudofrequency [points]'
            figsize=kwargs.pop('figsize',(4, 13))  
        
        _, fig = plot2D (BGNxx, extent=extent, figsize=figsize,title='Noise map',  
                         ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax, 
                         cmap=cmap, **kwargs) 
        
        _, fig = plot2D (Sxx_out, extent=extent, figsize=figsize,title=title,  
                         ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax, 
                         cmap=cmap, **kwargs) 
        
        fig2, (ax1, ax2) = plt.subplots(2, sharex=True)
        fig2.set_size_inches((5,4))
        ax1,_ = plot1D(fn, np.mean(Sxx,axis=1), ax=ax1, legend='Original profile', 
                       color = 'b',
                       xlabel = '', ylabel = 'Amplitude [dB]', figtitle='') 
        ax1,_ = plot1D(fn, np.mean(BGNxx,1), ax =ax1, legend='Noise profile', 
                       color = 'r',
                       xlabel = '', ylabel = 'Amplitude [dB]', figtitle='')
        ax2,_ = plot1D(fn, np.mean(Sxx_out,axis=1), ax=ax2, color = 'k', 
                       legend='Denoized profile', 
                       xlabel = xlabel, ylabel = 'Amplitude [dB]', figtitle='') 
        fig2.tight_layout()     
        
        # SAVE FIGURE 
        if savefig is not None :  
            dpi   =kwargs.pop('dpi',96)             
            dpi=kwargs.pop('dpi', 96)  
            bbox_inches=kwargs.pop('bbox_inches', 'tight')  
            format=kwargs.pop('format','png') 
            savefilename=kwargs.pop('savefilename', '_spectro_after_noise_subtraction')   
            filename = savefig+savefilename+'.'+format 
            print('\n''save figure : %s' %filename) 
            fig.savefig(fname=filename, dpi=dpi, bbox_inches=bbox_inches, 
                        format=format, **kwargs)  
 
    return Sxx_out, noise_profile, BGNxx
 
    
def remove_background_along_axis (Sxx, mode ='median', axis=1, N=25, N_bins=50, 
                                  display=False, savefig=None, **kwargs): 
    """ 
    Get the noisy profile along the defined axis and remove this profile from
    the spectrogram
    
    Parameters 
    ---------- 
    Sxx : 2D numpy array  
        Original spectrogram (or image) 
    
    mode : str, optional, default is 'median'
        Select the mode to remove the noise
        Possible values for mode are :
        - 'ale' : Adaptative Level Equalization algorithm [Lamel & al. 1981]
        - 'median' : subtract the median value
        - 'mean' : subtract the mean value (DC)
    
    axis : integer, default is 1
        if matrix, estimate the mode for each row (axis=0) or each column (axis=1)
        
    N : int, default is 25
        length of window to compute the running mean of the noise profile
        
    N_bins : int (only for mode = "ale"), default is 50
        number of bins to compute the histogram 
     
    display : boolean, optional, default is False 
        Display the signal if True 
         
    savefig : string, optional, default is None 
        Root filename (with full path) is required to save the figures. Postfix 
        is added to the root filename. 
         
    \*\*kwargs, optional. This parameter is used by plt.plot and savefig functions 
             
        - savefilename : str, optional, default :'_spectro_after_noise_subtraction.png' 
            Postfix of the figure filename 
         
        - figsize : tuple of integers, optional, default: (4,10) 
            width, height in inches.   
         
        - title : string, optional, default : 'Spectrogram' 
            title of the figure 
         
        - xlabel : string, optional, default : 'Time [s]' 
            label of the horizontal axis 
         
        - ylabel : string, optional, default : 'Amplitude [AU]' 
            label of the vertical axis 
         
        - cmap : string or Colormap object, optional, default is 'gray' 
            See https://matplotlib.org/examples/color/colormaps_reference.html 
            in order to get all the  existing colormaps 
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic',  
            'viridis'... 
         
        - vmin, vmax : scalar, optional, default: None 
            `vmin` and `vmax` are used in conjunction with norm to normalize 
            luminance data.  Note if you pass a `norm` instance, your 
            settings for `vmin` and `vmax` will be ignored. 
         
        - extent : scalars (left, right, bottom, top), optional, default: None 
            The location, in data-coordinates, of the lower-left and 
            upper-right corners. If `None`, the image is positioned such that 
            the pixel centers fall on zero-based (row, column) indices. 
         
        - dpi : integer, optional, default is 96 
            Dot per inch.  
            For printed version, choose high dpi (i.e. dpi=300) => slow 
            For screen version, choose low dpi (i.e. dpi=96) => fast 
         
        - format : string, optional, default is 'png' 
            Format to save the figure 
             
        ... and more, see matplotlib    
        
    Reference:
    ---------
    .. [1] Towsey, M., 2013b. Noise Removal from Wave-forms and Spectrograms Derived from
    Natural Recordings of the Environment. Queensland University of Technology,
    Brisbane
                       
    Returns 
    ------- 
    Sxx_out : 2d ndarray of scalar 
        Spectrogram after denoising 
        
    noise_profile : 1d ndarray of scalar
        Noise profile
        
    Examples:
    ---------
    
    Load audio recording and convert it into spectrogram
    
    >>> s, fs = maad.sound.load('../data/guyana_tropical_forest.wav')
    >>> Sxx,tn,fn,ext = maad.sound.spectrogram (s, fs)   
    
    Convert linear spectrogram into dB
    
    >>> Sxx_dB = maad.util.power2dB(Sxx) +96
    
    Remove stationnary noise from the spectrogram 
    - with mode 'ale"
    
    >>> Sxx_dB_noNoise_ale,_ = maad.rois.remove_background_along_axis(Sxx_dB, mode='ale')
    
    - with mode 'median"
    
    >>> Sxx_dB_noNoise_med,_ = maad.rois.remove_background_along_axis(Sxx_dB, mode='median')
    
    - with mode 'mean"
    
    >>> Sxx_dB_noNoise_mean,_ = maad.rois.remove_background_along_axis(Sxx_dB, mode='mean')

    Plot spectrograms
    
    >>> import matplotlib.pyplot as plt 
    >>> fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    >>> maad.util.plot2D(Sxx_dB, ax=ax1, extent=ext, title='original', vmin=np.median(Sxx_dB), vmax=np.median(Sxx_dB)+40)
    >>> maad.util.plot2D(Sxx_dB_noNoise_ale, ax=ax2, extent=ext, title='Without stationary noise (mode = ''ale'')',vmin=np.median(Sxx_dB_noNoise_ale), vmax=np.median(Sxx_dB_noNoise_ale)+40)
    >>> maad.util.plot2D(Sxx_dB_noNoise_med, ax=ax3, extent=ext, title='Without stationary noise (mode = ''med'')',vmin=np.median(Sxx_dB_noNoise_med), vmax=np.median(Sxx_dB_noNoise_med)+40)
    >>> maad.util.plot2D(Sxx_dB_noNoise_mean, ax=ax4, extent=ext, title='Without stationary noise (mode = ''mean'')',vmin=np.median(Sxx_dB_noNoise_mean), vmax=np.median(Sxx_dB_noNoise_mean)+40)
    >>> fig.set_size_inches(8,10)
    >>> fig.tight_layout()   
    
    """
       
    # get the noise profile, N define the running mean size of the histogram
    # in case of mode='ale'
    noise_profile = get_unimode (Sxx, mode, axis, N=7, N_bins=N_bins)
    # smooth the profile by removing spurious thin peaks
    noise_profile = running_mean(noise_profile,N)
    # Remove horizontal noisy peaks profile (BGN_VerticalNoise is an estimation) 
    # and negative value to zero
    if axis == 1 :
        Sxx_out = Sxx - noise_profile[..., np.newaxis]
    elif axis == 0 :
        Sxx_out = Sxx - noise_profile[np.newaxis, ...]
    
    # Set negative value to 0
    Sxx_out[Sxx_out<0] = 0 
    
    # Display 
    if display :  
        
        ylabel =kwargs.pop('ylabel','Frequency [Hz]') 
        xlabel =kwargs.pop('xlabel','Time [sec]')  
        title  =kwargs.pop('title','Spectrogram without stationnary noise') 
        cmap   =kwargs.pop('cmap','gray')  
        vmin=kwargs.pop('vmin',np.min(Sxx_out))  
        vmax=kwargs.pop('vmax',np.max(Sxx_out))  
        extent=kwargs.pop('extent',None) 
        
        Nf, Nw = Sxx.shape 
        
        if extent is not None : 
            fn = np.arange(0, Nf)*(extent[3]-extent[2])/(Nf-1) + extent[2]  
            xlabel = 'frequency [Hz]' 
            figsize=kwargs.pop('figsize', (4, 0.33*(extent[1]-extent[0])))
        else: 
            fn = np.arange(Nf) 
            xlabel = 'pseudofrequency [points]'
            figsize=kwargs.pop('figsize',(4, 13))  
            
        _, fig1 = plot2D (Sxx_out, extent=extent, figsize=figsize,title=title,  
                         ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax, 
                         cmap=cmap, **kwargs) 
        
        fig2, (ax1, ax2) = plt.subplots(2, sharex=True)
        fig2.set_size_inches((5,4))
        ax1,_ = plot1D(fn, mean_dB(Sxx,axis=axis), ax=ax1, legend='Original profile',
                       color = 'b',
                       xlabel = '', ylabel = 'Amplitude [dB]', figtitle='') 
        ax1,_ = plot1D(fn, noise_profile, ax =ax1, legend='Noise profile', 
                       color = 'r',
                       xlabel = '', ylabel = 'Amplitude [dB]', figtitle='')
        ax2,_ = plot1D(fn, mean_dB(Sxx_out,axis=axis), ax=ax2, color = 'k', 
                       legend='Denoized profile', 
                       xlabel = xlabel, ylabel = 'Amplitude [dB]', figtitle='') 
        fig2.tight_layout()
        
        # SAVE FIGURE 
        if savefig is not None :  
            dpi   =kwargs.pop('dpi',96)             
            dpi=kwargs.pop('dpi', 96)  
            bbox_inches=kwargs.pop('bbox_inches', 'tight')  
            format=kwargs.pop('format','png') 
            savefilename=kwargs.pop('savefilename', '_spectro_after_noise_subtraction')   
            filename = savefig+savefilename+'.'+format 
            print('\n''save figure : %s' %filename) 
            fig1.savefig(fname=filename, dpi=dpi, bbox_inches=bbox_inches, 
                        format=format, **kwargs) 
                
    return Sxx_out, noise_profile 

"""**************************************************************************** 
*************                      smooth                            *********** 
****************************************************************************""" 
def smooth (im, std=1, verbose=False, display = False, savefig=None, **kwargs): 
    """ 
    Smooth a spectrogram with a gaussian filter 
     
    Parameters 
    ---------- 
    im : 2d ndarray of scalars 
        Spectrogram (or image) 
     
    std : scalar, optional, default is 1 
        Standard deviation of the gaussian kernel used to smooth the image 
        The larger is the number, the smoother will be the image and the longer 
        it takes. Standard values should fall between 0.5 to 3 
    
    verbose : boolean, optional, default is False 
        print messages
    
    display : boolean, optional, default is False 
        Display the signal if True 
         
    savefig : string, optional, default is None 
        Root filename (with full path) is required to save the figures. Postfix 
        is added to the root filename. 
         
    \*\*kwargs, optional. This parameter is used by plt.plot and savefig functions 
             
        - savefilename : str, optional, default :'_spectro_after_noise_subtraction.png' 
            Postfix of the figure filename 
          
        - figsize : tuple of integers, optional, default: (4,10) 
            width, height in inches.   
         
        - title : string, optional, default : 'Spectrogram' 
            title of the figure 
         
        - xlabel : string, optional, default : 'Time [s]' 
            label of the horizontal axis 
         
        - ylabel : string, optional, default : 'Amplitude [AU]' 
            label of the vertical axis 
         
        - cmap : string or Colormap object, optional, default is 'gray' 
            See https://matplotlib.org/examples/color/colormaps_reference.html 
            in order to get all the  existing colormaps 
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic',  
            'viridis'... 
         
        - vmin, vmax : scalar, optional, default: None 
            `vmin` and `vmax` are used in conjunction with norm to normalize 
            luminance data.  Note if you pass a `norm` instance, your 
            settings for `vmin` and `vmax` will be ignored. 
         
        - extent : scalars (left, right, bottom, top), optional, default: None 
            The location, in data-coordinates, of the lower-left and 
            upper-right corners. If `None`, the image is positioned such that 
            the pixel centers fall on zero-based (row, column) indices. 
         
        - dpi : integer, optional, default is 96 
            Dot per inch.  
            For printed version, choose high dpi (i.e. dpi=300) => slow 
            For screen version, choose low dpi (i.e. dpi=96) => fast 
         
        - format : string, optional, default is 'png' 
            Format to save the figure 
             
        ... and more, see matplotlib    
         
    Returns 
    ------- 
    im_out: smothed or blurred image  
    
    Examples:
    ---------
    
    Load audio recording and convert it into spectrogram
    
    >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
    >>> Sxx,tn,fn,ext = maad.sound.spectrogram (s, fs, tcrop=(5,10), fcrop=(0,10000))   
    
    Convert linear spectrogram into dB
    
    >>> Sxx_dB = maad.util.power2dB(Sxx) +96
    
    Smooth the spectrogram
    
    >>> Sxx_dB_std05 = maad.rois.smooth(Sxx_dB, std=0.5)
    >>> Sxx_dB_std10 = maad.rois.smooth(Sxx_dB, std=1)
    >>> Sxx_dB_std15 = maad.rois.smooth(Sxx_dB, std=1.5)
    
    Plot spectrograms
    
    >>> import matplotlib.pyplot as plt 
    >>> fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    >>> maad.util.plot2D(Sxx_dB, ax=ax1, extent=ext, title='original', vmin=10, vmax=70)
    >>> maad.util.plot2D(Sxx_dB_std05, ax=ax2, extent=ext, title='smooth (std=0.5)', vmin=10, vmax=70)
    >>> maad.util.plot2D(Sxx_dB_std10, ax=ax3, extent=ext, title='smooth (std=1)', vmin=10, vmax=70)
    >>> maad.util.plot2D(Sxx_dB_std15, ax=ax4, extent=ext, title='smooth (std=1.5)', vmin=10, vmax=70)
    >>> fig.set_size_inches(7,9)
    >>> fig.tight_layout() 
    
    """ 
    
    if verbose:
        print(72 * '_') 
        print('Smooth the image with a gaussian filter (std = %.1f)' %std) 
     
    # use scikit image (faster than scipy) 
    im_out = filters.gaussian(im,std) 
     
    # Display 
    if display :  
        ylabel =kwargs.pop('ylabel','Frequency [Hz]') 
        xlabel =kwargs.pop('xlabel','Time [sec]')   
        cmap   =kwargs.pop('cmap','gray')  
        vmin=kwargs.pop('vmin',np.percentile(im_out,0.1)) 
        vmax=kwargs.pop('vmax',np.percentile(im_out,99.9)) 
        extent=kwargs.pop('extent',None)
            
        if extent is not None : 
            xlabel = 'frequency [Hz]' 
            figsize=kwargs.pop('figsize', (4*2, 0.33*(extent[1]-extent[0])))
        else: 
            xlabel = 'pseudofrequency [points]'
            figsize=kwargs.pop('figsize',(4*2, 13)) 
        
         
        fig, (ax1, ax2) = plt.subplots(2, 1)
        plot2D (im, ax=ax1, extent=extent, figsize=figsize,
                title=('Orignal Spectrogram'),  
                ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax, 
                cmap=cmap, **kwargs)
        plot2D (im_out, ax=ax2, extent=extent, figsize=figsize,
                title='Blurred Spectrogram (std='+str(std)+')',  
                ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax, 
                cmap=cmap, **kwargs) 
        
        # SAVE FIGURE 
        if savefig is not None :  
            dpi   =kwargs.pop('dpi',96) 
            format=kwargs.pop('format','png') 
            filename=kwargs.pop('filename','_spectro_blurr')              
            filename = savefig+filename+'.'+format 
            print('\n''save figure : %s' %filename) 
            fig.savefig(filename, bbox_inches='tight', dpi=dpi, format=format, 
                        **kwargs)    

    return im_out 
 
#**************************************************************************** 
#*************                double_threshold                    *********** 
#**************************************************************************** 
 
def _double_threshold_rel (im, bin_std=6, bin_per=0.5, 
                           verbose=False, display=False, savefig=None, **kwargs): 
    """ 
    Binarize an image based on a double relative threshold.  
    The values used for the thresholding depends on the values found in the  
    image. => relative threshold 
     
    Parameters 
    ---------- 
    im : 2d ndarray of scalars 
        Spectrogram (or image) 
 
    bin_std : scalar, optional, default is 6 
        Set the first threshold. This threshold is not an absolute value but 
        depends on values that are similar to 75th percentile (pseudo_mean) and 
        a sort of std value of the image.  
        threshold1 = "pseudo_mean" + "std" * bin_std    
        Value higher than threshold1 are set to 1, they are the seeds for  
        the second step. The others are set to 0.  
         
    bin_per: scalar, optional, defautl is 0.5 
        Set how much the second threshold is lower than the first 
        threshold value. From 0 to 1. ex: 0.1 = 10 %. 
        threshold2 = threshold1 (1-bin_per)   
        Value higher than threshold2 and connected (directly or not) to the  
        seeds are set to 1, the other remains 0 
 
    verbose : boolean, optional, default is False
        print messages
    
    display : boolean, optional, default is False 
        Display the signal if True 
         
    savefig : string, optional, default is None 
        Root filename (with full path) is required to save the figures. Postfix 
        is added to the root filename. 
         
    \*\*kwargs, optional. This parameter is used by plt.plot and savefig functions 
             
        - savefilename : str, optional, default :'_spectro_after_noise_subtraction.png' 
            Postfix of the figure filename 
          
        - figsize : tuple of integers, optional, default: (4,10) 
            width, height in inches.   
         
        - title : string, optional, default : 'Spectrogram' 
            title of the figure 
         
        - xlabel : string, optional, default : 'Time [s]' 
            label of the horizontal axis 
         
        - ylabel : string, optional, default : 'Amplitude [AU]' 
            label of the vertical axis 
         
        - cmap : string or Colormap object, optional, default is 'gray' 
            See https://matplotlib.org/examples/color/colormaps_reference.html 
            in order to get all the  existing colormaps 
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic',  
            'viridis'... 
         
        - vmin, vmax : scalar, optional, default: None 
            `vmin` and `vmax` are used in conjunction with norm to normalize 
            luminance data.  Note if you pass a `norm` instance, your 
            settings for `vmin` and `vmax` will be ignored. 
         
        - extent : scalars (left, right, bottom, top), optional, default: None 
            The location, in data-coordinates, of the lower-left and 
            upper-right corners. If `None`, the image is positioned such that 
            the pixel centers fall on zero-based (row, column) indices. 
         
        - dpi : integer, optional, default is 96 
            Dot per inch.  
            For printed version, choose high dpi (i.e. dpi=300) => slow 
            For screen version, choose low dpi (i.e. dpi=96) => fast 
         
        - format : string, optional, default is 'png' 
            Format to save the figure 
             
        ... and more, see matplotlib    
     
    Returns 
    ------- 
    im_out: binary image  
     
    References 
    ---------- 
    .. [1] from MATLAB: Threshold estimation (from Oliveira et al, 2015) 
       Adapted by S. Haupert Dec 12, 2017 
    """ 
    
    # test if im is full of zeros
    if not im.any() :
        im_out = np.zeros(im.shape)   
    else:
    
        # Compute the qth percentile of the data along the specified axis 
        val1 = np.percentile(im[np.where(im>0)],75)   # value corresponding to the limit between the 75% lowest value and 25% largest value 
         
        # The interquartile range (IQR) is the difference between the 75th and  
        # 25th percentile of the data. It is a measure of the dispersion similar  
        # to standard deviation or variance, but is much more robust against outliers 
        val2 = iqr(im[np.where(im>0)])*bin_std 
         
        # Threshold : qth  percentile + sort of std 
        h_th = val1 + val2 
        # Low threshold limit 
        l_th = (h_th-h_th*bin_per) 
        
        if verbose :
            print(72 * '_') 
            print('Double thresholding with values relative to the image...') 
            print ('**********************************************************') 
            print ('  high threshold value %.2f | low threshold value %.2f' % (h_th, l_th)) 
            print ('**********************************************************') 
         
        # binarisation  
        im_t1 = im > h_th    # mask1 
        im_t2 = im > l_th    # mask2 
        im_t3 = im * im_t1   # selected parts of the image 
         
        #find index of regions which meet the criteria 
        conncomp_t2 = measure.label(im_t2)    #Find connected components in binary image 
        rprops = measure.regionprops(conncomp_t2,im_t3)  
         
        rprops_mean_intensity = [region['mean_intensity'] for region in rprops]   
        rprops_mean_intensity = np.asarray(rprops_mean_intensity) 
         
        rprops_label = [region['label'] for region in rprops]  
        rprops_label = np.asarray(rprops_label) 
             
        [ind]=np.where(rprops_mean_intensity>0) 
         
        im_out = np.isin(conncomp_t2, rprops_label[ind])    # test if the indice is in the matrix of indices 
        im_out =im_out*1    #  boolean to 0,1 conversion 
                     
        # Display 
        if display :  
            ylabel =kwargs.pop('ylabel','Frequency [Hz]') 
            xlabel =kwargs.pop('xlabel','Time [sec]')  
            title  =kwargs.pop('title','binary image => MASK') 
            cmap   =kwargs.pop('cmap','gray')  
            vmin=kwargs.pop('vmin',0)  
            vmax=kwargs.pop('vmax',1)  
            extent=kwargs.pop('extent',None)
                
            if extent is not None : 
                xlabel = 'frequency [Hz]' 
                figsize=kwargs.pop('figsize', (4, 0.33*(extent[1]-extent[0])))
            else: 
                xlabel = 'pseudofrequency [points]'
                figsize=kwargs.pop('figsize',(4, 13)) 
             
            _, fig = plot2D (im_out, extent=extent, figsize=figsize,title=title,  
                             ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax, 
                             cmap=cmap, **kwargs) 
            # SAVE FIGURE 
            if savefig is not None :  
                dpi   =kwargs.pop('dpi',96) 
                format=kwargs.pop('format','png') 
                filename=kwargs.pop('filename','_spectro_binary')              
                filename = savefig+filename+'.'+format 
                if verbose :
                    print('\n''save figure : %s' %filename) 
                fig.savefig(filename, bbox_inches='tight', dpi=dpi, format=format, 
                            **kwargs)    
 
    return im_out 
 
#**************************************************************************** 
#*************                double_threshold                    *********** 
#**************************************************************************** 
def _double_threshold_abs(im, bin_h=0.7, bin_l=0.2, 
                          verbose=False,display=False, savefig=None, **kwargs): 
    """ 
    Binarize an image based on a double relative threshold.  
    The values used for the thresholding are independent of the values in the 
     
    image => absolute threshold 
     
    Parameters 
    ---------- 
    im : 2d ndarray of scalars 
        Spectrogram (or image) 

    bin_h : scalar, optional, default is 0.7 
        Set the first threshold. Value higher than this value are set to 1,  
        the others are set to 0. They are the seeds for the second step 
         
    bin_l: scalar, optional, defautl is 0.2 
        Set the second threshold. Value higher than this value and connected 
        to the seeds or to other pixels connected to the seeds are set to 1,  
        the other remains 0 
        
    verbose : boolean, optional, default is False
        print messages
 
    display : boolean, optional, default is False 
        Display the signal if True 
         
    savefig : string, optional, default is None 
        Root filename (with full path) is required to save the figures. Postfix 
        is added to the root filename. 
         
    \*\*kwargs, optional. This parameter is used by plt.plot and savefig functions 
           
        - savefilename : str, optional, default :'_spectro_after_noise_subtraction.png' 
            Postfix of the figure filename 
         
        - figsize : tuple of integers, optional, default: (4,10) 
            width, height in inches.   
         
        - title : string, optional, default : 'Spectrogram' 
            title of the figure 
         
        - xlabel : string, optional, default : 'Time [s]' 
            label of the horizontal axis 
         
        - ylabel : string, optional, default : 'Amplitude [AU]' 
            label of the vertical axis 
         
        - cmap : string or Colormap object, optional, default is 'gray' 
            See https://matplotlib.org/examples/color/colormaps_reference.html 
            in order to get all the  existing colormaps 
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic',  
            'viridis'... 
         
        - vmin, vmax : scalar, optional, default: None 
            `vmin` and `vmax` are used in conjunction with norm to normalize 
            luminance data.  Note if you pass a `norm` instance, your 
            settings for `vmin` and `vmax` will be ignored. 
         
        - extent : scalars (left, right, bottom, top), optional, default: None 
            The location, in data-coordinates, of the lower-left and 
            upper-right corners. If `None`, the image is positioned such that 
            the pixel centers fall on zero-based (row, column) indices. 
         
        - dpi : integer, optional, default is 96 
            Dot per inch.  
            For printed version, choose high dpi (i.e. dpi=300) => slow 
            For screen version, choose low dpi (i.e. dpi=96) => fast 
         
        - format : string, optional, default is 'png' 
            Format to save the figure 
             
        ... and more, see matplotlib    
 
    Returns 
    ------- 
    im_out: binary image  
     
    References 
    ---------- 
    .. [1] from MATLAB: Threshold estimation (from Oliveira et al, 2015) 
       Adapted by S. Haupert Dec 12, 2017 
    """ 
     
    # binarisation  
    im_t1 = im > bin_h  # mask1 
    im_t2 = im > bin_l  # mask2 
    im_t3 = im * im_t1  # selected parts of the image 
     
    #find index of regions which meet the criteria 
    conncomp_t2 = measure.label(im_t2)    #Find connected components in binary image 
    rprops = measure.regionprops(conncomp_t2,im_t3)  
     
    rprops_mean_intensity = [region['mean_intensity'] for region in rprops]   
    rprops_mean_intensity = np.asarray(rprops_mean_intensity) 
     
    rprops_label = [region['label'] for region in rprops]  
    rprops_label = np.asarray(rprops_label) 
         
    [ind]=np.where(rprops_mean_intensity>0) 
     
    im_out = np.isin(conncomp_t2, rprops_label[ind])    # test if the indice is in the maxtrix of indices 
    im_out =im_out*1    #  boolean to 0,1 conversion 
    
    if verbose :
        print(72 * '_') 
        print('Double thresholding with  absolute values...') 
        print ('**********************************************************') 
        print ('  Number of rois %.2f | Rois cover %.2f%' % (len(rprops_label), 
                                                             sum(im_out)/(im_out.shape[1]*im_out.shape[0])*100)) 
        print ('**********************************************************') 
                 
    # Display 
    if display :  
        ylabel =kwargs.pop('ylabel','Frequency [Hz]') 
        xlabel =kwargs.pop('xlabel','Time [sec]')  
        title  =kwargs.pop('title','binary image => MASK') 
        cmap   =kwargs.pop('cmap','gray')  
        vmin=kwargs.pop('vmin',0)  
        vmax=kwargs.pop('vmax',1)  
        extent=kwargs.pop('extent',None)
            
        if extent is not None : 
            xlabel = 'frequency [Hz]' 
            figsize=kwargs.pop('figsize', (4, 0.33*(extent[1]-extent[0])))
        else: 
            xlabel = 'pseudofrequency [points]'
            figsize=kwargs.pop('figsize',(4, 13)) 
         
        _, fig = plot2D (im_out, extent=extent, figsize=figsize,title=title,  
                         ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax, 
                         cmap=cmap, **kwargs) 
        # SAVE FIGURE 
        if savefig is not None :  
            dpi   =kwargs.pop('dpi',96) 
            format=kwargs.pop('format','png') 
            filename=kwargs.pop('filename','_spectro_binary')              
            filename = savefig+filename+'.'+format 
            if verbose :
                print('\n''save figure : %s' %filename) 
            fig.savefig(filename, bbox_inches='tight', dpi=dpi, format=format, 
                        **kwargs)    
 
    return im_out 
 
"""**************************************************************************** 
*************                   create_mask wrapper                 *********** 
****************************************************************************""" 
def create_mask(im, mode_bin = 'relative', 
                verbose= False, display = False, savefig = None, **kwargs): 
    """ 
    Binarize an image based on a double threshold.  
     
    Parameters 
    ---------- 
    im : 2d ndarray of scalars 
        Spectrogram (or image) 
 
    mode_bin : string in {'relative', 'absolute'}, optional, default is 'relative' 
        if 'relative', a relative double threshold is performed 
        if 'absolute', an double threshold with absolute value is performed 
        
    verbose : boolean, optional, default is False
        print messages
         
    display : boolean, optional, default is False 
        Display the signal if True 
         
    savefig : string, optional, default is None 
        Root filename (with full path) is required to save the figures. Postfix 
        is added to the root filename. 
         
    \*\*kwargs, optional. This parameter is used by the maad functions as well 
        as the plt.plot and savefig functions. 
        All the input arguments required or optional in the signature of the 
        functions above can be passed as kwargs : 
         
        - double_threshold_abs(im, bin_h=0.7, bin_l=0.2, display=False, savefig=None, \*\*kwargs) 
         
        - double_threshold_rel (im, bin_std=5, bin_per=0.5, display=False, savefig=None, \*\*kwargs) 
            
        ... and more, see matplotlib    
 
    Returns 
    ------- 
    im_bin: binary image 
     
    Examples 
    -------- 
    
    Load audio recording and convert it into spectrogram
    
    >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
    >>> Sxx,tn,fn,ext = maad.sound.spectrogram (s, fs, fcrop=(0,10000))   
    
    Convert linear spectrogram into dB
    
    >>> Sxx_dB = maad.util.power2dB(Sxx) +96
    
    Smooth the spectrogram
    
    >>> Sxx_dB_blurred = maad.rois.smooth(Sxx_dB)
    
    Detection of the acoustic signature => creation of a mask
    
    >>> im_bin = maad.rois.create_mask(Sxx_dB_blurred, bin_std=1.5, bin_per=0.25, mode='relative') 
    
    Plot spectrograms
    
    >>> import matplotlib.pyplot as plt 
    >>> fig, (ax1, ax2) = plt.subplots(2, 1)
    >>> maad.util.plot2D(Sxx_dB, ax=ax1, extent=ext, title='original', vmin=10, vmax=70)
    >>> maad.util.plot2D(im_bin, ax=ax2, extent=ext, title='mask)')
    >>> fig.set_size_inches(13,8)
    >>> fig.tight_layout() 
    
    """ 
        
    if mode_bin == 'relative': 
        bin_std=kwargs.pop('bin_std', 6)  
        bin_per=kwargs.pop('bin_per', 0.5)  
        im_bin = _double_threshold_rel(im, bin_std, bin_per, 
                                       verbose, display, savefig, **kwargs) 
         
    elif mode_bin == 'absolute': 
        bin_h=kwargs.pop('bin_h', 0.7)  
        bin_l=kwargs.pop('bin_l', 0.3)  
        im_bin = _double_threshold_abs(im, bin_h, bin_l,
                                       verbose, display, savefig, **kwargs)    
    
    return im_bin  
 
#**************************************************************************** 
#*************                 select_rois                   *********** 
#**************************************************************************** 
def select_rois(im_bin, min_roi=None ,max_roi=None, 
                verbose=False, display=False, savefig = None, **kwargs): 
    """ 
    Select regions of interest based on its dimensions.
    
    The input is a binary mask, and the output is an image with labelled pixels. 
 
    Parameters 
    ---------- 
    im : 2d ndarray of scalars 
        Spectrogram (or image) 
         
    min_roi, max_roi : scalars, optional, default : None 
        Define the minimum and the maximum area possible for an ROI. If None,  
        the minimum ROI area is 1 pixel and the maximum ROI area is the area of  
        the image 
        
    verbose : boolean, optional, default is False
        print messages
         
    display : boolean, optional, default is False 
        Display the signal if True 
         
    savefig : string, optional, default is None 
        Root filename (with full path) is required to save the figures. Postfix 
        is added to the root filename. 
         
    \*\*kwargs, optional. This parameter is used by plt.plot and savefig functions 
            
        - savefilename : str, optional, default :'_spectro_after_noise_subtraction.png' 
            Postfix of the figure filename 
         
        - figsize : tuple of integers, optional, default: (4,10) 
            width, height in inches.   
         
        - title : string, optional, default : 'Spectrogram' 
            title of the figure 
         
        - xlabel : string, optional, default : 'Time [s]' 
            label of the horizontal axis 
         
        - ylabel : string, optional, default : 'Amplitude [AU]' 
            label of the vertical axis 
         
        - cmap : string or Colormap object, optional, default is 'gray' 
            See https://matplotlib.org/examples/color/colormaps_reference.html 
            in order to get all the  existing colormaps 
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic',  
            'viridis'... 
         
        - vmin, vmax : scalar, optional, default: None 
            `vmin` and `vmax` are used in conjunction with norm to normalize 
            luminance data.  Note if you pass a `norm` instance, your 
            settings for `vmin` and `vmax` will be ignored. 
         
        - extent : scalars (left, right, bottom, top), optional, default: None 
            The location, in data-coordinates, of the lower-left and 
            upper-right corners. If `None`, the image is positioned such that 
            the pixel centers fall on zero-based (row, column) indices. 
         
        - dpi : integer, optional, default is 96 
            Dot per inch.  
            For printed version, choose high dpi (i.e. dpi=300) => slow 
            For screen version, choose low dpi (i.e. dpi=96) => fast 
         
        - format : string, optional, default is 'png' 
            Format to save the figure 
             
        ... and more, see matplotlib    
 
    Returns 
    ------- 
    im_rois: 2d ndarray 
        image with labels as values 
             
    rois: pandas DataFrame 
        Regions of interest with future descriptors will be computed. 
        Array have column names: ``labelID``, ``label``, ``min_y``, ``min_x``,
        ``max_y``, ``max_x``,
        Use the function ``maad.util.format_features`` before using 
        centroid_features to format of the ``rois`` DataFrame 
        correctly.
        
    Examples 
    -------- 
    
    Load audio recording and convert it into spectrogram
    
    >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
    >>> Sxx,tn,fn,ext = maad.sound.spectrogram (s, fs, fcrop=(0,20000), display=True)   
    
    Convert linear spectrogram into dB
    
    >>> Sxx_dB = maad.util.power2dB(Sxx) +96
    
    Smooth the spectrogram
    
    >>> Sxx_dB_blurred = maad.rois.smooth(Sxx_dB)
    
    Detection of the acoustic signature => creation of a mask
    
    >>> im_bin = maad.rois.create_mask(Sxx_dB_blurred, bin_std=1.5, bin_per=0.5, mode='relative')
    
    Select rois from the mask
    
    >>> im_rois, df_rois = maad.rois.select_rois(im_bin, display=True)
    
    We can observe that we detect as ROI background noise and also that we merge
    all together several ROIs. A solution consits in subtracting the background
    noise before finding ROIs.
    
    >>> Sxx_noNoise = maad.rois.median_equalizer(Sxx)
    
    Convert linear spectrogram into dB
    
    >>> Sxx_noNoise_dB = maad.util.power2dB(Sxx_noNoise) 
    
    Smooth the spectrogram
    
    >>> Sxx_noNoise_dB_blurred = maad.rois.smooth(Sxx_noNoise_dB)
    
    Detection of the acoustic signature => creation of a mask
    
    >>> im_bin2 = maad.rois.create_mask(Sxx_noNoise_dB_blurred, bin_std=6, bin_per=0.5, mode='relative') 
    
    Select rois from the mask
    
    >>> im_rois2, df_rois2 = maad.rois.select_rois(im_bin2, display=True)
    
    """ 
 
    # test if max_roi and min_roi are defined 
    if max_roi is None:  
        # the maximum ROI is set to the aera of the image 
        max_roi=im_bin.shape[0]*im_bin.shape[1] 
         
    if min_roi is None: 
        # the min ROI area is set to 1 pixel 
        min_roi = 1 
    
    if verbose :
        print(72 * '_') 
        print('Automatic ROIs selection in progress...') 
        print ('**********************************************************') 
        print ('  Min ROI area %d pix² | Max ROI area %d pix²' % (min_roi, max_roi)) 
        print ('**********************************************************') 
 
    labels = measure.label(im_bin)    #Find connected components in binary image 
    rprops = measure.regionprops(labels) 
     
    rois_bbox = [] 
    rois_label = [] 
     
    for roi in rprops: 
         
        # select the rois  depending on their size 
        if (roi.area >= min_roi) & (roi.area <= max_roi): 
            # get the label 
            rois_label.append(roi.label) 
            # get rectangle coordonates            
            rois_bbox.append (roi.bbox)     
                 
    im_rois = np.isin(labels, rois_label)    # test if the indice is in the matrix of indices 
    im_rois = im_rois* labels 
     
    # create a list with labelID and labelName (None in this case) 
    rois_label = list(zip(rois_label,['unknown']*len(rois_label))) 
    
    # test if there is a roi
    if len(rois_label)>0 :
        # create a dataframe rois containing the coordonates and the label 
        rois = np.concatenate((np.asarray(rois_label), np.asarray(rois_bbox)), axis=1) 
        rois = pd.DataFrame(rois, columns = ['labelID', 'label', 'min_y','min_x','max_y', 'max_x']) 
        # force type to integer 
        rois = rois.astype({'label': str,'min_y':int,'min_x':int,'max_y':int, 'max_x':int}) 
        # compensate half-open interval of bbox from skimage 
        rois.max_y -= 1 
        rois.max_x -= 1 
        
    else :
        rois = []    
        rois = pd.DataFrame(rois, columns = ['labelID', 'label', 'min_y','min_x','max_y', 'max_x']) 
        rois = rois.astype({'label': str,'min_y':int,'min_x':int,'max_y':int, 'max_x':int}) 
     
    # Display 
    if display :  
        ylabel =kwargs.pop('ylabel','Frequency [Hz]') 
        xlabel =kwargs.pop('xlabel','Time [sec]')  
        title  =kwargs.pop('title','Selected ROIs')  
        extent=kwargs.pop('extent',None)
            
        if extent is not None : 
            xlabel = 'frequency [Hz]' 
            figsize=kwargs.pop('figsize', (4, 0.33*(extent[1]-extent[0])))
        else: 
            xlabel = 'pseudofrequency [points]'
            figsize=kwargs.pop('figsize',(4, 13)) 
         
        randcmap = rand_cmap(len(rois_label)) 
        cmap   =kwargs.pop('cmap',randcmap)  
         
        _, fig = plot2D (im_rois, extent=extent, figsize=figsize,title=title,  
                         ylabel = ylabel, xlabel = xlabel, 
                         cmap=cmap, **kwargs) 
        # SAVE FIGURE 
        if savefig is not None :  
            dpi   =kwargs.pop('dpi',96) 
            format=kwargs.pop('format','png')  
            filename=kwargs.pop('filename','_spectro_selectrois')                 
            filename = savefig+filename+'.'+format 
            fig.savefig(filename, bbox_inches='tight', dpi=dpi, format=format, 
                        **kwargs)  
            
    return im_rois, rois 
 
#**************************************************************************** 
#*************                   overlay_rois                     *********** 
#**************************************************************************** 
def overlay_rois (im_ref, rois, savefig=None, **kwargs): 
    """ 
    Overlay bounding box on the original spectrogram 
     
    Parameters 
    ---------- 
    im_ref : 2d ndarray of scalars 
        Spectrogram (or image) 
 
    rois_bbox : list of tuple (min_y,min_x,max_y,max_x) 
        Contains the bounding box of each ROI 
 
         
    savefig : string, optional, default is None 
        Root filename (with full path) is required to save the figures. Postfix 
        is added to the root filename. 
         
    \*\*kwargs, optional. This parameter is used by plt.plot and savefig functions 
            
        - savefilename : str, optional, default :'_spectro_overlayrois.png' 
            Postfix of the figure filename 
         
        - figsize : tuple of integers, optional, default: (4,10) 
            width, height in inches.   
         
        - title : string, optional, default : 'Spectrogram' 
            title of the figure 
         
        - xlabel : string, optional, default : 'Time [s]' 
            label of the horizontal axis 
         
        - ylabel : string, optional, default : 'Amplitude [AU]' 
            label of the vertical axis 
         
        - cmap : string or Colormap object, optional, default is 'gray' 
            See https://matplotlib.org/examples/color/colormaps_reference.html 
            in order to get all the  existing colormaps 
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic',  
            'viridis'... 
         
        - vmin, vmax : scalar, optional, default: None 
            `vmin` and `vmax` are used in conjunction with norm to normalize 
            luminance data.  Note if you pass a `norm` instance, your 
            settings for `vmin` and `vmax` will be ignored. 
         
        - dpi : integer, optional, default is 96 
            Dot per inch.  
            For printed version, choose high dpi (i.e. dpi=300) => slow 
            For screen version, choose low dpi (i.e. dpi=96) => fast 
         
        - format : string, optional, default is 'png' 
            Format to save the figure  
         
        ... and more, see matplotlib  
 
    Returns 
    ------- 
    ax : axis object (see matplotlib) 
         
    fig : figure object (see matplotlib) 
    
    Examples 
    -------- 
    
    Load audio recording and convert it into spectrogram
    
    >>> s, fs = maad.sound.load('../data/cold_forest_daylight.wav')
    >>> Sxx,tn,fn,ext = maad.sound.spectrogram (s, fs, fcrop=(0,10000))   

    Subtract the background noise before finding ROIs.
    
    >>> Sxx_noNoise = maad.rois.median_equalizer(Sxx)
    
    Convert linear spectrogram into dB
    
    >>> Sxx_noNoise_dB = maad.util.power2dB(Sxx_noNoise) 
    
    Smooth the spectrogram
    
    >>> Sxx_noNoise_dB_blurred = maad.rois.smooth(Sxx_noNoise_dB)
    
    Detection of the acoustic signature => creation of a mask
    
    >>> im_bin = maad.rois.create_mask(Sxx_noNoise_dB_blurred, bin_std=6, bin_per=0.5, mode='relative') 
    
    Select rois from the mask and display bounding box over the spectrogram without noise
    
    >>> im_rois, df_rois = maad.rois.select_rois(im_bin)  
    >>> maad.rois.overlay_rois (Sxx_noNoise_dB, df_rois, extent=ext,vmin=np.median(Sxx_noNoise_dB), vmax=np.median(Sxx_noNoise_dB)+60) 
        
    """        
     
    # Check format of the input data 
    if type(rois) is not pd.core.frame.DataFrame: 
        raise TypeError('Rois must be of type pandas DataFrame.')    
    if not(('min_y' and 'min_x' and 'max_y' and 'max_x') in rois): 
        raise TypeError('Array must be a Pandas DataFrame with column names: min_y, min_x, max_y, max_x. Check example in documentation.')     
     
    ylabel =kwargs.pop('ylabel','Frequency [Hz]') 
    xlabel =kwargs.pop('xlabel','Time [sec]')  
    title  =kwargs.pop('title','ROIs Overlay') 
    cmap   =kwargs.pop('cmap','gray')  
    vmin=kwargs.pop('vmin',np.percentile(im_ref,0.05))  
    vmax=kwargs.pop('vmax',np.percentile(im_ref,0.95))   
    ax =kwargs.pop('ax',None)  
    fig=kwargs.pop('fig',None)  
    extent=kwargs.pop('extent',None)
        
    if extent is not None : 
        xlabel = 'frequency [Hz]' 
        figsize=kwargs.pop('figsize', (4, 0.33*(extent[1]-extent[0])))
    else: 
        xlabel = 'pseudofrequency [points]'
        figsize=kwargs.pop('figsize',(4, 13)) 
    
         
    if (ax is None) and (fig is None): 
        ax, fig = plot2D (im_ref,extent=extent,now=False, figsize=figsize,title=title,  
                         ylabel=ylabel,xlabel=xlabel,vmin=vmin,vmax=vmax,  
                         cmap=cmap, **kwargs) 
 
    # Convert pixels into time and frequency values 
    y_len, x_len = im_ref.shape 
    xmin, xmax = ax.get_xlim() 
    ymin, ymax = ax.get_ylim()  
    x_scaling = (xmax-xmin) / x_len 
    y_scaling = (ymax-ymin) / y_len 
     
    # test if rois has a label column     
    if 'label' in rois : 
        # select the label column 
        rois_label = rois.label.values 
        uniqueLabels = np.unique(np.array(rois_label)) 
    else:  
        uniqueLabels = [] 
                 
    # if only one label or no label in rois 
    if (len(uniqueLabels)<=1) : 
        for index, row in rois.iterrows(): 
            y0 = row['min_y'] 
            x0 = row['min_x'] 
            y1 = row['max_y'] 
            x1 = row['max_x'] 
            rect = mpatches.Rectangle((x0*x_scaling+xmin, y0*y_scaling+ymin),  
                                      (x1-x0)*x_scaling,  
                                      (y1-y0)*y_scaling, 
                                      fill=False, edgecolor='yellow', linewidth=1)   
            # draw the rectangle 
            ax.add_patch(rect)                 
    else : 
        # Colormap 
        color = rand_cmap(len(uniqueLabels)+1,first_color_black=False)  
        cc = 0 
        for index, row in rois.iterrows(): 
            cc = cc+1 
            y0 = row['min_y'] 
            x0 = row['min_x'] 
            y1 = row['max_y'] 
            x1 = row['max_x'] 
            for index, name in enumerate(uniqueLabels): 
                if row['label'] in name: ii = index 
            rect = mpatches.Rectangle((x0*x_scaling+xmin, y0*y_scaling+ymin),  
                                      (x1-x0)*x_scaling,  
                                      (y1-y0)*y_scaling, 
                                      fill=False, edgecolor=color(ii), linewidth=1)   
            # draw the rectangle 
            ax.add_patch(rect) 
 
    fig.canvas.draw() 
     
    # SAVE FIGURE 
    if savefig is not None :  
        dpi   =kwargs.pop('dpi',96) 
        format=kwargs.pop('format','png')  
        filename=kwargs.pop('filename','_spectro_overlayrois')                 
        filename = savefig+filename+'.'+format 
        fig.savefig(filename, bbox_inches='tight', dpi=dpi, format=format, 
                    **kwargs)  
        
    return ax, fig 
 
#**************************************************************************** 
#*************                   rois_to_imblobs                  *********** 
#**************************************************************************** 
 
def rois_to_imblobs(im_blobs, rois): 
    """  
    Add 1 corresponding to rois to im_blobs which is an empty matrix 
 
    Parameters 
    ---------- 
    im_blobs : ndarray 
        matrix full of zeros with the size to the image where the rois come from 
     
    rois : DataFrame 
        rois must have the columns names:((min_y, min_x, max_y, max_x) which 
        correspond to the bounding box coordinates 
     
    Returns 
    ------- 
    im_blobs : ndarray 
        matrix with 1 corresponding to the rois and 0 elsewhere 
 
    """ 
    # Check format of the input data 
    if type(rois) is not pd.core.frame.DataFrame : 
        raise TypeError('Rois must be of type pandas DataFrame')   
         
    if not(('min_y' and 'min_x' and 'max_y' and 'max_x')  in rois)  : 
            raise TypeError('Array must be a Pandas DataFrame with column names:((min_y, min_x, max_y, max_x). Check example in documentation.')   
     
    # select the columns 
    rois_bbox = rois[['min_y', 'min_x', 'max_y', 'max_x']] 
    # roi to image blob 
    for min_y, min_x, max_y, max_x in rois_bbox.values: 
        im_blobs[int(min_y):int(max_y+1), int(min_x):int(max_x+1)] = 1 
     
    im_blobs = im_blobs.astype(int) 
     
    return im_blobs 

#**************************************************************************** 
#*************                   sharpness                     *********** 
#**************************************************************************** 

def sharpness (im) :
    """ 
    Compute the sharpness of an image (or spectrogram)
     
    Parameters 
    ---------- 
    im : 2d ndarray of scalars 
        Spectrogram (or image) 
        
    Returns
    -------
    sharpness : scalar
        sharpness of the spectrogram (or image)
    """
    
    Gt = np.gradient(im, edge_order=1, axis=1)
    Gf = np.gradient(im, edge_order=1, axis=0)
    S = np.sqrt(Gt**2+ Gf**2)
    sharpness=sum(sum(S))/(Gt.shape[0]*Gt.shape[1])
    return sharpness   
 
 
 