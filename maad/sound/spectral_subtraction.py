#!/usr/bin/env python
""" 
Collection of functions to remove background noise from spectrogram using
spectral subtraction methods
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
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import morphology 
from skimage.morphology import reconstruction
from scipy import signal 
# min value
import sys
_MIN_ = sys.float_info.min

# Import internal modules 
from maad.util import (plot1d, plot2d, running_mean, 
                       get_unimode, mean_dB, power2dB)

#%%
# =============================================================================
# public functions
# =============================================================================
def remove_background (Sxx, gauss_win=50, gauss_std = 25, beta1=1, beta2=1,  
                      llambda=1, verbose = False, display = False, 
                      savefig=None, **kwargs): 
    """ 
    Remove background noise using spectral subtraction.
 
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

    Examples
    --------
    
    Load audio recording and convert it into spectrogram
    
    >>> s, fs = maad.sound.load('../data/guyana_tropical_forest.wav')
    >>> Sxx,tn,fn,ext = maad.sound.spectrogram (s, fs)   
    
    Convert linear spectrogram into dB and add 96dB (which is the maximum dB
    for 16 bits wav) in order to have positive values
    
    >>> Sxx_dB = maad.util.power2dB(Sxx) + 96
    
    Remove stationnary noise from the spectrogram in dB
    
    >>> Sxx_dB_noNoise, noise_profile, _ = maad.sound.remove_background(Sxx_dB)

    Plot both spectrograms
    
    >>> import matplotlib.pyplot as plt 
    >>> import numpy as np
    >>> fig, (ax1, ax2) = plt.subplots(2, 1)
    >>> maad.util.plot2d(Sxx_dB, ax=ax1, extent=ext, title='original', vmin=np.median(Sxx_dB), vmax=np.median(Sxx_dB)+40)
    >>> maad.util.plot2d(Sxx_dB_noNoise, ax=ax2, extent=ext, title='Without stationary noise', vmin=np.median(Sxx_dB_noNoise), vmax=np.median(Sxx_dB_noNoise)+40)
    >>> fig.set_size_inches(15,8)
    >>> fig.tight_layout()
       
    """   
    if verbose :
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
    SNR_est= Sxx - noise_spectro
    # to avoid dividing by 0
    SNR_est[SNR_est<=0] = 0
    noise_spectro[noise_spectro==0] = _MIN_
    # ratio
    SNR_est=(Sxx/noise_spectro)  
    # keep only positive values 
    SNR_est=SNR_est*(SNR_est>0) 
     
    # compute attenuation map 
    # if llambda, beta1 and beta 2 are equal to 1, it is (1 - noise_spectro) 
    an_lk=(1-llambda*((1./(SNR_est+1))**beta1))**beta2 
    an_lk=an_lk*(an_lk>0) # keep only positive values 
     
    if verbose : print('Remove the stochastic background noise...') 
     
    # Apply the attenuation map to the STFT coefficients 
    Sxx_out=an_lk*Sxx 
    
    # noise map BGNxx
    BGNxx = Sxx - Sxx_out
     
    # if nan in the image, convert nan into 0 
    np.nan_to_num(Sxx_out,0) 
    
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
            
        if extent is not None : 
            fn = np.arange(0, Nf)*(extent[3]-extent[2])/(Nf-1) + extent[2]  
            xlabel = 'frequency [Hz]' 
            figsize=kwargs.pop('figsize', (4, 0.33*(extent[1]-extent[0])))
        else: 
            fn = np.arange(Nf) 
            xlabel = 'pseudofrequency [points]'
            figsize=kwargs.pop('figsize',(4, 13))  
        
        _, fig = plot2d (Sxx_out, extent=extent, figsize=figsize,title=title,  
                         ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax, 
                         cmap=cmap, **kwargs) 
        
        fig2, (ax1, ax2) = plt.subplots(2, sharex=True)
        fig2.set_size_inches((5,4))
        ax1,_ = plot1d(fn, mean_profile, ax=ax1, legend='Original profile',
                       color = 'b',
                       xlabel = '', ylabel = 'Amplitude [dB]', figtitle='') 
        ax1,_ = plot1d(fn, np.mean(BGNxx, axis=1), ax =ax1, legend='Noise profile',
                       color = 'r',
                       xlabel = '', ylabel = 'Amplitude [dB]', figtitle='')
        ax2,_ = plot1d(fn, np.mean(Sxx_out,axis=1), ax=ax2, color = 'k', 
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
 
#%%
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
    
    Examples
    --------
    
    Load audio recording and convert it into spectrogram
    
    >>> s, fs = maad.sound.load('../data/guyana_tropical_forest.wav')
    >>> Sxx,tn,fn,ext = maad.sound.spectrogram (s, fs)   
    
    Convert linear spectrogram into dB
    
    >>> Sxx_dB = maad.util.power2dB(Sxx) +96
    
    Remove stationnary noise from the spectrogram 
    
    >>> Sxx_dB_noNoise,_,_ = maad.sound.remove_background_morpho(Sxx_dB, q=0.5)

    Plot both spectrograms
    
    >>> import matplotlib.pyplot as plt 
    >>> import numpy as np
    >>> fig, (ax1, ax2) = plt.subplots(2, 1)
    >>> maad.util.plot2d(Sxx_dB, ax=ax1, extent=ext, title='original', vmin=np.median(Sxx_dB), vmax=np.median(Sxx_dB)+40)
    >>> maad.util.plot2d(Sxx_dB_noNoise, ax=ax2, extent=ext, title='Without stationary noise',vmin=np.median(Sxx_dB_noNoise), vmax=np.median(Sxx_dB_noNoise)+40)
    >>> fig.set_size_inches(15,8)
    >>> fig.tight_layout()     
    
    Load audio recording and convert it into spectrogram
    
    >>> s, fs = maad.sound.load('../data/guyana_tropical_forest.wav')
    >>> Sxx,tn,fn,ext = maad.sound.spectrogram (s, fs, tcrop=(0,20))   
    >>> Sxx_dB = maad.util.power2dB(Sxx) +96
    
    Remove stationnary noise from the spectrogram with different q
    
    >>> Sxx_dB_noNoise_q25,_,_ = maad.sound.remove_background_morpho(Sxx_dB, q=0.25)
    >>> Sxx_dB_noNoise_q50,_,_ = maad.sound.remove_background_morpho(Sxx_dB, q=0.5)
    >>> Sxx_dB_noNoise_q75,_,_ = maad.sound.remove_background_morpho(Sxx_dB, q=0.75)
    
    Plot 3 spectrograms
    
    >>> import matplotlib.pyplot as plt 
    >>> fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    >>> maad.util.plot2d(Sxx_dB_noNoise_q25, ax=ax1, extent=ext, title='Without stationary noise (q=0.25)',vmin=np.median(Sxx_dB_noNoise_q25), vmax=np.median(Sxx_dB_noNoise_q25)+40)
    >>> maad.util.plot2d(Sxx_dB_noNoise_q50, ax=ax2, extent=ext, title='Without stationary noise (q=0.50)',vmin=np.median(Sxx_dB_noNoise_q50), vmax=np.median(Sxx_dB_noNoise_q50)+40)
    >>> maad.util.plot2d(Sxx_dB_noNoise_q75, ax=ax3, extent=ext, title='Without stationary noise (q=0.75)',vmin=np.median(Sxx_dB_noNoise_q75), vmax=np.median(Sxx_dB_noNoise_q75)+40)
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
        
        _, fig = plot2d (BGNxx, extent=extent, figsize=figsize,title='Noise map',  
                         ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax, 
                         cmap=cmap, **kwargs) 
        
        _, fig = plot2d (Sxx_out, extent=extent, figsize=figsize,title=title,  
                         ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax, 
                         cmap=cmap, **kwargs) 
        
        fig2, (ax1, ax2) = plt.subplots(2, sharex=True)
        fig2.set_size_inches((5,4))
        ax1,_ = plot1d(fn, np.mean(Sxx,axis=1), ax=ax1, legend='Original profile', 
                       color = 'b',
                       xlabel = '', ylabel = 'Amplitude [dB]', figtitle='') 
        ax1,_ = plot1d(fn, np.mean(BGNxx,1), ax =ax1, legend='Noise profile', 
                       color = 'r',
                       xlabel = '', ylabel = 'Amplitude [dB]', figtitle='')
        ax2,_ = plot1d(fn, np.mean(Sxx_out,axis=1), ax=ax2, color = 'k', 
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
 
#%%
def remove_background_along_axis (Sxx, mode ='median', axis=1, N=25, N_bins=50, 
                                  display=False, savefig=None, **kwargs): 
    """ 
    Get the noisy profile along the defined axis and remove this profile from
    the spectrogram.
    
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
        
    Returns 
    ------- 
    Sxx_out : 2d ndarray of scalar 
        Spectrogram after denoising 
        
    noise_profile : 1d ndarray of scalar
        Noise profile


    References
    ----------
    
    .. [1] Towsey, M., 2013b. Noise Removal from Wave-forms and Spectrograms Derived from
    Natural Recordings of the Environment. Queensland University of Technology,
    Brisbane
                       
        
    Examples
    --------
    
    Load audio recording and convert it into spectrogram
    
    >>> s, fs = maad.sound.load('../data/guyana_tropical_forest.wav')
    >>> Sxx,tn,fn,ext = maad.sound.spectrogram (s, fs)   
    
    Convert linear spectrogram into dB
    
    >>> Sxx_dB = maad.util.power2dB(Sxx) +96
    
    Remove stationnary noise from the spectrogram with modes 'ale', 'median', and 'mean'.
    
    >>> Sxx_dB_noNoise_ale,_ = maad.sound.remove_background_along_axis(Sxx_dB, mode='ale')    
    >>> Sxx_dB_noNoise_med,_ = maad.sound.remove_background_along_axis(Sxx_dB, mode='median')
    >>> Sxx_dB_noNoise_mean,_ = maad.sound.remove_background_along_axis(Sxx_dB, mode='mean')

    Plot spectrograms
    
    >>> import matplotlib.pyplot as plt 
    >>> import numpy as np
    >>> fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
    >>> maad.util.plot2d(Sxx_dB, ax=ax1, extent=ext, title='original', vmin=np.median(Sxx_dB), vmax=np.median(Sxx_dB)+40)
    >>> maad.util.plot2d(Sxx_dB_noNoise_ale, ax=ax2, extent=ext, title='Without stationary noise (mode = ''ale'')',vmin=np.median(Sxx_dB_noNoise_ale), vmax=np.median(Sxx_dB_noNoise_ale)+40)
    >>> maad.util.plot2d(Sxx_dB_noNoise_med, ax=ax3, extent=ext, title='Without stationary noise (mode = ''med'')',vmin=np.median(Sxx_dB_noNoise_med), vmax=np.median(Sxx_dB_noNoise_med)+40)
    >>> maad.util.plot2d(Sxx_dB_noNoise_mean, ax=ax4, extent=ext, title='Without stationary noise (mode = ''mean'')',vmin=np.median(Sxx_dB_noNoise_mean), vmax=np.median(Sxx_dB_noNoise_mean)+40)
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
            
        _, fig1 = plot2d (Sxx_out, extent=extent, figsize=figsize,title=title,  
                         ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax, 
                         cmap=cmap, **kwargs) 
        
        fig2, (ax1, ax2) = plt.subplots(2, sharex=True)
        fig2.set_size_inches((5,4))
        ax1,_ = plot1d(fn, mean_dB(Sxx,axis=axis), ax=ax1, legend='Original profile',
                       color = 'b',
                       xlabel = '', ylabel = 'Amplitude [dB]', figtitle='') 
        ax1,_ = plot1d(fn, noise_profile, ax =ax1, legend='Noise profile', 
                       color = 'r',
                       xlabel = '', ylabel = 'Amplitude [dB]', figtitle='')
        ax2,_ = plot1d(fn, mean_dB(Sxx_out,axis=axis), ax=ax2, color = 'k', 
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

#%%
def median_equalizer (Sxx, display=False, savefig=None, **kwargs): 
    """ 
    Remove background noise in spectrogram using median equalizer.
     
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
       
    Examples
    --------
    
    Load audio recording and convert it into spectrogram
    
    >>> s, fs = maad.sound.load('../data/guyana_tropical_forest.wav')
    >>> Sxx,tn,fn,ext = maad.sound.spectrogram (s, fs)   
    
    Convert linear spectrogram into dB
    
    >>> Sxx_dB = maad.util.power2dB(Sxx) +96
    
    Remove stationnary noise from the spectrogram 
    
    >>> Sxx_noNoise = maad.sound.median_equalizer(Sxx)
    >>> Sxx_dB_noNoise =  maad.util.power2dB(Sxx_noNoise) 

    Plot both spectrograms
    
    >>> import matplotlib.pyplot as plt 
    >>> import numpy as np
    >>> fig, (ax1, ax2) = plt.subplots(2, 1)
    >>> maad.util.plot2d(Sxx_dB, ax=ax1, extent=ext, title='original', vmin=np.median(Sxx_dB), vmax=np.median(Sxx_dB)+40)
    >>> maad.util.plot2d(Sxx_dB_noNoise, ax=ax2, extent=ext, title='Without stationary noise',vmin=np.median(Sxx_dB_noNoise), vmax=np.median(Sxx_dB_noNoise)+40)
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
         
        _, fig = plot2d (Sxx_out_dB, extent=extent, figsize=figsize,title=title,  
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

