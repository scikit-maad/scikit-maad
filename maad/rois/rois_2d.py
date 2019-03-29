#!/usr/bin/env python
""" Multiresolution Analysis of Acoustic Diversity
    functions for processing ROIS """
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
from math import ceil
from scipy import signal
from scipy.ndimage import morphology
from scipy.stats import iqr
from skimage import measure, filters
from skimage.io import imread

import matplotlib.patches as mpatches
from ..util import plot1D, plot2D, linear_scale,read_audacity_annot,rand_cmap

"""
====== TO DO
"""


def select_bandwidth():
    return
"""
====== 
"""

"""****************************************************************************
************* Load an image and convert it in gray level if needed  ***********
****************************************************************************"""
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
        
    **kwargs, optional. This parameter is used by plt.plot 
        figsize : tuple of integers, optional, default: (4,10)
            width, height in inches.  
        title : string, optional, default : 'Spectrogram'
            title of the figure
        xlabel : string, optional, default : 'Time [s]'
            label of the horizontal axis
        ylabel : string, optional, default : 'Amplitude [AU]'
            label of the vertical axis
        cmap : string or Colormap object, optional, default is 'gray'
            See https://matplotlib.org/examples/color/colormaps_reference.html
            in order to get all the  existing colormaps
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic', 
                      'viridis'...
        vmin, vmax : scalar, optional, default: None
            `vmin` and `vmax` are used in conjunction with norm to normalize
            luminance data.  Note if you pass a `norm` instance, your
            settings for `vmin` and `vmax` will be ignored.
        ext : scalars (left, right, bottom, top), optional, default: None
            The location, in data-coordinates, of the lower-left and
            upper-right corners. If `None`, the image is positioned such that
            the pixel centers fall on zero-based (row, column) indices.
        dpi : integer, optional, default is 96
            Dot per inch. 
            For printed version, choose high dpi (i.e. dpi=300) => slow
            For screen version, choose low dpi (i.e. dpi=96) => fast
        format : string, optional, default is 'png'
            Format to save the figure 
        
        ... and more, see matplotlib 
        
    Returns
    -------
    im : ndarray
        The different color bands/channels are stored in the
        third dimension, such that a gray-image is MxN, an
        RGB-image MxNx3 and an RGBA-image MxNx4.
        
    ext : list of scalars [left, right, bottom, top], optional, default: None
        The location, in data-coordinates, of the lower-left and
        upper-right corners. If `None`, the image is positioned such that
        the pixel centers fall on zero-based (row, column) indices.  
        
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
    ext = [0, duration, 0, fs/2]
    
    # flip the image vertically
    if flipud: im = np.flip(im, 0)
    
    # Display
    if display : 
        ylabel =kwargs.pop('ylabel','Frequency [Hz]')
        xlabel =kwargs.pop('xlabel','Time [sec]') 
        title  =kwargs.pop('title','loaded spectrogram')
        cmap   =kwargs.pop('cmap','gray') 
        figsize=kwargs.pop('figsize',(4, 13)) 
        vmin=kwargs.pop('vmin',0) 
        vmax=kwargs.pop('vmax',1) 
        
        _, fig = plot2D (im, extent=ext, figsize=figsize,title=title, 
                         ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax,
                         cmap=cmap, **kwargs)
    
    return im, ext, dt, df

"""****************************************************************************
*************               noise_subtraction                       ***********
****************************************************************************"""
def remove_background(im, ext, gauss_win=50, gauss_std = 25, beta1=1, beta2=1, 
                      llambda=1, display = False, savefig=None, **kwargs):
    """
    Remove the background noise using spectral subtraction

    Based on the spectrum of the A posteriori noise profile.  
    It computes an atenuation map in the time-frequency domain. 
    See [1] or [2] for more detail about the algorithm.

    References:

    [1] Steven F. Boll, "Suppression of Acoustic Noise in Speech Using Spectral
    Subtraction", IEEE Transactions on Signal Processing, 27(2),pp 113-120,
    1979

    [2] Y. Ephraim and D. Malah, Speech enhancement using a minimum mean square 
    error short-time spectral amplitude estimator, IEEE. Transactions in
    Acoust., Speech, Signal Process., vol. 32, no. 6, pp. 11091121, Dec. 1984.

    Parameters
    ----------
    im : 2d ndarray of scalars
        Spectrogram 
        
    ext : list of scalars [left, right, bottom, top], optional, default: None
        The location, in data-coordinates, of the lower-left and
        upper-right corners. If `None`, the image is positioned such that
        the pixel centers fall on zero-based (row, column) indices.  
        
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
        
    display : boolean, optional, default is False
        Display the signal if True
        
    savefig : string, optional, default is None
        Root filename (with full path) is required to save the figures. Postfix
        is added to the root filename.
        
    **kwargs, optional. This parameter is used by plt.plot and savefig functions
        ****************************************************    
        savefilename : str, optional, default :'_spectro_after_noise_subtraction.png'
            Postfix of the figure filename
        **************************************************** 
        figsize : tuple of integers, optional, default: (4,10)
            width, height in inches.  
        title : string, optional, default : 'Spectrogram'
            title of the figure
        xlabel : string, optional, default : 'Time [s]'
            label of the horizontal axis
        ylabel : string, optional, default : 'Amplitude [AU]'
            label of the vertical axis
        cmap : string or Colormap object, optional, default is 'gray'
            See https://matplotlib.org/examples/color/colormaps_reference.html
            in order to get all the  existing colormaps
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic', 
                      'viridis'...
        vmin, vmax : scalar, optional, default: None
            `vmin` and `vmax` are used in conjunction with norm to normalize
            luminance data.  Note if you pass a `norm` instance, your
            settings for `vmin` and `vmax` will be ignored.
        ext : scalars (left, right, bottom, top), optional, default: None
            The location, in data-coordinates, of the lower-left and
            upper-right corners. If `None`, the image is positioned such that
            the pixel centers fall on zero-based (row, column) indices.
        dpi : integer, optional, default is 96
            Dot per inch. 
            For printed version, choose high dpi (i.e. dpi=300) => slow
            For screen version, choose low dpi (i.e. dpi=96) => fast
        format : string, optional, default is 'png'
            Format to save the figure
            
        ... and more, see matplotlib    
                      
    Returns
    -------
    Sxx_out : 2d ndarray of scalar
        Spectrogram after denoising           
    """  
    
    print(72 * '_' )
    print('Determine the profile of the stochastic background noise...')
    
    Nf, Nw = im.shape
          
    # noise spectrum extraction
    absS=np.abs(im)**2;
    mean_profile=np.mean(absS,1) # average spectrum (assumed to be ergodic)
    if display == True:
        if ext is not None :
            #fn = np.arange(ext[2], ext[3]+(ext[3]-ext[2])/(Nf-1), (ext[3]-ext[2])/(Nf-1))
            fn = np.arange(0, Nf)*(ext[3]-ext[2])/(Nf-1) + ext[2] 
            xlabel = 'frequency [Hz]'
        else:
            fn = np.arange(Nf)
            xlabel = 'pseudofrequency [points]'
        ax1,_ = plot1D(fn, mean_profile, now=False, figtitle = 'Noise profile',
                       xlabel = xlabel, ylabel = 'Amplitude [AU]',
                       linecolor = 'r')  
    
    # White Top Hat (to remove non uniform background) = i - opening(i)
    selem = signal.gaussian(gauss_win, gauss_std)
    noise_profile = morphology.grey_opening(mean_profile, structure=selem)
    if display == True: 
        if ext is not None :
            #fn = np.arange(ext[2], ext[3]+(ext[3]-ext[2])/(Nf-1), (ext[3]-ext[2])/(Nf-1))
            fn = np.arange(0, Nf)*(ext[3]-ext[2])/(Nf-1) + ext[2] 
            xlabel = 'frequency [Hz]'
        else:
            fn = np.arange(Nf)
            xlabel = 'pseudofrequency [points]'
        plot1D(fn, noise_profile, ax =ax1, now=False,figtitle = 'Noise profile', 
               xlabel = xlabel, ylabel = 'Amplitude [AU]', linecolor = 'k')  

    # Remove the artefact at the end of the spectrum (2 highest frequencies)
    noise_profile[-2:] = mean_profile [-2:]
    noise_profile[:2] = mean_profile [:2]
    
    # Create a matrix with the noise profile
    noise_spectro=np.kron(np.ones((Nw,1)),noise_profile)
    noise_spectro=noise_spectro.transpose()
        
    # snr estimate a posteriori
    SNR_est=(absS/noise_spectro) -1  
    SNR_est=SNR_est*(SNR_est>0) # keep only positive values
    
    # compute attenuation map
    # if llambda, beta1 and beta 2 are equal to 1, it is (1 - noise_spectro)
    an_lk=(1-llambda*((1./(SNR_est+1))**beta1))**beta2
    an_lk=an_lk*(an_lk>0) # keep only positive values
    
    
    print('Remove the stochastic background noise...')
    
    # Apply the attenuation map to the STFT coefficients
    im_out=an_lk*im
    
    # if nan in the image, convert nan into 0
    np.nan_to_num(im_out,0)
                
    # Display
    if display : 
        ylabel =kwargs.pop('ylabel','Frequency [Hz]')
        xlabel =kwargs.pop('xlabel','Time [sec]') 
        title  =kwargs.pop('title','Spectrogram without stationnary noise')
        cmap   =kwargs.pop('cmap','gray') 
        figsize=kwargs.pop('figsize',(4, 13)) 
        vmin=kwargs.pop('vmin',0) 
        vmax=kwargs.pop('vmax',1) 
        
        _, fig = plot2D (im_out, extent=ext, figsize=figsize,title=title, 
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
            
    return im_out  

def median_equalizer (x):
    """
    Median equalizer : remove background noise in a spectrogram
    
    Parameters
    ----------
    x : 2D numpy array 
        Original spectrogram (or image)
    Returns
    -------
    y : 1D numpy array 
        Ouput spectrogram (or image) without background noise
    
    References:
    ----------
    This function has been proposed first by Carol BEDOYA <carol.bedoya@pg.canterbury.ac.nz>
    Adapted by S. Haupert Oct 9, 2018 for Python
    """ 
    
    y = (((x.transpose()-np.median(x.transpose(),axis=0)))/(np.median(x.transpose())-np.min(x.transpose(),axis=0))).transpose()

    return y


"""****************************************************************************
*************                      smooth                            ***********
****************************************************************************"""
def smooth (im, ext, std=1, display = False, savefig=None, **kwargs):
    """
    Smooth (i.e. blurr) the image with a gaussian filter
    
    Parameters
    ----------
    im : 2d ndarray of scalars
        Spectrogram (or image)
        
    ext : list of scalars [left, right, bottom, top], optional, default: None
        The location, in data-coordinates, of the lower-left and
        upper-right corners. If `None`, the image is positioned such that
        the pixel centers fall on zero-based (row, column) indices.  
    
    std : scalar, optional, default is 1
        Standard deviation of the gaussian kernel used to smooth the image
        The larger is the number, the smoother will be the image and the longer
        it takes. Standard values should fall between 0.5 to 3
        
    display : boolean, optional, default is False
        Display the signal if True
        
    savefig : string, optional, default is None
        Root filename (with full path) is required to save the figures. Postfix
        is added to the root filename.
        
    **kwargs, optional. This parameter is used by plt.plot and savefig functions
        ****************************************************    
        savefilename : str, optional, default :'_spectro_after_noise_subtraction.png'
            Postfix of the figure filename
        **************************************************** 
        figsize : tuple of integers, optional, default: (4,10)
            width, height in inches.  
        title : string, optional, default : 'Spectrogram'
            title of the figure
        xlabel : string, optional, default : 'Time [s]'
            label of the horizontal axis
        ylabel : string, optional, default : 'Amplitude [AU]'
            label of the vertical axis
        cmap : string or Colormap object, optional, default is 'gray'
            See https://matplotlib.org/examples/color/colormaps_reference.html
            in order to get all the  existing colormaps
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic', 
                      'viridis'...
        vmin, vmax : scalar, optional, default: None
            `vmin` and `vmax` are used in conjunction with norm to normalize
            luminance data.  Note if you pass a `norm` instance, your
            settings for `vmin` and `vmax` will be ignored.
        ext : scalars (left, right, bottom, top), optional, default: None
            The location, in data-coordinates, of the lower-left and
            upper-right corners. If `None`, the image is positioned such that
            the pixel centers fall on zero-based (row, column) indices.
        dpi : integer, optional, default is 96
            Dot per inch. 
            For printed version, choose high dpi (i.e. dpi=300) => slow
            For screen version, choose low dpi (i.e. dpi=96) => fast
        format : string, optional, default is 'png'
            Format to save the figure
            
        ... and more, see matplotlib   
        
    Returns
    -------
        im_out: smothed or blurred image 
    """
    
    print(72 * '_')
    print('Smooth the image with a gaussian filter (std = %.1f)' %std)
    
    # use scikit image (faster than scipy)
    im_out = filters.gaussian(im,std)
    
    # Display
    if display : 
        ylabel =kwargs.pop('ylabel','Frequency [Hz]')
        xlabel =kwargs.pop('xlabel','Time [sec]') 
        title  =kwargs.pop('title','Blurred spectrogram')
        cmap   =kwargs.pop('cmap','gray') 
        figsize=kwargs.pop('figsize',(4, 13)) 
        vmin=kwargs.pop('vmin',0) 
        vmax=kwargs.pop('vmax',1) 
        
        _, fig = plot2D (im_out, extent=ext, figsize=figsize,title=title, 
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

"""****************************************************************************
*************                double_threshold                       ***********
****************************************************************************"""
def double_threshold_rel (im, ext, bin_std=5, bin_per=0.5, display=False, savefig=None,
                     **kwargs):
    """
    Binarize an image based on a double relative threshold. 
    The values used for the thresholding depends on the values found in the 
    image. => relative threshold
    
    Parameters
    ----------
    im : 2d ndarray of scalars
        Spectrogram (or image)
        
    ext : list of scalars [left, right, bottom, top], optional, default: None
        The location, in data-coordinates, of the lower-left and
        upper-right corners. If `None`, the image is positioned such that
        the pixel centers fall on zero-based (row, column) indices.  

    bin_std : scalar, optional, default is 5
        Set the first threshold. This threshold is not an absolute value but
        depends on values that are similar to mean and std value of the image. 
        threshold1 = "mean + "std" * bin_std   
        Value higher than threshold1 are set to 1, they are the seeds for 
        the second step. The others are set to 0. 
        
    bin_per: scalar, optional, defautl is 0.5
        Set how much the second threshold is lower than the first
        threshold value. From 0 to 1. ex: 0.1 = 10 %.
        threshold2 = threshold1 (1-bin_per)  
        Value higher than threshold2 and connected (directly or not) to the 
        seeds are set to 1, the other remains 0

    display : boolean, optional, default is False
        Display the signal if True
        
    savefig : string, optional, default is None
        Root filename (with full path) is required to save the figures. Postfix
        is added to the root filename.
        
    **kwargs, optional. This parameter is used by plt.plot and savefig functions
        ****************************************************    
        savefilename : str, optional, default :'_spectro_after_noise_subtraction.png'
            Postfix of the figure filename
        **************************************************** 
        figsize : tuple of integers, optional, default: (4,10)
            width, height in inches.  
        title : string, optional, default : 'Spectrogram'
            title of the figure
        xlabel : string, optional, default : 'Time [s]'
            label of the horizontal axis
        ylabel : string, optional, default : 'Amplitude [AU]'
            label of the vertical axis
        cmap : string or Colormap object, optional, default is 'gray'
            See https://matplotlib.org/examples/color/colormaps_reference.html
            in order to get all the  existing colormaps
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic', 
                      'viridis'...
        vmin, vmax : scalar, optional, default: None
            `vmin` and `vmax` are used in conjunction with norm to normalize
            luminance data.  Note if you pass a `norm` instance, your
            settings for `vmin` and `vmax` will be ignored.
        ext : scalars (left, right, bottom, top), optional, default: None
            The location, in data-coordinates, of the lower-left and
            upper-right corners. If `None`, the image is positioned such that
            the pixel centers fall on zero-based (row, column) indices.
        dpi : integer, optional, default is 96
            Dot per inch. 
            For printed version, choose high dpi (i.e. dpi=300) => slow
            For screen version, choose low dpi (i.e. dpi=96) => fast
        format : string, optional, default is 'png'
            Format to save the figure
            
        ... and more, see matplotlib   
    
    Returns
    -------
        im_out: binary image 
    
    References :
    ------------
            from MATLAB: Threshold estimation (from Oliveira et al, 2015)
        Adapted by S. Haupert Dec 12, 2017
    """
    
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
        figsize=kwargs.pop('figsize',(4, 13)) 
        vmin=kwargs.pop('vmin',0) 
        vmax=kwargs.pop('vmax',1) 
        
        _, fig = plot2D (im_out, extent=ext, figsize=figsize,title=title, 
                         ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax,
                         cmap=cmap, **kwargs)
        # SAVE FIGURE
        if savefig is not None : 
            dpi   =kwargs.pop('dpi',96)
            format=kwargs.pop('format','png')
            filename=kwargs.pop('filename','_spectro_binary')             
            filename = savefig+filename+'.'+format
            print('\n''save figure : %s' %filename)
            fig.savefig(filename, bbox_inches='tight', dpi=dpi, format=format,
                        **kwargs)   

    return im_out

"""****************************************************************************
*************                double_threshold                       ***********
****************************************************************************"""
def double_threshold_abs(im, ext, bin_h=0.7, bin_l=0.2, display=False, savefig=None,
                     **kwargs):
    """
    Binarize an image based on a double relative threshold. 
    The values used for the thresholding are independent of the values in the
    image => absolute threshold
    
    Parameters
    ----------
    im : 2d ndarray of scalars
        Spectrogram (or image)
        
    ext : list of scalars [left, right, bottom, top], optional, default: None
        The location, in data-coordinates, of the lower-left and
        upper-right corners. If `None`, the image is positioned such that
        the pixel centers fall on zero-based (row, column) indices.  

    bin_h : scalar, optional, default is 5
        Set the first threshold. Value higher than this value are set to 1, 
        the others are set to 0. They are the seeds for the second step
        
    bin_l: scalar, optional, defautl is 0.5
        Set the second threshold. Value higher than this value and connected
        to the seeds or to other pixels connected to the seeds are set to 1, 
        the other remains 0

    display : boolean, optional, default is False
        Display the signal if True
        
    savefig : string, optional, default is None
        Root filename (with full path) is required to save the figures. Postfix
        is added to the root filename.
        
    **kwargs, optional. This parameter is used by plt.plot and savefig functions
        ****************************************************    
        savefilename : str, optional, default :'_spectro_after_noise_subtraction.png'
            Postfix of the figure filename
        **************************************************** 
        figsize : tuple of integers, optional, default: (4,10)
            width, height in inches.  
        title : string, optional, default : 'Spectrogram'
            title of the figure
        xlabel : string, optional, default : 'Time [s]'
            label of the horizontal axis
        ylabel : string, optional, default : 'Amplitude [AU]'
            label of the vertical axis
        cmap : string or Colormap object, optional, default is 'gray'
            See https://matplotlib.org/examples/color/colormaps_reference.html
            in order to get all the  existing colormaps
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic', 
                      'viridis'...
        vmin, vmax : scalar, optional, default: None
            `vmin` and `vmax` are used in conjunction with norm to normalize
            luminance data.  Note if you pass a `norm` instance, your
            settings for `vmin` and `vmax` will be ignored.
        ext : scalars (left, right, bottom, top), optional, default: None
            The location, in data-coordinates, of the lower-left and
            upper-right corners. If `None`, the image is positioned such that
            the pixel centers fall on zero-based (row, column) indices.
        dpi : integer, optional, default is 96
            Dot per inch. 
            For printed version, choose high dpi (i.e. dpi=300) => slow
            For screen version, choose low dpi (i.e. dpi=96) => fast
        format : string, optional, default is 'png'
            Format to save the figure
            
        ... and more, see matplotlib   

    Returns
    -------
        im_out: binary image 
    
    References :
    ------------
        from MATLAB: Threshold estimation (from Oliveira et al, 2015)
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
                
    # Display
    if display : 
        ylabel =kwargs.pop('ylabel','Frequency [Hz]')
        xlabel =kwargs.pop('xlabel','Time [sec]') 
        title  =kwargs.pop('title','binary image => MASK')
        cmap   =kwargs.pop('cmap','gray') 
        figsize=kwargs.pop('figsize',(4, 13)) 
        vmin=kwargs.pop('vmin',0) 
        vmax=kwargs.pop('vmax',1) 
        
        _, fig = plot2D (im_out, extent=ext, figsize=figsize,title=title, 
                         ylabel = ylabel, xlabel = xlabel,vmin=vmin, vmax=vmax,
                         cmap=cmap, **kwargs)
        # SAVE FIGURE
        if savefig is not None : 
            dpi   =kwargs.pop('dpi',96)
            format=kwargs.pop('format','png')
            filename=kwargs.pop('filename','_spectro_binary')             
            filename = savefig+filename+'.'+format
            print('\n''save figure : %s' %filename)
            fig.savefig(filename, bbox_inches='tight', dpi=dpi, format=format,
                        **kwargs)   

    return im_out

"""****************************************************************************
*************                   create_mask wrapper                 ***********
****************************************************************************"""
def create_mask(im, ext, mode_bin = 'relative', display = False, savefig = None,
                **kwargs):
    """
    Binarize an image based on a double threshold. 
    
    Parameters
    ----------
    im : 2d ndarray of scalars
        Spectrogram (or image)
        
    ext : list of scalars [left, right, bottom, top], optional, default: None
        The location, in data-coordinates, of the lower-left and
        upper-right corners. If `None`, the image is positioned such that
        the pixel centers fall on zero-based (row, column) indices.  

    mode_bin : string in {'relative', 'absolute'}, optional, default is 'relative'
        if 'relative', a relative double threshold is performed
        if 'absolute', an double threshold with absolute value is performed
        
    display : boolean, optional, default is False
        Display the signal if True
        
    savefig : string, optional, default is None
        Root filename (with full path) is required to save the figures. Postfix
        is added to the root filename.
        
    **kwargs, optional. This parameter is used by the maad functions as well
        as the plt.plot and savefig functions.
        All the input arguments required or optional in the signature of the
        functions above can be passed as kwargs :
           double_threshold_abs(im, ext, bin_h=0.7, bin_l=0.2, display=False, 
                                 savefig=None, **kwargs)
           double_threshold_rel (im, ext, bin_std=5, bin_per=0.5, display=False, 
                                  savefig=None, **kwargs)
           
        ... and more, see matplotlib   
        
        example :
        im_bin = create_mask(im, ext, bin_std=5, bin_per=0.5, mode='relative',
                             display=True, savefig=None, dpi=300)    

    Returns
    -------
        im_bin: binary image 
    """
       
    if mode_bin == 'relative':
        bin_std=kwargs.pop('bin_std', 3) 
        bin_per=kwargs.pop('bin_per', 0.5) 
        im_bin = double_threshold_rel(im, ext, bin_std, bin_per, display, savefig, **kwargs)
        
    elif mode_bin == 'absolute':
        bin_h=kwargs.pop('bin_h', 0.7) 
        bin_l=kwargs.pop('bin_l', 0.3) 
        im_bin = double_threshold_abs(im, ext, bin_h, bin_l, display, savefig, **kwargs)   
    
    return im_bin 

"""****************************************************************************
*************                 select_rois auto                      ***********
****************************************************************************"""
def select_rois_auto(im_bin, ext, min_roi=None ,max_roi=None, display=False, 
                savefig = None, **kwargs):
    """
    Select rois candidates based on area of rois. min and max boundaries.
    The ouput image contains pixels with label as value.

    Parameters
    ----------
    im : 2d ndarray of scalars
        Spectrogram (or image)
        
    ext : list of scalars [left, right, bottom, top], optional, default: None
        The location, in data-coordinates, of the lower-left and
        upper-right corners. If `None`, the image is positioned such that
        the pixel centers fall on zero-based (row, column) indices. 
        
    min_roi, max_roi : scalars, optional, default : None
        Define the minimum and the maximum area possible for an ROI. If None, 
        the minimum ROI area is 1 pixel and the maximum ROI area is the area of 
        the image
        
    display : boolean, optional, default is False
        Display the signal if True
        
    savefig : string, optional, default is None
        Root filename (with full path) is required to save the figures. Postfix
        is added to the root filename.
        
    **kwargs, optional. This parameter is used by plt.plot and savefig functions
        ****************************************************    
        savefilename : str, optional, default :'_spectro_after_noise_subtraction.png'
            Postfix of the figure filename
        **************************************************** 
        figsize : tuple of integers, optional, default: (4,10)
            width, height in inches.  
        title : string, optional, default : 'Spectrogram'
            title of the figure
        xlabel : string, optional, default : 'Time [s]'
            label of the horizontal axis
        ylabel : string, optional, default : 'Amplitude [AU]'
            label of the vertical axis
        cmap : string or Colormap object, optional, default is 'gray'
            See https://matplotlib.org/examples/color/colormaps_reference.html
            in order to get all the  existing colormaps
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic', 
                      'viridis'...
        vmin, vmax : scalar, optional, default: None
            `vmin` and `vmax` are used in conjunction with norm to normalize
            luminance data.  Note if you pass a `norm` instance, your
            settings for `vmin` and `vmax` will be ignored.
        ext : scalars (left, right, bottom, top), optional, default: None
            The location, in data-coordinates, of the lower-left and
            upper-right corners. If `None`, the image is positioned such that
            the pixel centers fall on zero-based (row, column) indices.
        dpi : integer, optional, default is 96
            Dot per inch. 
            For printed version, choose high dpi (i.e. dpi=300) => slow
            For screen version, choose low dpi (i.e. dpi=96) => fast
        format : string, optional, default is 'png'
            Format to save the figure
            
        ... and more, see matplotlib   

    Returns
    -------
        im_label: 2d ndarray
            image with labels as values
            
        rois_bbox : list of tuple (min_y,min_x,max_y,max_x)
            Contain the bounding box of each ROI
            
        rois_label : list of tuple (labelID, labelname)
            Contain the label (LabelID=scalar,labelname=string) for each ROI
            LabelID is a number from 1 to the number of ROI. The pixel value 
            of im_label correspond the labelID
            Labname is a string. As the selection is auto, label is 'unknown'
            by default.
    """

    # test if max_roi and min_roi are defined
    if max_roi is None: 
        # the maximum ROI is set to the aera of the image
        max_roi=im_bin.shape[0]*im_bin.shape[1]
        
    if min_roi is None:
        # the min ROI area is set to 1 pixel
        min_roi = 1
        
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
    
    # Display
    if display : 
        ylabel =kwargs.pop('ylabel','Frequency [Hz]')
        xlabel =kwargs.pop('xlabel','Time [sec]') 
        title  =kwargs.pop('title','Selected ROIs') 
        figsize=kwargs.pop('figsize',(4, 13)) 
        
        randcmap = rand_cmap(len(rois_label))
        cmap   =kwargs.pop('cmap',randcmap) 
        
        _, fig = plot2D (im_rois, extent=ext, figsize=figsize,title=title, 
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
 
    return im_rois, rois_bbox, rois_label  

"""****************************************************************************
*************                 select_rois_manually                  ***********
****************************************************************************"""
def select_rois_man(im_bin, ext, filename, software='audacity', mask=True, 
                         display=False, savefig = None, **kwargs):
    """
    Select rois candidates that were previously labelled (annotated) manually
    The ouput image contains pixels with label as value.
    Only annotation done with Audacity software is supported 

    Parameters
    ----------
    im_bin : 2d ndarray of scalars
        Spectrogram (or image)
        
    ext : list of scalars [left, right, bottom, top], optional, default: None
        The location, in data-coordinates, of the lower-left and
        upper-right corners. If `None`, the image is positioned such that
        the pixel centers fall on zero-based (row, column) indices. 
        
    filename : string
        Filename (full path) with the annotation done with Audacity
        
    software : string, optional, default is 'audacity'
        Give the name of the software used to export the annotations.
    
    mask : boolean
        if True, the out image contains only ROI pixels that are in common
        with the automatic process. Otherwise, labels are rectangle shape 
        corresponding to the bounding box selected manually in the annotation
        software.
        
    display : boolean, optional, default is False
        Display the signal if True
        
    savefig : string, optional, default is None
        Root filename (with full path) is required to save the figures. Postfix
        is added to the root filename.
        
    **kwargs, optional. This parameter is used by plt.plot and savefig functions
        ****************************************************    
        savefilename : str, optional, default :'_spectro_after_noise_subtraction.png'
            Postfix of the figure filename
        **************************************************** 
        figsize : tuple of integers, optional, default: (4,10)
            width, height in inches.  
        title : string, optional, default : 'Spectrogram'
            title of the figure
        xlabel : string, optional, default : 'Time [s]'
            label of the horizontal axis
        ylabel : string, optional, default : 'Amplitude [AU]'
            label of the vertical axis
        cmap : string or Colormap object, optional, default is 'gray'
            See https://matplotlib.org/examples/color/colormaps_reference.html
            in order to get all the  existing colormaps
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic', 
                      'viridis'...
        vmin, vmax : scalar, optional, default: None
            `vmin` and `vmax` are used in conjunction with norm to normalize
            luminance data.  Note if you pass a `norm` instance, your
            settings for `vmin` and `vmax` will be ignored.
        ext : scalars (left, right, bottom, top), optional, default: None
            The location, in data-coordinates, of the lower-left and
            upper-right corners. If `None`, the image is positioned such that
            the pixel centers fall on zero-based (row, column) indices.
        dpi : integer, optional, default is 96
            Dot per inch. 
            For printed version, choose high dpi (i.e. dpi=300) => slow
            For screen version, choose low dpi (i.e. dpi=96) => fast
        format : string, optional, default is 'png'
            Format to save the figure
            
        ... and more, see matplotlib   

    Returns
    -------
    im_label: 2d ndarray
       image with labels as values
            
    rois_bbox : list of tuple (min_y,min_x,max_y,max_x)
        Contain the bounding box of each ROI
            
    rois_label : list of tuple (labelID, labelname)
        Contain the label (LabelID=scalar,labelname=string) for each ROI
        LabelID is a number from 1 to the number of ROI. The pixel value 
        of im_label correspond the labelID
        Labelname is a string. As the selection is auto, label is 'unknown'
        by default.
    """
 
    if filename is None:     
        raise Exception("Manual ROI selection requires an annotations filename")
        
    print(72 * '_')
    print('Manual ROIs selection in progress...')
    print ('Annotating filename %s' % filename)

    
    Nf, Nt = im_bin.shape
    df = (ext[3]-ext[2])/(Nf-1)
    dt = (ext[1]-ext[0])/(Nt-1)
    t0 = ext[0]
    f0 = ext[2] 
    
    # get the offset in x and y depending on the starting time and freq
    offset_x = np.round((t0)/dt).astype(int)
    offset_y = np.round((f0)/df).astype(int) 

    im_rois = np.zeros(im_bin.shape).astype(int) 
    bbox = []
    rois_bbox = []
    rois_label = []
    if software=='audacity':
        tab_out = read_audacity_annot(filename)
        
        ymin = (tab_out['fmin']/df-offset_y).astype(int)
        xmin = (tab_out['tmin']/dt-offset_x).astype(int)
        ymax = (tab_out['fmax']/df-offset_y).astype(int)
        xmax = (tab_out['tmax']/dt-offset_x).astype(int)
        zipped = zip(ymin,xmin,ymax,xmax)
        
        # add current bbox to the list of bbox
        bbox= list(zipped) 

    # Construction of the ROIS image with the bbox and labels
    index = 0
    labelID = []
    labelName = []
    for ymin,xmin,ymax,xmax in bbox:
        # test if bbox limit is inside the rois image
        if (ymin>0 and xmin>0 and ymax>0 and xmax>0) :
            rois_bbox.extend([(ymin,xmin,ymax,xmax)])
            index= index +1
            im_rois[ymin:ymax,xmin:xmax] = index
            labelName.append(tab_out['label'][index-1])
            labelID.append(index)
        
    rois_label = list(zip(labelID,labelName)) 

    rois_label_man = []
    rois_bbox_man= []
    if mask:
        im_rois = (im_bin * im_rois).astype(int)
        # Select only rois_label that are present into im_rois
        for index, (ID, LABEL) in enumerate (rois_label): 
            if ID in im_rois:
                 rois_label_man.append(rois_label[index])
                 rois_bbox_man.append(rois_bbox[index])
    else:
        rois_label_man = rois_label
        rois_bbox_man = rois_bbox
                
    # Display
    if display : 
        ylabel =kwargs.pop('ylabel','Frequency [Hz]')
        xlabel =kwargs.pop('xlabel','Time [sec]') 
        title  =kwargs.pop('title','Selected ROIs')
        figsize=kwargs.pop('figsize',(4, 13)) 
        
        randcmap = rand_cmap(len(np.unique(rois_label_man)))
        cmap   =kwargs.pop('cmap',randcmap) 
        
        _, fig = plot2D (im_rois, extent=ext, figsize=figsize,title=title, 
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
      
    return im_rois, rois_bbox_man, rois_label_man 

"""****************************************************************************
*************                   select ROIS wrapper                 ***********
****************************************************************************"""
def select_rois(im_bin,ext,mode_roi='auto',display=False,savefig=None,**kwargs):
    """
    Wrapper function
    Select rois candidates automatically or manually (annotating file)
    The ouput image contains pixels with labels as values.
    
    Parameters
    ----------
    im : 2d ndarray of scalars
        Spectrogram (or image)
        
    ext : list of scalars [left, right, bottom, top], optional, default: None
        The location, in data-coordinates, of the lower-left and
        upper-right corners. If `None`, the image is positioned such that
        the pixel centers fall on zero-based (row, column) indices.  

    mode_roi : string in {'auto', 'manual'}, optional, default is 'auto'
        if 'auto', an automatic selection of the ROIs is performed
        if 'manual', ROIs manually found using another software (i.e. Audacity) 
        are retrieved. If mask = True, a boolean AND operation is performed 
        between the manual ROIs (Rectangles) and the binary image, else no boolean
        operation is performed with the binary image and the output label image
        is directly the rectangles (bounding box)
        
    display : boolean, optional, default is False
        Display the signal if True
        
    savefig : string, optional, default is None
        Root filename (with full path) is required to save the figures. Postfix
        is added to the root filename.
        
    **kwargs, optional. This parameter is used by the maad functions as well
        as the plt.plot and savefig functions.
        All the input arguments required or optional in the signature of the
        functions above can be passed as kwargs :
           select_rois_auto(im_bin,ext,min_roi,max_roi,display,savefig,
                             **kwargs)
           select_rois_manually(im_bin,ext,filename,software,mask,display,
                                 savefig,**kwargs
           
        ... and more, see matplotlib   
        
        example :
        im_rois, rois_bbox, rois_label =select_rois_auto(im_bin,ext,mode='auto',
                                        min_roi=100,max_roi=1e6,
                                        display=True,savefig=None,**kwargs) 
        
        im_rois, rois_bbox, rois_label =select_rois_man(im_bin,ext,mode='manual',
                                        filename='annotation.txt',
                                        software='audacity,
                                        display=True,savefig=None,**kwargs)  

    Returns
    -------
        im_rois: 2d ndarray
            image with labels as values
            
        rois_bbox : list of tuple (min_y,min_x,max_y,max_x)
            Contains the bounding box of each ROI
            
        rois_label : list of tuple (labelID, labelname)
            Contains the label (LabelID=scalar,labelname=string) for each ROI
            LabelID is a number from 1 to the number of ROI. The pixel value 
            of im_label correspond the labelID
            Labname is a string. As the selection is auto, label is 'unknown'
            by default.
    """    

    if mode_roi == 'auto':
        min_roi=kwargs.pop('min_roi', None) 
        max_roi=kwargs.pop('max_roi', None) 
        im_rois,rois_bbox,rois_label=select_rois_auto(im_bin,ext,min_roi,
                                         max_roi,display,savefig,**kwargs)
        
    elif mode_roi == 'manual':
        filename=kwargs.pop('filename', None) 
        software=kwargs.pop('software', 'audacity') 
        mask=kwargs.pop('mask', True) 
        im_rois,rois_bbox,rois_label=select_rois_man(im_bin,ext,filename,
                                       software,mask,display,savefig,**kwargs)    
    
    return im_rois, rois_bbox, rois_label


"""****************************************************************************
*************                   display_rois                        ***********
****************************************************************************"""
def overlay_rois (im_ref, ext, rois_bbox, rois_label=None, savefig=None, **kwargs):
    """
    Overlay bounding box on the original spectrogram
    
    Parameters
    ----------
    im_ref : 2d ndarray of scalars
        Spectrogram (or image)
        
    ext : list of scalars [left, right, bottom, top], optional, default: None
        The location, in data-coordinates, of the lower-left and
        upper-right corners. If `None`, the image is positioned such that
        the pixel centers fall on zero-based (row, column) indices.  

    rois_bbox : list of tuple (min_y,min_x,max_y,max_x)
        Contains the bounding box of each ROI
        
    rois_label : list of tuple (labelID, labelname), optional, default is None
        Contains the label (LabelID=scalar,labelname=string) for each ROI
        LabelID is a number from 1 to the number of ROI. The pixel value 
        of im_label correspond the labelID
        Labname is a string. After an automatic selection, label is 'unknown'
        by default.  
        
    savefig : string, optional, default is None
        Root filename (with full path) is required to save the figures. Postfix
        is added to the root filename.
        
    **kwargs, optional. This parameter is used by plt.plot and savefig functions
        ****************************************************    
        savefilename : str, optional, default :'_spectro_overlayrois.png'
            Postfix of the figure filename
        **************************************************** 
        figsize : tuple of integers, optional, default: (4,10)
            width, height in inches.  
        title : string, optional, default : 'Spectrogram'
            title of the figure
        xlabel : string, optional, default : 'Time [s]'
            label of the horizontal axis
        ylabel : string, optional, default : 'Amplitude [AU]'
            label of the vertical axis
        cmap : string or Colormap object, optional, default is 'gray'
            See https://matplotlib.org/examples/color/colormaps_reference.html
            in order to get all the  existing colormaps
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic', 
                      'viridis'...
        vmin, vmax : scalar, optional, default: None
            `vmin` and `vmax` are used in conjunction with norm to normalize
            luminance data.  Note if you pass a `norm` instance, your
            settings for `vmin` and `vmax` will be ignored.
        ext : scalars (left, right, bottom, top), optional, default: None
            The location, in data-coordinates, of the lower-left and
            upper-right corners. If `None`, the image is positioned such that
            the pixel centers fall on zero-based (row, column) indices.
        dpi : integer, optional, default is 96
            Dot per inch. 
            For printed version, choose high dpi (i.e. dpi=300) => slow
            For screen version, choose low dpi (i.e. dpi=96) => fast
        format : string, optional, default is 'png'
            Format to save the figure 
        
        ... and more, see matplotlib 

    Returns
    -------
        ax : axis object (see matplotlib)
        
        fig : figure object (see matplotlib)
    """       
    
    ylabel =kwargs.pop('ylabel','Frequency [Hz]')
    xlabel =kwargs.pop('xlabel','Time [sec]') 
    title  =kwargs.pop('title','ROIs Overlay')
    cmap   =kwargs.pop('cmap','gray') 
    figsize=kwargs.pop('figsize',(4, 13)) 
    vmin=kwargs.pop('vmin',0) 
    vmax=kwargs.pop('vmax',1) 
        
    ax, fig = plot2D (im_ref,extent=ext,now=False, figsize=figsize,title=title, 
                     ylabel=ylabel,xlabel=xlabel,vmin=vmin,vmax=vmax, 
                     cmap=cmap, **kwargs)

    # Convert pixels into time and frequency values
    y_len, x_len = im_ref.shape
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim() 
    x_scaling = (xmax-xmin) / x_len
    y_scaling = (ymax-ymin) / y_len
    
    # get the labelName
    if rois_label is not None:
        labelID, labelName = zip(*rois_label)
        labelNames = np.unique(np.array(labelName))
    
    if (rois_label is None) or (len(labelNames)==1) :
        for y0, x0, y1, x1 in rois_bbox:
            rect = mpatches.Rectangle((x0*x_scaling+xmin, y0*y_scaling+ymin), 
                                      (x1-x0)*x_scaling, 
                                      (y1-y0)*y_scaling,
                                      fill=False, edgecolor='yellow', linewidth=1)  
            # draw the rectangle
            ax.add_patch(rect)
    else :
        # Colormap
        color = rand_cmap(len(labelNames)+1,first_color_black=False) 
        cc = 0
        for bbox, label in zip(rois_bbox, rois_label):
            cc = cc+1
            print(cc)
            y0 = bbox[0]
            x0 = bbox[1]
            y1 = bbox[2]
            x1 = bbox[3]
            for index, name in enumerate(labelNames):
                if label[1] in name: ii = index
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

"""****************************************************************************
*************                   find_rois_wrapper                   ***********
****************************************************************************"""
def find_rois_wrapper(im, ext, display=False, savefig=None, **kwargs):
    """
    Wrapper function to find and select ROIs in a spectrogram (or image)
    
    Parameters
    ----------
    im : 2d ndarray of scalars
        Spectrogram (or image) 
        
    ext : list of scalars [left, right, bottom, top], optional, default: None
        The location, in data-coordinates, of the lower-left and
        upper-right corners. If `None`, the image is positioned such that
        the pixel centers fall on zero-based (row, column) indices.  
            
    display : boolean, optional, default is False
        Display the signals and the spectrograms if True
        
    savefig : string, optional, default is None
        Root filename (with full path) is required to save the figures. Postfix
        is added to the root filename.
        
    **kwargs, optional. This parameter is used by the maad function as well
        as the plt.plot and savefig functions.
        All the input arguments required or optional in the signature functions
        can be passed.
        
        Specific parameters
        
             std_pre : scalar
                 Standard deviation used for the first call of the smooth() 
                 function. It defines the std of the gaussian kernel
                 
             std_post : scalar
                 Standard deviation used for the Second call of the smooth() 
                 function. It defines the std of the gaussian kernel
               
        See the signature of each maad function to know the other parameters 
        that can be passed as kwargs :
            remove_background(im, ext, gauss_win=50, gauss_std = 25, beta1=1, 
                              beta2=1,llambda=1, display = False, savefig=None, 
                              **kwargs)
            
            smooth (im, ext, std=1, display = False, savefig=None, **kwargs)
            
            create_mask(im, ext, mode_bin ='relative', display=False, savefig=None,
                **kwargs)
            
            select_rois(im_bin,ext,mode_roi='auto',display=False,savefig=None,
                        **kwargs)
            
            overlay_rois (im_ref, ext, rois_bbox, rois_label=None, savefig=None,
                          **kwargs)
            
        ... and more, see matplotlib  
        
        example : 
            find_rois_wrapper(im_ref, ext, display=True,
                              std_pre = 2, std_post=1, 
                              llambda=1.1, gauss_win = round(1000/df),
                              mode_bin='relative', bin_std=5, bin_per=0.5,
                              mode_roi='auto')
        
        ... and more, see matplotlib 

    Returns
    -------
        im_rois: 2d ndarray
            image with labels as values
            
        rois_bbox : list of tuple (min_y,min_x,max_y,max_x)
            Contains the bounding box of each ROI
            
        rois_label : list of tuple (labelID, labelname)
            Contains the label (LabelID=scalar,labelname=string) for each ROI
            LabelID is a number from 1 to the number of ROI. The pixel value 
            of im_label correspond the labelID
            Labname is a string. As the selection is auto, label is 'unknown'
            by default.
    """       
    
    
    # Frequency and time resolution
    df = (ext[3]-ext[2])/(im.shape[0]-1)
    dt = (ext[1]-ext[0])/(im.shape[1]-1)
   
    # keep a copy of the reference image
    im_ref=im
    
    gauss_win=kwargs.pop('gauss_win',round(1000/df))
    gauss_std=kwargs.pop('gauss_std',round(500/df))
    beta1=kwargs.pop('beta1', 0.8)
    beta2=kwargs.pop('beta2', 1)
    llambda=kwargs.pop('llambda', 1)  
    
    std_pre=kwargs.pop('std_pre', 2) 
    std_post=kwargs.pop('std_post', 1) 
       
    mode_bin=kwargs.pop('mode_bin', 'relative') 
    mode_roi=kwargs.pop('mode_roi', 'auto') 
        
    min_roi=kwargs.pop('min_roi', None)
    if min_roi is None:
        min_f = ceil(100/df) # 100Hz 
        min_t = ceil(0.1/dt) # 100ms 
        min_roi=np.min(min_f*min_t) 
        
    max_roi=kwargs.pop('max_roi', None) 
    if max_roi is None:   
        # 1000Hz or vertical size of the image
        max_f = np.asarray([round(1000/df), im.shape[0]]) 
        # horizontal size of the image or 1s
        max_t = np.asarray([im.shape[1], round(1/dt)])     
        max_roi =  np.max(max_f*max_t)
    
    kwargs['min_roi']=min_roi 
    kwargs['max_roi']=max_roi
    
    # smooth
    if std_pre>0 :
       im = smooth(im, ext, std=std_pre, display=display, 
                              savefig=savefig,**kwargs)

    # Noise subtraction
    im = remove_background(im, ext, gauss_win=gauss_win, 
                                    gauss_std=gauss_std, beta1=beta1, 
                                    beta2=beta2, llambda=llambda, 
                                    display= display, savefig=savefig, **kwargs)
    
    # smooth
    if std_post>0:
        im = smooth(im, ext, std=std_post, display=display,
                            savefig=savefig,**kwargs)
    
    # Binarization
    im = create_mask(im, ext, mode=mode_bin, display=display, savefig=savefig, **kwargs)
    
    # Rois extraction
    im_rois, rois_bbox, rois_label = select_rois(im,ext,mode=mode_roi,
                                                  display=display, 
                                                  savefig=savefig, **kwargs)
    
    if display: overlay_rois(im_ref, ext, rois_bbox, savefig=savefig, **kwargs)

    return im_rois, rois_bbox, rois_label
