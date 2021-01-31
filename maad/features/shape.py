#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
""" 
Ensemble of functions to compute acoustic descriptors from 2D spectrograms.
""" 
#
# Authors:  Juan Sebastian ULLOA <lisofomia@gmail.com>
#           Sylvain HAUPERT <sylvain.haupert@mnhn.fr>        
#
# License: New BSD License

#%%
#***************************************************************************
# -------------------       Load modules         ---------------------------
#***************************************************************************
# Import external modules

from __future__ import print_function 
import numpy as np 
import pandas as pd 
from scipy import ndimage as ndi 
import itertools as it 
import matplotlib.pyplot as plt 
from skimage import transform, measure 
from scipy import ndimage 

# Import internal modules
from maad.util import format_features, overlay_rois, plot_shape, overlay_centroid
from maad.rois import rois_to_imblobs
from maad.sound import spectrogram 
 
#%%
#============================================================================ 
# PRIVATE FUNCTIONS 
#============================================================================ 
def _gabor_kernel_nodc(frequency, theta=0, bandwidth=1, gamma=1, 
                      n_stds=3, offset=0): 
    """ 
    Computes complex 2D Gabor filter kernel with no DC offset. 
         
    Gabor kernel is a Gaussian kernel modulated by a complex harmonic function. 
    Harmonic function consists of an imaginary sine function and a real 
    cosine function. Spatial frequency is inversely proportional to the 
    wavelength of the harmonic and to the standard deviation of a Gaussian 
    kernel. The bandwidth is also inversely proportional to the standard 
    deviation. 
     
    Parameters 
    ---------- 
    frequency : float 
        Spatial frequency of the harmonic function. Specified in pixels. 
    theta : float, optional 
        Orientation in radians. If 0, the harmonic is in the x-direction. 
    bandwidth : float, optional 
        The bandwidth captured by the filter. For fixed bandwidth, `sigma_x` 
        and `sigma_y` will decrease with increasing frequency. This value is 
        ignored if `sigma_x` and `sigma_y` are set by the user. 
    gamma : float, optional 
        gamma changes the aspect ratio (ellipsoidal) of the gabor filter.  
        By default, gamma=1 which means no aspect ratio (circle) 
         
        if gamma>1, the filter is larger (x-dir) 
         
        if gamma<1, the filter is higher (y-dir) 
         
        This value is ignored if `sigma_x` and `sigma_y` are set by the user. 
    sigma_x, sigma_y : float, optional 
        Standard deviation in x- and y-directions. These directions apply to 
        the kernel before rotation. If `theta = pi/2`, then the kernel is 
        rotated 90 degrees so that `sigma_x` controls the vertical direction.    
    n_stds : scalar, optional 
        The linear size of the kernel is n_stds (3 by default) standard 
        deviations 
    offset : float, optional 
        Phase offset of harmonic function in radians. 
     
    Returns 
    ------- 
    g_nodc : complex 2d array 
        A complex gabor kernel with no DC offset 
     
    Notes 
    ----- 
    This function is a modification of the gabor_kernel function of scikit-image. 
     
    References 
    ---------- 
    .. [1] http://en.wikipedia.org/wiki/Gabor_filter 
    .. [2] http://mplab.ucsd.edu/tutorials/gabor.pdf 
    .. [3] http://www.cs.rug.nl/~imaging/simplecell.html 
     
    """ 
     
     # set gaussian parameters 
    b = bandwidth 
    sigma_pref = 1.0 / np.pi * np.sqrt(np.log(2) / 2.0) * (2.0 ** b + 1) / (2.0 ** b - 1) 
    sigma_y = sigma_pref / frequency 
    sigma_x = sigma_y / gamma 
    # meshgrid 
    x0 = np.ceil(max(np.abs(n_stds * sigma_x * np.cos(theta)), 
                     np.abs(n_stds * sigma_y * np.sin(theta)), 1)) 
    y0 = np.ceil(max(np.abs(n_stds * sigma_y * np.cos(theta)), 
                     np.abs(n_stds * sigma_x * np.sin(theta)), 1)) 
    y, x = np.mgrid[-y0:y0 + 1, -x0:x0 + 1] 
    # rotation matrix 
    rotx = x * np.cos(theta) + y * np.sin(theta) 
    roty = -x * np.sin(theta) + y * np.cos(theta) 
    # combine gambor and  
    g = np.zeros(y.shape, dtype=np.complex) 
    g[:] = np.exp(-0.5 * (rotx ** 2 / sigma_x ** 2 + roty ** 2 / sigma_y ** 2)) 
    g /= 2 * np.pi * sigma_x * sigma_y # gaussian envelope 
    oscil = np.exp(1j * (2 * np.pi * frequency * rotx + offset)) # harmonic / oscilatory function 
    g_dc = g*oscil 
    # remove dc component by subtracting the envelope weighted by K 
    K = np.sum(g_dc)/np.sum(g) 
    g_nodc = g_dc - K*g 
  
    return g_nodc 
 
#%%
def _plot_filter_bank(kernels, frequency, ntheta): 
    """ 
    Display filter bank 
     
    Parameters 
    ---------- 
    kernels: list 
        List of kernels from filter_bank_2d_nodc() 
         
    frequency: 1d ndarray of scalars 
        Spatial frequencies used to built the Gabor filters. Values should be 
        in [0;1] 
     
    ntheta: int 
        Number of angular steps between 0° to 90° 
     
    Returns 
    ------- 
    fig : Figure 
        The Figure instance  
    ax : Axis 
        The Axis instance 
    """ 
 
    ntheta = ntheta 
    nfreq = len(frequency) 
     
    # get maximum size 
    aux = list() 
    for kernel in kernels: 
        aux.append(max(kernel[0].shape)) 
    max_size = np.max(aux) 
     
    # plot kernels 
    fig, ax = plt.subplots(nfreq, ntheta) 
    ax = ax.transpose() 
    ax = ax.ravel() 
    for idx, k in enumerate(kernels): 
        kernel = np.real(k[0]) 
        ki, kj = kernel.shape 
        ci, cj = int(max_size/2 - ki/2), int(max_size/2 - kj/2) 
        canvas = np.zeros((max_size,max_size)) 
        canvas[ci:ci+ki,cj:cj+kj] = canvas[ci:ci+ki,cj:cj+kj] + kernel 
        ax[idx].imshow(canvas) 
        ax[idx].axis('off') 
    plt.tight_layout() 
    plt.subplots_adjust(wspace=0.02, hspace=0.02) 
    return ax, fig 
 
#%% 
def _plot_filter_results(im_ref, im_list, kernels, params, m, n): 
    """ 
    Display the resulting spectrograms after filtering with gabor filters 
     
    Parameters 
    ---------- 
    im_ref : 2D array 
        Reference image 
    im_list : list 
        List of filtered images 
    kernels: list 
        List of kernels from filter_bank_2d_nodc() 
    m: int 
        number of columns 
    n: int 
        number of rows 
         
    Returns 
    ------- 
    fig : Figure 
        The Figure instance  
    ax : Axis 
        The Axis instance 
    """     
         
    ncols = m  
    nrows = n 
     
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5)) 
    plt.gray() 
    fig.suptitle('Image responses for Gabor filter kernels', fontsize=12) 
    axes[0][0].axis('off') 
     
    # Plot original images 
    axes[0][1].imshow(im_ref, origin='lower') 
    axes[0][1].set_title('spectrogram', fontsize=9) 
    axes[0][1].axis('off') 
    plt.tight_layout 
     
    params_label = [] 
    for param in params: 
        params_label.append('theta=%d,\nf=%.2f' % (param[1] * 180 / np.pi, param[0])) 
     
    ii = 0 
    for ax_row in axes[1:]:  
        plotGabor = True 
        for ax in ax_row: 
            if plotGabor == True: 
                # Plot Gabor kernel 
                print(params_label[ii]) 
                ax.imshow(np.real(kernels[ii]), interpolation='nearest') 
                ax.set_ylabel(params_label[ii], fontsize=7) 
                ax.set_xticks([]) 
                ax.set_yticks([]) 
                plotGabor = False 
            else: 
                im_filtered = im_list[ii] 
                ax.imshow(im_filtered, origin='lower') 
                ax.axis('off') 
                plotGabor = True 
                ii=ii+1      
         
    plt.show() 
    return ax, fig 
 
#%%
def _filter_mag(im, kernel): 
    """ 
    Normalizes the image and computes im and real part of filter response using  
    the complex kernel and the modulus operation 
     
    Parameters 
    ---------- 
    im: 2D array 
        Input image to process  
    kernel: 2D array 
        Complex kernel or filter 
             
    Returns 
    ------- 
    im_out: Modulus operand on filtered image 
 
    """     
     
    im = (im - im.mean()) / im.std() 
    im_out = np.sqrt(ndi.convolve(im, np.real(kernel), mode='reflect')**2 + 
                   ndi.convolve(im, np.imag(kernel), mode='reflect')**2) 
    return im_out 
 
#%%
def _params_to_df(params_filter_bank, npyr): 
    """ 
    Organises parameters used in multiresolution analysis into a dataframe 
     
    Parameters 
    ---------- 
    params_filter_bank: numpy array 
        Output of parameters from function filter_bank_2d_nodc 
    npyr: int 
        Number of pyramids used in multiresolution analysis 
             
    Returns 
    ------- 
    params_multires: pandas DataFrame 
        Ordered parameters for each shape feature used 
    """     
 
    params = np.asarray(params_filter_bank) 
    orient = params[:,0] * 180 / np.pi 
    orient = orient.tolist() * npyr 
    pyr_level = np.sort(np.arange(npyr).tolist()*len(params))+1 
    freq = params[:,1].tolist() * npyr 
    nparams = len(params) * npyr 
    params_multires = np.zeros(nparams, dtype={'names':('theta', 'freq', 'pyr_level','scale'), 
                                               'formats':('f8', 'f8', 'f8','f8')}) 
    params_multires['theta'] = orient 
    params_multires['freq'] = freq 
    params_multires['scale'] = 1 / np.asarray(freq) 
    params_multires['pyr_level'] = pyr_level 
    params_multires = pd.DataFrame(params_multires) 
 
    return params_multires 
 
#%%
#============================================================================ 
# Public functions 
#============================================================================ 
def filter_multires(Sxx, kernels, npyr=4, rescale=True): 
    """ 
    Computes 2D wavelet coefficients at multiple scales using Gaussian pyramid  
    transformation to downscale the input spectrogram. 
     
    Parameters 
    ---------- 
    Sxx: list of 2D arrays 
        List of input spectrograms to filter 
    kernels: list of 2D arrays 
        List of 2D kernels or filters 
    npyr: int, optional 
        Number of pyramids to compute. Default is 4. 
    rescale: boolean, optional 
        Indicates if the reduced images should be rescaled. Default is True. 
             
    Returns 
    ------- 
    Sxx_out: list of 2D arrays 
        List of spectrograms filtered by each 2D kernel 
     
    Examples 
    -------- 
    >>> from maad.sound import load, spectrogram 
    >>> from maad.features import filter_bank_2d_nodc, filter_multires 
    >>> from maad import util
    >>> s, fs = load('../data/spinetail.wav') 
    >>> Sxx, dt, df, ext = spectrogram(s, fs) 
    >>> Sxx_db = util.power2dB(Sxx, db_range=80) + 80
    >>> util.plot2d(Sxx_db, **{'extension':ext})
    >>> params, kernels = filter_bank_2d_nodc(frequency=(0.5, 0.25), ntheta=2,gamma=2) 
    >>> Sxx_out = filter_multires(Sxx, kernels, npyr=2)
    
    Plot one of the resulting spectrograms.
    
    >>> util.plot2d(Sxx_out[5], **{'extension':ext})
     
    """     
 
    # Downscale image using gaussian pyramid 
    if npyr<2: 
        print('Warning: npyr should be int and larger than 2 for multiresolution') 
        im_pyr = tuple(transform.pyramid_gaussian(Sxx, downscale=2,  
                                                  max_layer=1, multichannel=False))  
    else:     
        im_pyr = tuple(transform.pyramid_gaussian(Sxx, downscale=2,  
                                                  max_layer=npyr-1, multichannel=False))  
 
    # filter 2d array at multiple resolutions using gabor kernels 
    im_filt=[] 
    for im in im_pyr:  # for each pyramid 
        for kernel, param in kernels:  # for each kernel 
            im_filt.append(_filter_mag(im, kernel))  #  magnitude response of filter 
     
    # Rescale image using gaussian pyramid 
    if rescale: 
        dims_raw = Sxx.shape 
        Sxx_out=[] 
        for im in im_filt: 
            ratio = np.array(dims_raw)/np.array(im.shape) 
            if ratio[0] > 1: 
                im = transform.rescale(im, scale = ratio, mode='reflect', 
                                       multichannel=False, anti_aliasing=True) 
            else: 
                pass 
            Sxx_out.append(im) 
    else: 
        pass 
 
    return Sxx_out 
 
#%%
def filter_bank_2d_nodc(frequency, ntheta, bandwidth=1, gamma=2, display=False, **kwargs): 
    r""" 
    Build an ensemble of complex 2D Gabor filters with no DC offset. 
    
    A Gabor kernel is a Gaussian kernel modulated by a complex harmonic function. 
    Harmonic function consists of an imaginary sine function and a real 
    cosine function. Spatial frequency is inversely proportional to the 
    wavelength of the harmonic and to the standard deviation of a Gaussian 
    kernel. The bandwidth is also inversely proportional to the standard 
    deviation [1]_. These filters have been used to classify textures in images [4]_ and 
    can be applied to characterize spectrograms to classify sounds [5]_.
     
    Parameters 
    ---------- 
    frequency: 1d ndarray of scalars 
        Spatial frequencies used to built the Gabor filters. Values should be 
        in the range [0;1] 
     
    ntheta: int 
        Number of angular steps between 0° to 90° 
     
    bandwidth: scalar, optional, default is 1 
        Spatial-frequency bandwidth of the filter. This parameters affect 
        the resolution of the filters. Lower bandwidth result in sharper 
        in filters with more details. 
     
    gamma: scalar, optional, default is 1 
        Gaussian window that modulates the continuous sine. 
        For ``gamma = 1``, the result is the same gaussian window in x and y direction (circle). 
        For ``gamma <1``, the result is an increased elongation of the filter size in the y direction (elipsoid). 
        For ``gamma >1``, the result is a reduction of the filter size in the y direction (elipsoid). 
     
    display: bool 
        Display a visualization of the filter bank. Default is False. 
 
    Returns 
    ------- 
    params: 2d structured array 
         Parameters used to calculate 2D gabor kernels.  
         Params array has 4 fields (theta, freq, bandwidth, gamma) 
         This can be useful to interpret the result of the filtering process. 
             
    kernels: 2d ndarray of complex values 
         Complex Gabor kernels 
    
    References 
    ---------- 
    .. [1] http://en.wikipedia.org/wiki/Gabor_filter 
    .. [2] http://mplab.ucsd.edu/tutorials/gabor.pdf 
    .. [3] http://www.cs.rug.nl/~imaging/simplecell.html 
    .. [4] Sifre, L., & Mallat, S. (2013). Rotation, scaling and deformation invariant scattering for texture discrimination. Computer Vision and Pattern Recognition (CVPR), 2013 IEEE Conference On, 1233–1240. http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=6619007
    .. [5] Ulloa, J. S., Aubin, T., Llusia, D., Bouveyron, C., & Sueur, J. (2018). Estimating animal acoustic diversity in tropical environments using unsupervised multiresolution analysis. Ecological Indicators, 90, 346–355. https://doi.org/10.1016/j.ecolind.2018.03.026

    
    See also
    --------
    filter_multires, opt_shape_presets, shape_features
    
    Examples 
    -------- 
    Build a filter bank using predefined parameters with the function maad.features.opt_shape_presets.
 
    >>> from maad.features import filter_bank_2d_nodc, opt_shape_presets 
    >>> opt = opt_shape_presets(resolution='med') 
    >>> params, kernels = filter_bank_2d_nodc(opt['frequency'], opt['ntheta'], opt['bandwidth'], opt['gamma'], display=True) 
 
    Alternatively, custom parameters can be provided to define the filter bank.
 
    >>> from maad.features import filter_bank_2d_nodc 
    >>> params, kernels = filter_bank_2d_nodc(frequency=(0.7, 0.5, 0.35, 0.25), ntheta=4, gamma=2, display=True) 
 
    """ 
     
    theta = np.arange(ntheta) 
    theta = theta / ntheta * np.pi 
    params=[i for i in it.product(theta,frequency)] 
    kernels = [] 
    for param in params: 
        kernel = _gabor_kernel_nodc(frequency=param[1], 
                                   theta=param[0], 
                                   bandwidth=bandwidth, 
                                   gamma=gamma, 
                                   offset=0, 
                                   n_stds=3) 
        kernels.append((kernel, param)) 
             
    if display:  
        _, fig = _plot_filter_bank(kernels, frequency, ntheta) 
             
    return params, kernels   
 
#%%
def opt_shape_presets(resolution, opt_shape=None): 
    """  
    Set parameters for multiresolution analysis using presets or custom parameters.
     
    Parameters 
    ---------- 
    resolution: str 
        Select resolution of analysis using presets.  
        Supported presets are: 'low', 'med', and 'high'. 
        Select 'custom' to select user-defined parameters using a dictionary. 
         
    opt_shape: dict 
        Key and values for shape settings. 
        Valid keys are: 'ntheta', 'bandwidth', 'frequency', 'gamma', 'npyr' 
     
    Returns 
    ------- 
    opt_shape: dict 
        A valid dictionary with shape settings 
        
    See also
    --------
    filter_bank_2d_nodc, filter_multires, shape_features
         
    Examples 
    -------- 
    Get parameters to analyse at low, med and high shape resolutions.
     
    >>> from maad.features import opt_shape_presets 
    >>> opt_shape_presets('low') 
    {'ntheta': 2,
     'bandwidth': 0.8,
     'frequency': (0.35, 0.5),
     'gamma': 2,
     'npyr': 4}
    
    >>> opt_shape_presets('med') 
    {'ntheta': 4,
     'bandwidth': 0.8,
     'frequency': (0.35, 0.5),
     'gamma': 2,
     'npyr': 6}
    
    >>> opt_shape_presets('high')
    {'ntheta': 8,
     'bandwidth': 0.8,
     'frequency': (0.35, 0.5),
     'gamma': 2,
     'npyr': 6}
    
    """ 
    # Factory presets 
    opt_shape_low = dict(ntheta=2,  
                         bandwidth=0.8,  
                         frequency=(0.35, 0.5),  
                         gamma=2,  
                         npyr = 4) 
    opt_shape_med =  dict(ntheta=4,  
                          bandwidth=0.8,  
                          frequency=(0.35, 0.5),  
                          gamma=2,  
                          npyr = 6) 
    opt_shape_high =  dict(ntheta=8,  
                           bandwidth=0.8,  
                           frequency=(0.35, 0.5),  
                           gamma=2,  
                           npyr = 6) 
     
    if resolution == 'low': 
        opt_shape = opt_shape_low 
     
    elif resolution == 'med': 
       opt_shape = opt_shape_med 
         
    elif resolution == 'high': 
       opt_shape = opt_shape_high 
     
    elif resolution == 'custom': 
        if opt_shape is not None:  # check valid values on opt_shape  
           if all (opt in opt_shape for opt in ('ntheta', 'bandwidth', 'frequency', 'gamma', 'npyr')): 
               pass 
           else: 
               print('Warning: opt_shape must have all keys-values pairs:') 
               print('ntheta, bandwidth, frequency, gamma, npyr') 
               print('Setting resolution to low') 
               opt_shape = opt_shape_low 
 
        else: 
            print('Warning: if resolution is set to custom, a valid opt_shape dictionnary should be provided.') 
            print('Setting resolution to low') 
            opt_shape = opt_shape_low 
    
    else: 
       raise TypeError("Resolution should be: 'low', 'med' or 'high'") 
       opt_shape = opt_shape_low 
 
    return opt_shape 
 
#%%
def shape_features(Sxx, resolution='low', rois=None): 
    """ 
    Computes time-frequency shape coefficients at multiple resolutions using 2D Gabor filters.
     
    Parameters 
    ---------- 
    Sxx: 2D array 
        Input image to process  
    resolution: str or dict 
        Specify resolution of shape descriptors. Can be: 'low', 'med', 'high'. 
        Default is 'low'. Alternatively, custom resolution can be provided  
        using a dictionary with options to define the filter bank. Valid keys  
        are: ntheta, bandwidth, frequency, gamma, npyr.
    rois: pandas DataFrame 
        Regions of interest where descriptors will be computed. Array must  
        have a valid input format with column names: min_t min_f, max_t, max_f. 
        Use format_features(rois,tn,fn) before using shape_features to be sure that 
        the format of the rois DataFrame is correct.
             
    Returns 
    ------- 
    shape: pandas DataFrame 
        Shape coefficient for each region of interest
    params: 2D numpy structured array 
        Corresponding parameters of the 2D fileters used to compute the  
        shape coefficients. Params has 4 fields (theta, freq, pyr_level, scale) 
        
    See Also
    --------
    maad.features.opt_shape_presets, maad.features.filter_bank_2d_nodc, maad.util.plot_shape
    
    References
    ----------
    1. Ulloa, J. S., Aubin, T., Llusia, D., Bouveyron, C., & Sueur, J. (2018). Estimating animal acoustic diversity in tropical environments using unsupervised multiresolution analysis. Ecological Indicators, 90, 346–355. https://doi.org/10.1016/j.ecolind.2018.03.026
    2. Sifre, L., & Mallat, S. (2013). Rotation, scaling and deformation invariant scattering for texture discrimination. Computer Vision and Pattern Recognition (CVPR), 2013 IEEE Conference On, 1233–1240. http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=6619007
    3. Mallat, S. (2008). A Wavelet Tour of Signal Processing: The Sparse Way. Academic Press.
    
    Examples 
    -------- 
 
    Get shape features from the whole power spectrogram 
 
    >>> from maad.sound import load, spectrogram 
    >>> from maad.features import shape_features
    >>> from maad.util import format_features, power2dB, plot_shape 
    >>> s, fs = load('../data/spinetail.wav')
    >>> Sxx, tn, fn, ext = spectrogram(s, fs, db_range=100, display=True) 
    >>> Sxx_db = power2dB(Sxx, db_range=100)
    >>> shape, params = shape_features(Sxx_db, resolution='med') 
    >>> ax = plot_shape(shape.mean(), params) 
     
    Or get shape features from specific regions of interest 
     
    >>> from maad.util import read_audacity_annot
    >>> rois = read_audacity_annot('../data/spinetail.txt') 
    >>> rois = format_features(rois, tn, fn) 
    >>> shape, params = shape_features(Sxx_db, resolution='med', rois=rois)
    
    The shape coefficient can be visualized using ``plot_shape``. Note that these
    coefficient allow to disciminate between the two vocalization types found in the
    audio file, the spinetail song and the background calls.
    
    >>> shape_crer, shape_sp = shape.groupby('label')
    >>> ax1 = plot_shape(shape_crer[1].mean(), params) 
    >>> ax1.set_title('Spinetail song')
    >>> ax2 = plot_shape(shape_sp[1].mean(), params) 
    >>> ax2.set_title('Background call')
    """     
 
    # Check input data and unpack settings 
    if type(Sxx) is not np.ndarray and len(Sxx.shape) != 2: 
        raise TypeError('Sxx must be a numpy 2D array')   
     
    if type(resolution) is str: 
        opt_shape = opt_shape_presets(resolution) 
    elif type(resolution) is dict: 
        opt_shape = opt_shape_presets(resolution='custom', opt_shape=resolution) 
    else: 
        raise TypeError('Resolution must be string or a dictionary. See function documentation.') 
         
    npyr = opt_shape['npyr']
      
    # build filterbank 
    params, kernels = filter_bank_2d_nodc(ntheta=opt_shape['ntheta'], 
                                          bandwidth=opt_shape['bandwidth'], 
                                          frequency=opt_shape['frequency'], 
                                          gamma=opt_shape['gamma']) 
    # filter spectrogram 
    im_rs = filter_multires(Sxx, kernels, npyr, rescale=True)  
     
    # If rois are provided get mean intensity for each ROI,  
    # else compute mean intensity for the whole spectrogram 
    if rois is None: 
        shape = [] 
        for im in im_rs: 
            shape.append(np.mean(im)) 
        shape = [shape]  # for dataframe formating below
    else: 
        shape = np.zeros(shape=(len(rois),len(im_rs))) 
        index_pyr = 0 
        for im in im_rs: 
            index_row = 0 
            for _, row in rois.iterrows() : 
                row = pd.DataFrame(row).T 
                im_blobs = rois_to_imblobs(np.zeros(Sxx.shape), row)
                roi_mean = (im * im_blobs).sum() / im_blobs.sum() 
                shape[index_row,index_pyr]= roi_mean 
                index_row = index_row + 1 
            index_pyr = index_pyr + 1 
     
    # organise parameters 
    params_multires = _params_to_df(params, npyr) 
     
    # format shape into dataframe 
    cols=['shp_' + str(idx).zfill(3) for idx in range(1,len(shape[0])+1)] 
    shape = pd.DataFrame(data=np.asarray(shape),columns=cols) 
     
    if rois is not None:
        shape.index = rois.index
        shape = rois.join(shape)
         
    return shape, params_multires 
 
#%%
def shape_features_raw(im, resolution='low', opt_shape=None): 
    """ 
    Computes raw shape of 2D signal (image or spectrogram) at multiple resolutions  
    using 2D Gabor filters. Contrary to ``shape_features``, this function 
    delivers the raw response of the spectrogram to each filter of the filter 
    bank. This function can be used to have access to spectrograms filtered by the filterbank.
     
    Parameters 
    ---------- 
    Sxx: 2D array 
        Spectrogram to be analysed 
    resolution:  
        Resolution of analysis, i.e. number of filters used.  
        Three presets are provided, 'low', 'mid' and 'high', which control  
        the number of filters. 
    opt_shape: dictionary (optional) 
        options for the filter bank (kbank_opt) and the number of scales (npyr) 
             
    Returns 
    ------- 
    shape_raw: list 
        Raw shape response of spectrogram to every filter of the filter bank. 
        Every element of the list is the response to one of the filters. Detail  
        of each filter are available in the ``params`` DataFrame returned. 
    params: pandas DataFrame 
        Corresponding parameters of the 2D fileters used to compute the  
        shape coefficient. Params has 4 fields (theta, freq, pyr_level, scale)
    
    See also
    --------
    maad.features.shape_features, maad.features.opt_shape_presets, 
    maad.features.filter_bank_2d_nodc, maad.util.plot_shape
    
    Examples
    --------
    
    Scatter the spectrogram acoustic components using 2D Gabor filters.

    >>> from maad.sound import load, spectrogram 
    >>> from maad.features import shape_features_raw
    >>> from maad.util import power2dB, plot2d
    >>> s, fs = load('../data/spinetail.wav')
    >>> Sxx, tn, fn, ext = spectrogram(s, fs, db_range=80, display=True) 
    >>> Sxx_db = power2dB(Sxx, db_range=80)
    >>> shape_raw, params = shape_features_raw(Sxx_db, resolution='low')
    >>> plot2d(shape_raw[0], **{'extent':ext, 'title': 'High-frequency vertical component'})
    >>> plot2d(shape_raw[13], **{'extent':ext, 'title': 'Low-frequency vertical components'})
    >>> plot2d(shape_raw[2], **{'extent':ext, 'title': 'High-frequency horizontal components'})
    >>> plot2d(shape_raw[15], **{'extent':ext, 'title': 'Low-frequency horizontal components'})
    """     
     
    # unpack settings 
    opt_shape = opt_shape_presets(resolution, opt_shape) 
    npyr = opt_shape['npyr'] 
    # build filterbank 
    params, kernels = filter_bank_2d_nodc(ntheta=opt_shape['ntheta'], 
                                          bandwidth=opt_shape['bandwidth'], 
                                          frequency=opt_shape['frequency'], 
                                          gamma=opt_shape['gamma']) 
    # filter images 
    shape_raw = filter_multires(im, kernels, npyr, rescale=True)  
     
    # organise parameters 
    params_multires = _params_to_df(params, npyr) 
     
    return shape_raw, params_multires
 
#%%
def centroid_features(Sxx, rois=None, im_rois=None): 
    """ 
    Computes intensity centroid of a spectrogram. If regions of interest 
    ``rois`` are provided, the centroid is computed for each region.
    
    Parameters 
    ---------- 
    Sxx :  2D array 
        Spectrogram 
    rois: pandas DataFrame, default is None 
        Regions of interest where descriptors will be computed. Array must  
        have a valid input format with column names: ``min_t``, ``min_f``, 
        ``max_t``, and ``max_f``. Use the function ``maad.util.format_features``
        before using centroid_features to format of the ``rois`` DataFrame 
        correctly.
    im_rois: 2d ndarray 
        image with labels as values
             
    Returns 
    ------- 
    centroid: pandas DataFrame 
        Centroid of each region of interest.
        
    See Also
    --------
    maad.features.shape_features, maad.util.overlay_rois, 
    maad.util.format_features
 
    Examples
    --------
 
    Get centroid from the whole power spectrogram 
 
    >>> from maad.sound import load, spectrogram
    >>> from maad.features import centroid_features
    >>> from maad.util import (power2dB, format_features, overlay_rois, plot2d,
                               overlay_centroid)
     
    Load audio and compute spectrogram 
     
    >>> s, fs = load('../data/spinetail.wav') 
    >>> Sxx,tn,fn,ext = spectrogram(s, fs, db_range=80) 
    >>> Sxx = power2dB(Sxx, db_range=80)
     
    Load annotations and plot
    
    >>> from maad.util import read_audacity_annot
    >>> rois = read_audacity_annot('../data/spinetail.txt') 
    >>> rois = format_features(rois, tn, fn) 
    >>> ax, fig = plot2d (Sxx, extent=ext)
    >>> ax, fig = overlay_rois(Sxx,rois, extent=ext, ax=ax, fig=fig)
    
    Compute the centroid of each rois, format to get results in the 
    temporal and spectral domain and overlay the centroids.
     
    >>> centroid = centroid_features(Sxx, rois) 
    >>> centroid = format_features(centroid, tn, fn)
    >>> ax, fig = overlay_centroid(Sxx,centroid, extent=ext, ax=ax, fig=fig)
    
    """     
     
    # Check input data 
    if type(Sxx) is not np.ndarray and len(Sxx.shape) != 2: 
        raise TypeError('Sxx must be an numpy 2D array')       
     
    # check rois 
    if rois is not None: 
        if not(('min_t' and 'min_f' and 'max_t' and 'max_f') in rois): 
            raise TypeError('Array must be a Pandas DataFrame with column names: min_t, min_f, max_t, max_f. Check example in documentation.') 
     
    centroid=[] 
    area = []   
    if rois is None: 
        centroid = ndimage.center_of_mass(Sxx) 
        centroid = pd.DataFrame(np.asarray(centroid)).T 
        centroid.columns = ['centroid_y', 'centroid_x'] 
        centroid['area_xy'] = Sxx.shape[0] * Sxx.shape[1]
        centroid['duration_x'] = Sxx.shape[1]
        centroid['bandwidth_y'] = Sxx.shape[0]
    else: 
        if im_rois is not None : 
            # real centroid and area
            rprops = measure.regionprops(im_rois, intensity_image=Sxx)
            centroid = [roi.weighted_centroid for roi in rprops]
            area = [roi.area for roi in rprops]
        else:
            # rectangular area (overestimation) 
            area = (rois.max_y -rois.min_y) * (rois.max_x -rois.min_x)  
            # centroid of rectangular roi
            for _, row in rois.iterrows() : 
                row = pd.DataFrame(row).T 
                im_blobs = rois_to_imblobs(np.zeros(Sxx.shape), row)     
                rprops = measure.regionprops(im_blobs, intensity_image=Sxx)
                centroid.append(rprops.pop().weighted_centroid) 

        centroid = pd.DataFrame(centroid, columns=['centroid_y', 'centroid_x'], index=rois.index)
        
        ##### duration in number of pixels 
        centroid['duration_x'] = (rois.max_x -rois.min_x)  
        ##### bandwidth in number of pixels 
        centroid['bandwidth_y'] = (rois.max_y -rois.min_y) 
        ##### area
        centroid['area_xy'] = area      
     
        # concat rois and centroid dataframes 
        centroid = rois.join(pd.DataFrame(centroid, index=rois.index))  

    return centroid 
 
#%%     
def all_shape_features(s, fs, rois, resolution='low',  
                          display=False, **kwargs): 
    """ 
    Computes shape and central frequency features from signal at specified 
    time-frequency limits defined by regions of interest (ROIs).
     
    Parameters 
    ---------- 
    s: ndarray 
        Singal to be analysed 
    fs: int 
        Sampling frequency of the signal 
    rois: pandas DataFrame 
        Time frequency limits for the analysis. Columns should have at 
        least min_t, max_t, min_f, max_f. Can be computed with multiple 
        detection methods, such as find_rois_cwt 
    resolution: str or dict, default is 'med' 
        Specify resolution of shape descriptors. Can be: 'low', 'med', 'high'. 
        Default is 'low'. Alternatively, custom resolution can be provided  
        using a dictionary with options to define the filter bank. Valid keys  
        are: ntheta, bandwidth, frequency, gamma, npyr         
    opt_spec: dictionnary, optional 
        Options for the spectrogram with keys, window length 'nperseg', 
        window overlap in percentage 'overlap' and frequency limits 'flims' 
        which is a list of 2 scalars : minimum and maximum boundary frequency  
        values in Hertz 
    display: boolean, optional, default is False 
        Flag. If display is True, plot results 
        
    \*\*kwargs, optional. This parameter is used by plt.plot and savefig functions 
            
        - savefilename : str, optional, default :'_spectro_overlaycentroid.png' 
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
    feature_rois: pandas Dataframe 
        A dataframe with each column corresponding to a feature 
     
    Examples 
    -------- 
    >>> from maad.util import read_audacity_annot 
    >>> from maad.sound import load 
    >>> from maad.features import all_shape_features 
    >>> s, fs = load('../data/spinetail.wav')
    >>> rois = read_audacity_annot('../data/spinetail.txt')
    >>> features = all_shape_features(s, fs, rois, resolution='low', display=True)
    >>> features
        label      min_t     ...       duration_x  bandwidth_y
        0     SP   0.101385     ...              0.0          0.0
        1   CRER   0.506924     ...              0.0          0.0
        2     SP   1.203945     ...              0.0          0.0
        3     SP   2.724718     ...              0.0          0.0
        4   CRER   5.221319     ...              0.0          0.0
        5     SP   6.260514     ...              0.0          0.0
        6     SP   7.946037     ...              0.0          0.0
        7     SP   8.896519     ...              0.0          0.0
        8     SP   9.973733     ...              0.0          0.0
        9   CRER  11.329756     ...              0.0          0.0
        10    SP  11.773314     ...              0.0          0.0
        ....
    >>> features.columns
    Index(['label', 'min_t', 'min_f', 'max_t', 'max_f', 'min_y', 'min_x', 'max_y',
           'max_x', 'shp_001', 'shp_002', 'shp_003', 'shp_004', 'shp_005',
           'shp_006', 'shp_007', 'shp_008', 'shp_009', 'shp_010', 'shp_011',
           'shp_012', 'shp_013', 'shp_014', 'shp_015', 'shp_016', 'centroid_y',
           'centroid_x', 'duration_x', 'bandwidth_y', 'area_xy', 'centroid_f',
           'centroid_t', 'duration_t', 'bandwidth_f', 'area_tf'],
          dtype='object')
    """    
   
    # Spectro computing
    N=1024
    window  = kwargs.pop('window','hann')
    nperseg = kwargs.pop('nperseg',N) 
    overlap = kwargs.pop('overlap',N//2)  
    mode    = kwargs.pop('mode', 'psd')   
    fcrop   = kwargs.pop('fcrop',None) 
    tcrop   = kwargs.pop('tcrop',None)
    verbose = kwargs.pop('verbose',False) 
    # Spectro display
    ax      = kwargs.pop('ax',None)  
    fig     = kwargs.pop('fig',None)  
    vmin   = kwargs.pop('vmin',-120) 
    vmax   = kwargs.pop('vmax',-20)
    
    Sxx, tn, fn, ext = spectrogram(s, fs, window, nperseg, overlap, fcrop,
                                   tcrop, mode, verbose, display=False, **kwargs) 
    
    # dB scale
    Sxx = 10*np.log10(Sxx)
     
    # format rois to bbox 
    rois = format_features(rois, tn, fn) 
    
    # if rois becomes empty aveter format_features
    if rois.empty :
        print ('rois is empty')
        features = rois
        return features
    else:
        print('number of rois : %d' % len(rois))
     
    # Compute shape features and centroid features 
    shape, params = shape_features(Sxx, resolution, rois) 
    shape = format_features(shape, tn, fn)
    if verbose : 
        print('Dataframe with shapes features')
        print(shape)
        
    centroid = centroid_features(Sxx, rois) 
    centroid = format_features(centroid, tn, fn)
    if verbose : 
        print('Dataframe with centroids features')
        print(centroid)
    
    if display :
        # view bbox
        ax, fig = overlay_rois(Sxx, rois, vmin=vmin, vmax=vmax, **kwargs)
        # view centroids
        overlay_centroid(Sxx, centroid, savefig=None, 
                         fig=fig, ax=ax, **kwargs)
        # plot shape
        plot_shape(shape.mean(), params)
    
    # combine into a single df without columns duplication
    features = pd.concat([rois, shape, centroid], axis=1, sort=False)
    features = features.loc[:,~features.columns.duplicated()]
    
    if verbose : 
        print('list of features')
        print(list(features))
     
    return features 
 