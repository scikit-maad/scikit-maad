#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble of functions to compute acoustic descriptors from 2D spectrograms

"""

# Load required modules
from __future__ import print_function
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import itertools as it
import matplotlib.pyplot as plt
from skimage.io import imsave 
from skimage import transform, measure
from scipy import ndimage
from maad import sound
from skimage.filters import gaussian
from maad.util import format_rois, rois_to_imblobs, normalize_2d


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
    >>> s, fs = load('./data/spinetail.wav')
    >>> Sxx, dt, df, ext = spectrogram(s, fs, db_range=120, display=True)
    >>> params, kernels = filter_bank_2d_nodc(frequency=(0.5, 0.25), ntheta=2,gamma=2)
    >>> Sxx_out = filter_multires(Sxx, kernels, npyr=2)
    
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

        
def filter_bank_2d_nodc(frequency, ntheta, bandwidth=1, gamma=2, display=False, **kwargs):
    """
    Build an ensemble of complex 2D Gabor filters with no DC offset.
    
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
    
    Examples
    --------

    It is possible to load presets to build the filter bank using predefined 
    parameters with the function maad.features.opt_shape_presets

    >>> from maad.features import filter_bank_2d_nodc, opt_shape_presets
    >>> opt = opt_shape_presets(resolution='med')
    >>> params, kernels = filter_bank_2d_nodc(opt['frequency'], opt['ntheta'], opt['bandwidth'], opt['gamma'], display=True)

    Alternatively, custom parameters can be provided to define the filter bank

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


def shape_features(Sxx, resolution='low', rois=None, dt=None, df=None):
    """
    Computes time-frequency shape descriptors at multiple resolutions using 2D Gabor filters
    
    Parameters
    ----------
    Sxx: 2D array
        Input image to process 
    resolution: str or dict
        Specify resolution of shape descriptors. Can be: 'low', 'med', 'high'.
        Default is 'low'. Alternatively, custom resolution can be provided 
        using a dictionary with options to define the filter bank. Valid keys 
        are: ntheta, bandwidth, frequency, gamma, npyr
    rois: pandas DataFrame
        Regions of interest where descriptors will be computed. Array must 
        have a valid input format with column names: min_t min_f, max_t, max_f
    dt: 1D array
        Vector of time instants that can be used as x coordinates for the spectrogram
    df: 1D array
        Vector of frequencies that can be used as y coordinates for the spectrogram
    opt_shape: dictionary
        options for the filter bank (kbank_opt) and the number of scales (npyr)
            
    Returns
    -------
    shape: 1D array
        Shape coeficients of each filter
    params: 2D numpy structured array
        Corresponding parameters of the 2D fileters used to calculate the 
        shape coefficient. Params has 4 fields (theta, freq, pyr_level, scale)
    
    Notes
    -----
    Overlapping regions of interest (ROIs) will be combined in the workflow. 
    Identify and process separately these ROIs and then combine the output. 
    Future versions of this function will facilitate the use of overlapping ROIs.
    
    Examples
    --------

    Get shape features from the whole power spectrogram

    >>> from maad.sound import load, spectrogram
    >>> from maad.features import shape_features
    >>> from maad.util import linear2dB
    >>> s, fs = load('./data/spinetail.wav')
    >>> Sxx, dt, df, ext = spectrogram(s, fs, db_range=100, display=True)
    >>> Sxx_db = linear2dB(Sxx, db_range=100)
    >>> shape, params = shape_features(Sxx_db, resolution='med')
    
    Or get shape features from specific regions of interest
    
    >>> from maad.util import read_audacity_annot
    >>> rois_tf = read_audacity_annot('./data/spinetail.txt')
    >>> rois = rois_tf.loc[rois_tf.label=='CRER',]  
    >>> shape, params = shape_features(Sxx_db, resolution='med', rois=rois, dt=dt, df=df)
    
    """    
    # TODO: 
    #    - output of Rois has some incertitudes associated to the spectrogram

    # Check input data and unpack settings
    if type(Sxx) is not np.ndarray and len(Sxx.shape) != 2:
        raise TypeError('Sxx must be an numpy 2D array')  
    
    if type(resolution) is str:
        opt_shape = opt_shape_presets(resolution)
    elif type(resolution) is dict:
        opt_shape = opt_shape_presets(resolution='custom', opt_shape=resolution)
    else:
        raise TypeError('Resolution must be string or a dictionary. See function documentation.')
        
    npyr = opt_shape['npyr']

    # transform ROIs to im_blobs
    if rois is not None:
        if ~(pd.Series(['min_t', 'min_f', 'max_t', 'max_f']).isin(rois.columns).all()):
            raise TypeError('Array must be a Pandas DataFrame with column names: min_t, min_f, max_t, max_f. Check example in documentation.')
        rois_bbox = format_rois(rois, dt, df, fmt='bbox')
        im_blobs = rois_to_imblobs(np.zeros(Sxx.shape), rois_bbox)
    else:
        im_blobs = None
    
    # build filterbank
    params, kernels = filter_bank_2d_nodc(ntheta=opt_shape['ntheta'],
                                          bandwidth=opt_shape['bandwidth'],
                                          frequency=opt_shape['frequency'],
                                          gamma=opt_shape['gamma'])
    # filter spectrogram
    im_rs = filter_multires(Sxx, kernels, npyr, rescale=True) 
    
    # If ROIs are provided get mean intensity for each ROI, 
    # else compute mean intensity for the whole spectrogram
    shape = []
    if im_blobs is None:
        for im in im_rs:
            shape.append(np.mean(im))
            rois_bbox=None
        shape = [shape]  # for dataframe formating below
    else:
        for im in im_rs:
            labels = measure.label(im_blobs)
            rprops = measure.regionprops(labels, intensity_image=im)
            roi_mean = [roi.mean_intensity for roi in rprops]
            shape.append(roi_mean)
        rois_bbox = [roi.bbox for roi in rprops]
        shape = list(map(list, zip(*shape)))  # transpose shape
    
    # organise parameters
    params_multires = _params_to_df(params, npyr)
    
    # format shape into dataframe
    cols=['shp_' + str(idx).zfill(3) for idx in range(1,len(shape[0])+1)]
    shape = pd.DataFrame(data=np.asarray(shape),columns=cols)
    
    if rois is not None:
        # format rois into dataframe
        rois_bbox = pd.DataFrame(rois_bbox, columns=['min_y','min_x',
                                                     'max_y','max_x'])
        # compensate half-open interval of bbox from skimage
        rois_bbox.max_y = rois_bbox.max_y - 1
        rois_bbox.max_x = rois_bbox.max_x - 1
        
        # combine output
        rois_out = format_rois(rois_bbox, dt, df, fmt='tf')
        shape = pd.concat([rois_out, shape], axis='columns')
        
    return shape, params_multires


def centroid(im, im_blobs=None):
    """
    Computes intensity centroid of the 2D signal (usually time-frequency representation) 
    along a margin, frequency (0) or time (1).
    
    Parameters
    ----------
    im: 2D array
        Input image to process 
    im_blobs: 2D array, optional
        Optional binary array with '1' on the region of interest and '0' otherwise
    margin: 0 or 1
        Margin of the centroid, frequency=1, time=0
            
    Returns
    -------
    centroid: 1D array
        centroid of image. If im_blobs provided, centroid for each region of interest

    """    
    centroid=[]
    rois_bbox=[]
    if im_blobs is None:
        centroid = ndimage.center_of_mass(im)
    else:
        labels = measure.label(im_blobs)
        rprops = measure.regionprops(labels, intensity_image=im)
        centroid = [roi.weighted_centroid for roi in rprops]
        rois_bbox = [roi.bbox for roi in rprops]

    # variables to dataframes
    centroid = pd.DataFrame(centroid, columns=['y', 'x'])
    rois_bbox = pd.DataFrame(rois_bbox, columns=['min_y','min_x',
                                                 'max_y','max_x'])
    # compensate half-open interval of bbox from skimage
    rois_bbox.max_y = rois_bbox.max_y - 1
    rois_bbox.max_x = rois_bbox.max_x - 1
    
    return rois_bbox, centroid


def create_csv(shape_features, centroid_features, label_features = None, 
               display=False):
    """
    Create a .csv file containing all the features (shapes, centroids and 
    labels)
        
    Parameters
    ----------
    shape_features : 2d nd array of scalars
        Each column corresponds to a shape (linked to a kernel filter) 
        Each row corresponds to a ROI
        
    centroid_features: 2d nd array of scalars (centroid in freq and time)
        Centroid of image. If labels provided, centroid for each ROI (rows)
        - column 0 is 'cyear' 
        - column 1 is 'cmonth' 
        - column 2 is 'chour' 
        - column 3 is 'cminute'
        - column 4 is 'csecond' 
        - column 5 is 'cfreq' 
        
    label_features: 2d nd array of integers and strings, optional, default is 
    None
        - column 0 is 'labelID'
        - column 1 is 'labelName'
        
        Each row corresponds to a ROI
    
    Returns
    -------
    table : dataframe (Pandas)
        The table contains all the features extracted from the spectrogram    
    """
    if label_features is not None:
        table_label_features = pd.DataFrame({'labelID' : np.asarray(label_features)[:,0],
                                             'labelName' : np.asarray(label_features)[:,1]})  
        
    table_shape_features = pd.DataFrame(data=shape_features,
                        columns=["shp" + str(i) for i in range(1,len(shape_features[0])+1)])
    
    table_centroid_features = pd.DataFrame({'cyear' : centroid_features[:,0],
                                            'cmonth': centroid_features[:,1], 
                                            'cday'  : centroid_features[:,2], 
                                            'chour' : centroid_features[:,3], 
                                            'cminute': centroid_features[:,4], 
                                            'csecond': centroid_features[:,5], 
                                            'cfreq' : centroid_features[:,6]})
    if label_features is not None:
        table = pd.concat([table_label_features, table_centroid_features, table_shape_features], axis=1)
    else:
        table = pd.concat([table_centroid_features, table_shape_features], axis=1)
        
    if display:
        # -------------   FEATURES VIZUALIZATION WITH PANDAS   ----------------
        # table with a summray of the features value
        table.describe()
 
        # histograpm for each features
        table.hist(bins=50, figsize=(15,15))
        plt.show()    
        
    return table

def save_csv(filename, shape_features, centroid_features, label_features=None,
             mode='w'):
    """
    Create and save a .csv file containing all the features (shapes, centroids 
    and labels)
        
    Parameters
    ----------
    filename : string
        full name (path and name) of the .csv file
        
    mode : string, optional, default is 'w'
        Python write mode. For example
        'w'  Truncate file to zero length or create text file for writing.
        The stream is positioned at the beginning of the file.

        'a'  Open for writing. The file is created if it does not exist. The
        stream is positioned at the end of the file.  Subsequent writes
        to the file will always end up at the then current end of file,
        irrespective of any intervening fseek(3) or similar.
        
    shape_features : 2d nd array of scalars
        Each column corresponds to a shape (linked to a kernel filter) 
        Each row corresponds to a ROI
        
    centroid_features: 2d nd array of scalars (centroid in freq and time)
        Centroid of image. If labels provided, centroid for each ROI (rows)
        - column 0 is 'cyear' 
        - column 1 is 'cmonth' 
        - column 2 is 'chour' 
        - column 3 is 'cminute'
        - column 4 is 'csecond' 
        - column 5 is 'cfreq' 
        
    label_features: 2d nd array of integers and strings, optional, default is 
    None
        - column 0 is 'labelID'
        - column 1 is 'labelName'
   
    Returns
    -------
    table : dataframe (Pandas)
        The table contains all the features extracted from the spectrogram.
        Keys are {'labelID', 'labelName, 'cyear', 'cmonth', 'cday', 'chour',
        'cmin','csecond','cfreq','shp1,'shp2',...'shpn'}
    """
    table = create_csv(shape_features, centroid_features, label_features)
    table.to_csv(path_or_buf=filename,sep=',',mode=mode,header=True, index=False)
    return table


def get_features_wrapper(im, ext, display=False, savefig=None, save_csv=None, 
                         **kwargs):
    """
    Computes shape of 2D signal (image or spectrogram) at multiple resolutions 
    using 2D Gabor filters
    
    Parameters
    ----------
    im: 2D array
        Input image to process (spectrogram)

    ext : list of scalars [left, right, bottom, top], optional, default: None
        The location, in data-coordinates, of the lower-left and
        upper-right corners. If `None`, the image is positioned such that
        the pixel centers fall on zero-based (row, column) indices. 
        
    display : boolean, optional, default is False
        Display the signal if True
        
    savefig : string, optional, default is None
        Root filename (with full path) is required to save the figures. Postfix
        is added to the root filename.
        
    save_csv : string, optional, default is None
        Root filename (with full path) is required to save the table. Postfix
        is added to the root filename.
        
    \*\*kwargs, optional. This parameter is used by plt.plot and savefig functions
        
        - figsize : tuple of integers,
            width, height in inches.  
            
        - title : string, 
            title of the figure
        
        - xlabel : string, optional, 
            label of the horizontal axis
        
        - ylabel : string, optional, 
            label of the vertical axis
        
        - cmap : string or Colormap object, 
            See https://matplotlib.org/examples/color/colormaps_reference.html
            in order to get all the  existing colormaps
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic', 
            'viridis'...
        
        - vmin, vmax : scalar
            `vmin` and `vmax` are used in conjunction with norm to normalize
            luminance data.  Note if you pass a `norm` instance, your
            settings for `vmin` and `vmax` will be ignored.
        
        - ext : scalars (left, right, bottom, top),
            The location, in data-coordinates, of the lower-left and
            upper-right corners. If `None`, the image is positioned such that
            the pixel centers fall on zero-based (row, column) indices.
        
        - dpi : integer, optional
            Dot per inch. 
            For printed version, choose high dpi (i.e. dpi=300) => slow
            For screen version, choose low dpi (i.e. dpi=96) => fast
        
        - format : string, optional
            Format to save the figure
            
        ... and more, see matplotlib      
    
    Returns
    -------
    table : dataframe (Pandas)
        The table contains all the features extracted from the spectrogram.
        Keys are {'labelID', 'labelName, 'cyear', 'cmonth', 'cday', 'chour',
        'cmin','csecond','cfreq','shp0,'shp1',...'shpn'}
        
    params_shape: 2D numpy structured array
        Parameters used to calculate 2D gabor kernels.  
        params_shape has 5 fields (theta, freq, bandwidth, gamma, pyr_level)
        Each row corresponds to a shape (shp1, shp2...shpn)
    """   
       
    freq=kwargs.pop('freq',(0.75, 0.5))
    ntheta=kwargs.pop('ntheta',2)
    bandwidth=kwargs.pop('bandwidth', 1)
    gamma=kwargs.pop('gamma', 1)
    npyr=kwargs.pop('npyr', 3)  
    date=kwargs.pop('date', None) 
    im_rois=kwargs.pop('im_rois', None)  
    label_features=kwargs.pop('label_features', None) 
       
    params, kernels = filter_bank_2d_nodc(frequency=freq, ntheta=ntheta, 
                                          bandwidth=bandwidth,gamma=gamma, 
                                          display=display, savefig=savefig)
    
    # multiresolution image filtering (Gaussian pyramids)
    im_filtlist = filter_multires(im, ext, kernels, params, npyr=npyr, 
                                  display=display, savefig=savefig)
    
    # Extract shape features for each roi
    params_shape, shape = shape_features(im_filtlist=im_filtlist, 
                                          params = params, 
                                          im_rois=im_rois)
    # Extract centroids features for each roi
    centroid_features = centroid(im=im, ext=ext, date=date, im_rois=im_rois)
    
    if save_csv :
        table = save_csv(save_csv+'.csv', 
                         shape, centroid_features, label_features,
                         display=display)
    else:
        table = create_csv(shape, centroid_features, label_features,
                           display=display)
    
    return table, params_shape 


def save_figlist(fname, figlist):
    """
    Save a list of figures to file.
    
    Parameters
    ----------
    fname: string
        suffix name to save the figure. Extension indicates the format 
        of the image

    Returns
    -------
    Nothing
        
    """
    for i, fig in enumerate(figlist):
        fname_save='%d_%s' % (i, fname)
        imsave(fname_save,fig)


def opt_shape_presets(resolution, opt_shape=None):
    """ 
    Set parameters for multiresolution analysis using presets or custom parameters
    
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
        
    Examples
    --------
    
    Get parameters using predefined presets
    
    >>> from maad.features import opt_shape_presets
    >>> opt = opt_shape_presets('med')
        
    Get parameters to analyse at high shape resolution
    
    >>> from maad.features import opt_shape_presets
    >>> opt = opt_shape_presets('high')
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


def plot_shape(shape, params, row=0, display_values=False):
    """
    Plot shape features in a bidimensional plot
    
    Parameters
    ----------
    shape: 1D array, pd.Series or pd.DataFrame
        Shape features computed with shape_features function.
    
    params: pd.DataFrame
        Pandas dataframe returned by maad.features_rois.shape_features
    
    row: int
        Observation to be visualized
    
    display_values: bool
        Set to True to display the coefficient values. Default is False.
    
    Returns
    -------
    ax: matplotlib.axes
        Axes of the figure
        
    Examples
    --------
    >>> from maad.sound import load, spectrogram
    >>> from maad.features import shape_features, plot_shape
    >>> import numpy as np
    >>> s, fs = load('./data/spinetail.wav')
    >>> Sxx, ts, f, ext = spectrogram(s, fs)
    >>> shape, params = shape_features(np.log10(Sxx), ts, f, resolution='high')
    >>> plot_shape(shape, params)

    """

    # compute shape of matrix
    dirs_size = params.theta.unique().shape[0]
    scale_size = np.unique(params.freq).size * np.unique(params.pyr_level).size
    # reshape feature vector
    idx = params.sort_values(['theta','pyr_level','scale']).index
    
    if isinstance(shape, pd.DataFrame):
        shape_plt = shape.iloc[:,shape.columns.str.startswith('shp')]
        shape_plt = np.reshape(shape_plt.iloc[row,idx].values, (dirs_size, scale_size))
    elif isinstance(shape, pd.Series):
        shape_plt = shape.iloc[shape.index.str.startswith('shp')]
        shape_plt = np.reshape(shape_plt.iloc[idx].values, (dirs_size, scale_size))
    elif isinstance(shape, np.ndarray):
        shape_plt = np.reshape(shape_plt[idx], (dirs_size, scale_size))

    
    unique_scale = params.scale[idx] * 2**params.pyr_level[idx]
    # get textlab
    textlab = shape_plt
    textlab = np.round(textlab,2)
    
    # plot figure
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.imshow(shape_plt, aspect='auto', origin='lower', interpolation='None', cmap='viridis')
    if display_values:    
        for (j,i),label in np.ndenumerate(textlab):
            ax.text(i,j,label,ha='center',va='center')
        
    yticklab = np.round(params.theta.unique(),1)
    xticklab = np.reshape(unique_scale.values, 
                          (dirs_size, scale_size))
    ax.set_xticks(np.arange(scale_size))
    ax.set_xticklabels(np.round(xticklab,1)[0,:])
    ax.set_yticks(np.arange(dirs_size))
    ax.set_yticklabels(yticklab)
    ax.set_xlabel('Scale')
    ax.set_ylabel('Theta')
    plt.show()
    
    return ax


def compute_rois_features(s, fs, rois_tf, opt_spec, opt_shape, flims):
    """
    Computes shape and central frequency features from signal at specified
    time-frequency limits defined by regions of interest (ROIs)
    
    Parameters
    ----------
    s: ndarray
        Singal to be analysed
    fs: int
        Sampling frequency of the signal
    rois_tf: pandas DataFrame
        Time frequency limits for the analysis. Columns should have at
        least min_t, max_t, min_f, max_f. Can be computed with multiple
        detection methods, such as find_rois_cwt
    opt_spec: dictionnary
        Options for the spectrogram with keys, window lenght 'nperseg' and,
        window overlap in percentage 'overlap'
    opt_shape: dictionary
        Options for the filter bank (kbank_opt) and the number of scales (npyr)
    flims: list of 2 scalars
        Minimum and maximum boundary frequency values in Hertz
    
    Returns
    -------
    feature_rois: pandas Dataframe
        A dataframe with each column corresponding to a feature
    
    Examples
    --------
    >>> s, fs = sound.load('spinetail.wav')        
    >>> rois_tf = find_rois_cwt(s, fs, flims=(3000, 8000), tlen=2, th=0.003)
    >>> opt_spec = {'nperseg': 512, 'overlap': 0.5}
    >>> opt_shape = opt_shape_presets('med')
    >>> features_rois = compute_rois_features(s, fs, rois_tf, opt_spec, 
    opt_shape, flims)
        
    """
    im, dt, df, ext = sound.spectrogram(s, fs, nperseg=opt_spec['nperseg'], 
                                        overlap=opt_spec['overlap'], fcrop=flims, 
                                        rescale=False, db_range=opt_spec['db_range'])

    # format rois to bbox
    ts = np.arange(ext[0], ext[1], dt)
    f = np.arange(ext[2],ext[3]+df,df)
    rois_bbox = format_rois(rois_tf, ts, f, fmt='bbox')
        
    # roi to image blob
    im_blobs = rois_to_imblobs(np.zeros(im.shape), rois_bbox)
    
    # get features: shape, center frequency
    im = normalize_2d(im, 0, 1)
    #im = gaussian(im) # smooth image
    bbox, params, shape = shape_features(im, im_blobs, resolution='custom', 
                                         opt_shape=opt_shape)
    _, cent = centroid(im, im_blobs)
    cent['frequency']= f[round(cent.y).astype(int)]  # y values to frequency
    
    # format rois to time-frequency
    rois_out = format_rois(bbox, ts, f, fmt='tf')
    
    # combine into a single df
    rois_features = pd.concat([rois_out, shape, cent.frequency], axis=1)
    return rois_features

def shape_features_raw(im, resolution='low', opt_shape=None):
    """
    Computes raw shape of 2D signal (image or spectrogram) at multiple resolutions 
    using 2D Gabor filters. Contrary to shape_feature, this function delivers the raw
    response of the spectrogram to each filter of the filter bank.
    
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
        of each filter are available in the param DataFrame returned.
    params: pandas DataFrame
        Corresponding parameters of the 2D fileters used to calculate the 
        shape coefficient. Params has 4 fields (theta, freq, pyr_level, scale)
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
    params = np.asarray(params)
    orient = params[:,0]*180/np.pi
    orient = orient.tolist()*npyr
    pyr_level = np.sort(np.arange(npyr).tolist()*len(params))+1
    freq = params[:,1].tolist()*npyr
    nparams = len(params)*npyr
    params_multires = np.zeros(nparams, dtype={'names':('theta', 'freq', 'pyr_level','scale'),
                                               'formats':('f8', 'f8', 'f8','f8')})
    params_multires['theta'] = orient
    params_multires['freq'] = freq
    params_multires['scale'] = 1/np.asarray(freq)
    params_multires['pyr_level'] = pyr_level
    params_multires = pd.DataFrame(params_multires)
    
    return params_multires, shape_raw

