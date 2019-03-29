#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 High level functions for multiresolution analysis of spectrograms
 Code licensed under both GPL and BSD licenses
 Authors:  Juan Sebastian ULLOA <jseb.ulloa@gmail.com>
           Sylvain HAUPERT <sylvain.haupert@mnhn.fr>

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
from maad.util import format_rois, rois_to_imblobs, normalize_2d


def _sigma_prefactor(bandwidth):
    """
    Function from skimage. 
    
    Parameters
    ----------

    Returns
    -------


    """

    b = bandwidth
    # See http://www.cs.rug.nl/~imaging/simplecell.html
    return 1.0 / np.pi * np.sqrt(np.log(2) / 2.0) * \
        (2.0 ** b + 1) / (2.0 ** b - 1)

def gabor_kernel_nodc(frequency, theta=0, bandwidth=1, gamma=1,
                      n_stds=3, offset=0):
    """
    Return complex 2D Gabor filter kernel with no DC offset.
    
    This function is a modification of the gabor_kernel function of scikit-image
    
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
        the kernel *before* rotation. If `theta = pi/2`, then the kernel is
        rotated 90 degrees so that `sigma_x` controls the *vertical* direction.   
    n_stds : scalar, optional
        The linear size of the kernel is n_stds (3 by default) standard
        deviations
    offset : float, optional
        Phase offset of harmonic function in radians.
    Returns
    -------
    g_nodc : complex 2d array
        A single gabor kernel (complex) with no DC offset
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Gabor_filter
    .. [2] http://mplab.ucsd.edu/tutorials/gabor.pdf
    Examples
    --------
    >>> from skimage.filters import gabor_kernel
    >>> from skimage import io
    >>> from matplotlib import pyplot as plt  # doctest: +SKIP
    >>> gk = gabor_kernel(frequency=0.2)
    >>> plt.figure()        # doctest: +SKIP
    >>> io.imshow(gk.real)  # doctest: +SKIP
    >>> io.show()           # doctest: +SKIP
    >>> # more ripples (equivalent to increasing the size of the
    >>> # Gaussian spread)
    >>> gk = gabor_kernel(frequency=0.2, bandwidth=0.1)
    >>> plt.figure()        # doctest: +SKIP
    >>> io.imshow(gk.real)  # doctest: +SKIP
    >>> io.show()           # doctest: +SKIP
    """
    
     # set gaussian parameters
    b = bandwidth
    sigma_pref = 1.0 / np.pi * np.sqrt(np.log(2) / 2.0) * (2.0 ** b + 1) / (2.0 ** b - 1)
    sigma_y = sigma_pref / frequency
    sigma_x = sigma_y/gamma
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


def _plot_filter_bank(kernels, frequency, ntheta, bandwidth, gamma, **kwargs):
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
    
    bandwidth: scalar, optional, default is 1
        This parameter modifies the frequency of the Gabor filter
    
    gamma: scalar, optional, default is 1
        This parameter change the Gaussian window that modulates the continuous
        sine.
        1 => same gaussian window in x and y direction (circle)
        <1 => elongation of the filter size in the y direction (elipsoid)
        >1 => reduction of the filter size in the y direction (elipsoid)
            
    **kwargs, optional. This parameter is used by plt.plot and savefig functions
        figsize : tuple of integers, optional, default: (13,13)
            width, height in inches.  
        dpi : integer, optional
            Dot per inch. 
            For printed version, choose high dpi (i.e. dpi=300) => slow
            For screen version, choose low dpi (i.e. dpi=96) => fast
        interpolation : string, optional, default is 'nearest'
            Pixels interpolation
        aspect : string, optional, default is 'auto'
        fontsize : scalar, optional, default is 8/0.22*hmax*100/dpi)
            size of the font use to print the parameters of each filter
            
        ... and more, see matplotlib        
    Returns
    -------
    fig : Figure
        The Figure instance 
    ax : Axis
        The Axis instance
    """

    params = []
    for theta in range(ntheta):
        theta = theta/ntheta * np.pi
        for freq in frequency:
            params.append([freq, theta, bandwidth, gamma])

    w = []
    h = []
    for kernel in kernels:
        ylen, xlen = kernel.shape
        w.append(xlen)
        h.append(ylen)
        
    plt.gray()
    fig = plt.figure()
    
    dpi =kwargs.pop('dpi',fig.get_dpi())
    figsize =kwargs.pop('figsize',(13,13))
    interpolation =kwargs.pop('interpolation','nearest')
    aspect =kwargs.pop('aspect','auto')
    
    
    fig.set_figwidth(figsize[0])        
    fig.set_figheight(figsize[1])
    
    w = np.asarray(w)/dpi
    h = np.asarray(h)/dpi
    wmax = np.max(w)*1.25
    hmax = np.max(h)*1.05
    
    fontsize =kwargs.pop('fontsize',8/0.22*hmax*100/dpi) 

    params_label = []
    for param in params:
        params_label.append('theta=%d f=%.2f \n bandwidth=%.1f \n gamma=%.1f' 
                            % (param[1] * 180 / np.pi, param[0], param[2],
                               param[3]))
    
    n = len(frequency)
    
    for ii, kernel in enumerate(kernels):
        ax = plt.axes([(ii%n)*wmax + (wmax-w[ii])/2,(ii//n)*hmax + (hmax-h[ii])/2,w[ii],h[ii]])
        ax.imshow(np.real(kernel),interpolation=interpolation, aspect =aspect, **kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(params_label[ii],fontsize=fontsize)
        ax.axis('tight')
    plt.show()
    
    return ax, fig
    
def _plot_filter_results(im_ref, im_list, kernels, params, m, n):
    """
    Display the result after filtering
    
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


def filter_mag(im, kernel):
    """
    Normalizes the image and computes im and real part of filter response using 
    the complex kernel and the modulus operation
    
    Parameters
    ----------
        im: 2D array
            Input image to process 
        kernel: 2D array
            Complex kernel (or filter)
            
    Returns
    -------
        im_out: Modulus operand on filtered image

    """    
    
    im = (im - im.mean()) / im.std()
    im_out = np.sqrt(ndi.convolve(im, np.real(kernel), mode='reflect')**2 +
                   ndi.convolve(im, np.imag(kernel), mode='reflect')**2)
    return im_out

def filter_multires(im_in, kernels, npyr=4, rescale=True):
    """
    Computes 2D wavelet coefficients at multiple octaves/pyramids
    
    Parameters
    ----------
        im_in: list of 2D arrays
            List of input images to process 
        kernels: list of 2D arrays
            List of 2D wavelets to filter the images
        npyr: int
            Number of pyramids to compute
        rescale: boolean
            Indicates if the reduced images should be rescaled
            
    Returns
    -------
        im_out: list of 2D arrays
            List of images filtered by each 2D kernel
    """    

    # Downscale image using gaussian pyramid
    if npyr<2:
        print('Warning: npyr should be int and larger than 2 for multiresolution')
        im_pyr = tuple(transform.pyramid_gaussian(im_in, downscale=2, 
                                                  max_layer=1, multichannel=False)) 
    else:    
        im_pyr = tuple(transform.pyramid_gaussian(im_in, downscale=2, 
                                                  max_layer=npyr-1, multichannel=False)) 

    # filter 2d array at multiple resolutions using gabor kernels
    im_filt=[]
    for im in im_pyr:  # for each pyramid
        for kernel, param in kernels:  # for each kernel
            im_filt.append(filter_mag(im, kernel))  #  magnitude response of filter
    
    # Rescale image using gaussian pyramid
    if rescale:
        dims_raw = im_in.shape
        im_out=[]
        for im in im_filt:
            ratio = np.array(dims_raw)/np.array(im.shape)
            if ratio[0] > 1:
                im = transform.rescale(im, scale = ratio, mode='reflect',
                                       multichannel=False, anti_aliasing=True)
            else:
                pass
            im_out.append(im)
    else:
        pass

    return im_out

        


def filter_bank_2d_nodc(frequency, ntheta, bandwidth=1, gamma=1, display=False, 
                        savefig=None, **kwargs):
    """
    Build a Gabor filter bank with no offset component
    
    Parameters
    ----------
    frequency: 1d ndarray of scalars
        Spatial frequencies used to built the Gabor filters. Values should be
        in [0;1]
    
    ntheta: int
        Number of angular steps between 0° to 90°
    
    bandwidth: scalar, optional, default is 1
        This parameter modifies the frequency of the Gabor filter
    
    gamma: scalar, optional, default is 1
        This parameter change the Gaussian window that modulates the continuous
        sine.
        1 => same gaussian window in x and y direction (circle)
        <1 => elongation of the filter size in the y direction (elipsoid)
        >1 => reduction of the filter size in the y direction (elipsoid)

    Returns
    -------
    params: 2d structured array
         Parameters used to calculate 2D gabor kernels. 
         Params array has 4 fields (theta, freq, bandwidth, gamma)
            
    kernels: 2d ndarray of scalars
         Gabor kernels
    """
    
    theta = np.arange(ntheta)
    theta = theta / ntheta * np.pi
    params=[i for i in it.product(theta,frequency)]
    kernels = []
    for param in params:
        kernel = gabor_kernel_nodc(frequency=param[1],
                                   theta=param[0],
                                   bandwidth=bandwidth,
                                   gamma=gamma,
                                   offset=0,
                                   n_stds=3)
        kernels.append((kernel, param))
            
    if display: 
        _, fig = _plot_filter_bank(kernels, frequency, ntheta, bandwidth, 
                                   gamma, **kwargs)
        if savefig is not None : 
            dpi   =kwargs.pop('dpi',96)
            format=kwargs.pop('format','png')               
            filename = savefig+'_filter_bank2D.'+format
            fig.savefig(filename, bbox_inches='tight', dpi=dpi, format=format,
                        **kwargs) 
            
    return params, kernels  


def shape_features(im, im_blobs=None, resolution='low', opt_shape=None):
    """
    Computes shape of 2D signal (image or spectrogram) at multiple resolutions 
    using 2D Gabor filters
    
    Parameters
    ----------
        im: 2D array
            Input image to process 
        im_blobs: 2D array, optional
            Optional binary array with '1' on the region of interest and '0' otherwise
        opt: dictionary
            options for the filter bank (kbank_opt) and the number of scales (npyr)
            
    Returns
    -------
        shape: 1D array
            Shape coeficients of each filter
        params: 2D numpy structured array
            Corresponding parameters of the 2D fileters used to calculate the 
            shape coefficient. Params has 4 fields (theta, freq, pyr_level, scale)
        bbox: 
            If im_blobs provided, corresponding bounding box
    """    
    # unpack settings
    opt_shape = opt_shape_presets(resolution, opt_shape=opt_shape)
    npyr = opt_shape['npyr']
    # build filterbank
    params, kernels = filter_bank_2d_nodc(ntheta=opt_shape['ntheta'],
                                          bandwidth=opt_shape['bandwidth'],
                                          frequency=opt_shape['frequency'],
                                          gamma=opt_shape['gamma'])
    # filter images
    im_rs = filter_multires(im, kernels, npyr, rescale=True) 

    # Get mean intensity
    shape = []
    if im_blobs is None:
        for im in im_rs:
            shape.append(np.mean(im))
            rois_bbox=None
    else:
        for im in im_rs:
            labels = measure.label(im_blobs)
            rprops = measure.regionprops(labels, intensity_image=im)
            roi_mean = [roi.mean_intensity for roi in rprops]
            shape.append(roi_mean)
        rois_bbox = [roi.bbox for roi in rprops]
        shape = list(map(list, zip(*shape)))  # transpose shape
    
    # organise parameters
    params = np.asarray(params)
    orient = params[:,0]*180/np.pi
    orient = orient.tolist()*npyr
    pyr_level = np.sort(np.arange(npyr).tolist()*len(params))+1
    freq = params[:,1].tolist()*npyr
    #params_multires = np.vstack((np.asarray(orient), freq, pyr_level))
    nparams = len(params)*npyr
    params_multires = np.zeros(nparams, dtype={'names':('theta', 'freq', 'pyr_level','scale'),
                                               'formats':('f8', 'f8', 'f8','f8')})
    params_multires['theta'] = orient
    params_multires['freq'] = freq
    params_multires['scale'] = 1/np.asarray(freq)
    params_multires['pyr_level'] = pyr_level
    params_multires = pd.DataFrame(params_multires)
    
    # format shape into dataframe
    cols=['shp' + str(idx) for idx in range(1,len(shape[0])+1)]
    shape = pd.DataFrame(data=np.asarray(shape),columns=cols)
    
    # format rois into dataframe
    rois_bbox = pd.DataFrame(rois_bbox, columns=['min_y','min_x',
                                                 'max_y','max_x'])
    # compensate half-open interval of bbox from skimage
    rois_bbox.max_y = rois_bbox.max_y - 1
    rois_bbox.max_x = rois_bbox.max_x - 1
    
    return rois_bbox, params_multires, shape


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
        column 0 is 'cyear' 
        column 1 is 'cmonth' 
        column 2 is 'chour' 
        column 3 is 'cminute'
        column 4 is 'csecond' 
        column 5 is 'cfreq' 
        
    label_features: 2d nd array of integers and strings, optional, default is 
    None
        column 0 is 'labelID'
        column 1 is 'labelName'
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
        column 0 is 'cyear' 
        column 1 is 'cmonth' 
        column 2 is 'chour' 
        column 3 is 'cminute'
        column 4 is 'csecond' 
        column 5 is 'cfreq' 
        
    label_features: 2d nd array of integers and strings, optional, default is 
    None
        column 0 is 'labelID'
        column 1 is 'labelName'
   
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
        
    **kwargs, optional. This parameter is used by plt.plot and savefig functions
        figsize : tuple of integers,
            width, height in inches.  
        title : string, 
            title of the figure
        xlabel : string, optional, 
            label of the horizontal axis
        ylabel : string, optional, 
            label of the vertical axis
        cmap : string or Colormap object, 
            See https://matplotlib.org/examples/color/colormaps_reference.html
            in order to get all the  existing colormaps
            examples: 'hsv', 'hot', 'bone', 'tab20c', 'jet', 'seismic', 
                      'viridis'...
        vmin, vmax : scalar
            `vmin` and `vmax` are used in conjunction with norm to normalize
            luminance data.  Note if you pass a `norm` instance, your
            settings for `vmin` and `vmax` will be ignored.
        ext : scalars (left, right, bottom, top),
            The location, in data-coordinates, of the lower-left and
            upper-right corners. If `None`, the image is positioned such that
            the pixel centers fall on zero-based (row, column) indices.
        dpi : integer, optional
            Dot per inch. 
            For printed version, choose high dpi (i.e. dpi=300) => slow
            For screen version, choose low dpi (i.e. dpi=96) => fast
        format : string, optional
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
    Set values for multiresolution analysis using presets or custom parameters
    
    Parameters
    ----------
        resolution: str
            Chooses the opt_shape presets. 
            Supportes presets are: 'low', 'med', 'high' and 'custom'
        
        opt_shape: dict
            Key and values for shape settings.
            Valid keys are: ntheta, bandwidth, frequency, gamma, npyr
    Returns
    -------
        opt_shape: dict
        A valid dictionary with shape settings
    """
    # Factory presets
    opt_shape_low = dict(ntheta=2, 
                         bandwidth=1, 
                         frequency=(2**-1, 2**-2), 
                         gamma=2, 
                         npyr = 4)
    opt_shape_med =  dict(ntheta=4, 
                          bandwidth=1, 
                          frequency=(2**-1, 2**-2), 
                          gamma=2, 
                          npyr = 6)
    opt_shape_high =  dict(ntheta=8, 
                           bandwidth=1, 
                           frequency=(2**-0.5, 2**-1, 2**-1.5, 2**-2), 
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
       print('Resolution should be: low, med or high. Setting resolution to low')
       opt_shape = opt_shape_low

    return opt_shape


def plot_shape(shape_plt, params, display_values=False):
    """
    Plot shape features in 2D representation
    
    Parameters
    ----------
        shape: 1D array
        params: structured array returned by maad.features_rois.shape_features
        
    Returns
    -------
        plot
    """
    unique_theta = np.unique(params.theta)
    # compute shape of matrix
    dirs_size = unique_theta.size
    scale_size = np.unique(params.freq).size * np.unique(params.pyr_level).size
    # reshape feature vector
    idx = params.sort_values(['theta','pyr_level','scale']).index
    shape_plt = np.reshape(shape_plt[idx].values, (dirs_size, scale_size))
    unique_scale = params.scale * 2**params.pyr_level[idx]
    # get textlab
    textlab = shape_plt
    textlab = np.round(textlab,2)
    
    # plot figure
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    ax.imshow(shape_plt, origin='lower', interpolation='None', cmap='viridis')
    if display_values:    
        for (j,i),label in np.ndenumerate(textlab):
            ax.text(i,j,label,ha='center',va='center')
    else:
        pass
        
    yticklab = unique_theta
    xticklab = np.reshape(unique_scale.values, 
                          (dirs_size, scale_size))
    ax.set_xticks(np.arange(scale_size))
    ax.set_xticklabels(np.round(xticklab,2)[0,:])
    ax.set_yticks(np.arange(dirs_size))
    ax.set_yticklabels(yticklab)
    ax.set_xlabel('Scale')
    ax.set_ylabel('Theta')
    plt.show()


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
    
    Example
    -------
        s, fs = sound.load('spinetail.wav')        
        rois_tf = find_rois_cwt(s, fs, flims=(3000, 8000), tlen=2, th=0.003)
        opt_spec = {'nperseg': 512, 'overlap': 0.5}
        opt_shape = opt_shape_presets('med')
        features_rois = compute_rois_features(s, fs, rois_tf, opt_spec, 
                                              opt_shape, flims)
        
    """
    im, dt, df, ext = sound.spectrogram(s, fs, nperseg=opt_spec['nperseg'], 
                                        overlap=opt_spec['overlap'], fcrop=flims, 
                                        rescale=False, db_range=100)
    
    # format rois to bbox
    ts = np.arange(ext[0], ext[1], dt)
    f = np.arange(ext[2],ext[3]+df,df)
    rois_bbox = format_rois(rois_tf, ts, f, fmt='bbox')
        
    # roi to image blob
    im_blobs = rois_to_imblobs(np.zeros(im.shape), rois_bbox)
    
    # get features: shape, center frequency
    im = normalize_2d(im, 0, 1)
    bbox, params, shape = shape_features(im, im_blobs, resolution='custom', 
                                         opt_shape=opt_shape)
    _, cent = centroid(im, im_blobs)
    cent['frequency']= f[round(cent.y).astype(int)]  # y values to frequency
    
    # format rois to time-frequency
    rois_out = format_rois(bbox, ts, f, fmt='tf')
    
    # combine into a single df
    rois_features = pd.concat([rois_out, shape, cent.frequency], axis=1)
    return rois_features