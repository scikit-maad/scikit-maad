#!/usr/bin/env python
"""  Multiresolution Analysis of Acoustic Diversity
     features funtions 
"""
#
# Authors:  Juan Sebastian ULLOA <lisofomia@gmail.com>
#           Sylvain HAUPERT <sylvain.haupert@mnhn.fr>
#
# License: New BSD License

# Load required modules
from __future__ import print_function
import numpy as np
from datetime import datetime, timedelta
from scipy import ndimage as ndi
import matplotlib.pyplot as plt 
from skimage import transform, measure
import pandas as pd
from ..util import plot2D



def _gabor_kernel_nodc(frequency, theta=0, bandwidth=1, gamma=1, sigma_x=None, 
                       sigma_y=None, n_stds=3, offset=0):
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
    sigma_prefactor = 1.0 / np.pi * np.sqrt(np.log(2) / 2.0) * (2.0 ** b + 1) / (2.0 ** b - 1)

    if sigma_x is None:
        sigma_x = sigma_prefactor / frequency
    if sigma_y is None:
        sigma_y = sigma_prefactor / frequency /gamma
    
    # meshgrid
    x0 = np.ceil(max(np.abs(n_stds * sigma_x * np.cos(theta)),
                     np.abs(n_stds * sigma_y * np.sin(theta)), 1))
    y0 = np.ceil(max(np.abs(n_stds * sigma_y * np.cos(theta)),
                     np.abs(n_stds * sigma_x * np.sin(theta)), 1))
    y, x = np.mgrid[-y0:y0 + 1, -x0:x0 + 1]
    
    # rotation matrix
    rotx = x * np.cos(theta) + y * np.sin(theta)
    roty = -x * np.sin(theta) + y * np.cos(theta)
    
    # combine gabor and oscillatory function
    g = np.zeros(y.shape, dtype=np.complex)
    g[:] = np.exp(-0.5 * (rotx ** 2 / sigma_x ** 2 + roty ** 2 / sigma_y ** 2))
    g /= 2 * np.pi * sigma_x * sigma_y # gaussian envelope
    oscil = np.exp(1j * (2 * np.pi * frequency * rotx + offset)) # harmonic / oscilatory function
    g_dc = g*oscil
    
    # remove dc component by subtracting the envelope weighted by K
    K = np.sum(g_dc)/np.sum(g)
    g_nodc = g_dc - K*g

    # normalize between -1 and 1
    # g_nodc = (g_nodc - np.min(g_nodc))*(2 / (np.max(g_nodc)-np.min(g_nodc))) -1
    
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


def _filter_mag(im, kernel):
    """
    Normalizes the image and computes im and real part of filter response using 
    the complex kernel and the modulus operation
    
    Parameters
    ----------
    im: 2d ndarray of scalars
        Input image to process 
    
    kernel: 2d ndarray of scalars
        Complex kernel (or filter)
            
    Returns
    -------
    im_out: 2d ndarray of scalars
        Modulus operand on filtered image
    """    
    
    im = (im - im.mean()) / im.std()
    im_out = np.sqrt(ndi.convolve(im, np.real(kernel), mode='reflect')**2 +
                   ndi.convolve(im, np.imag(kernel), mode='reflect')**2)
    return im_out

def filter_multires(im_in, ext, kernels, params, npyr=4, display=False, 
                    savefig=True, **kwargs):
    """
    Computes 2D wavelet coefficients at multiple octaves/pyramids
    
    Parameters
    ----------
    im_in: list of 2D ndarray of scalars
        List of input images to process 
        
    ext : list of scalars [left, right, bottom, top], optional, default: None
        The location, in data-coordinates, of the lower-left and
        upper-right corners. If `None`, the image is positioned such that
        the pixel centers fall on zero-based (row, column) indices.
        
    kernels: list of 2D ndarray of scalars
        List of 2D kernel (Gabor) to filter the images
        
    npyr: int, optional, default is 4
        Number of pyramids to compute
        
    display : boolean, optional, default is False
        Display the signal if True
        
    savefig : string, optional, default is None
        Root filename (with full path) is required to save the figures. Postfix
        is added to the root filename.

    **kwargs, optional. This parameter is used by plt.plot and savefig functions
        figsize : tuple of integers, optional, default: (4,13)
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
    im_filt : list of 2D ndarray of scalars
        List of images filtered by each 2D kernel
    """    

    # Downscale image using gaussian pyramid
    if npyr<2:
        print('Warning: npyr should be int and larger than 1 for  multiresolution')
        im_pyr = tuple(transform.pyramid_gaussian(im_in, downscale=2, max_layer=0)) 
    else:    
        im_pyr = tuple(transform.pyramid_gaussian(im_in, downscale=2, max_layer=npyr-1)) 
    
    # Display the image pyramids
    if display:
        rows, cols = im_in.shape  
        composite_image = np.zeros((rows+ int(np.ceil(rows/2)), cols), dtype=np.double)
        composite_image[:rows, :cols] = im_pyr[0]
        i_col = 0
        for p in im_pyr[1:]:
            n_rows, n_cols = p.shape[:2]
            composite_image[rows:rows + n_rows, i_col:i_col + n_cols] = p
            i_col += n_cols

        cmap=kwargs.pop('cmap','gray') 
        vmin=kwargs.pop('vmin',0) 
        vmax=kwargs.pop('vmax',1) 
        _, fig = plot2D (composite_image, figsize=(6, 13),title='Spetrogram Pyramides', 
                         ylabel = 'pixels', xlabel = 'pixels', vmin = vmin, vmax= vmax,
                         cmap=cmap)
        kwargs.update({'cmap': cmap})
        kwargs.update({'vmin': vmin})
        kwargs.update({'vmax': vmax})
        
        # SAVE FIGURE
        if savefig is not None : 
            format=kwargs.pop('format','png') 
            dpi = kwargs.pop('dpi', 96)               
            filename = savefig+'_spectro_pyramides.'+format
            fig.savefig(filename, bbox_inches='tight', dpi=dpi, format=format,
                        **kwargs) 
            kwargs.update({'dpi': dpi})


    # filter 2d array at multiple resolutions using gabor kernels
    im_filt=[]
    
    if display: 
        ylabel =kwargs.pop('ylabel','Frequency [Hz]')
        xlabel =kwargs.pop('xlabel','Time [sec]') 
        title  =kwargs.pop('title','Spectrogram')
        figsize=kwargs.pop('figsize',(4, 13)) 
        cmap=kwargs.pop('cmap','gray') 
        vmin=kwargs.pop('vmin',0) 
        vmax=kwargs.pop('vmax',1) 
    if savefig is not None :
        filename=kwargs.pop('filename','_spectro_multires')  
        dpi=kwargs.pop('dpi',96)  
    
    for ii, im in enumerate(im_pyr):  # for each pyramid
        for jj, kernel in enumerate(kernels):  # for each kernel
            im_filt.append(_filter_mag(im, kernel))  #  magnitude response of filter
            if display: 
                _, fig = plot2D(im_filt[-1], extent=ext, figsize=figsize,
                                title=title+' shp'+str(ii*len(kernels)+jj+1), 
                                ylabel = ylabel, xlabel = xlabel, cmap=cmap, 
                                vmin = vmin, vmax= vmax,
                                **kwargs)
                if savefig is not None :              
                    ff = savefig+filename+'_shp'+str(ii*len(kernels)+jj+1)+'.'+format
                    fig.savefig(ff, bbox_inches='tight', dpi=dpi, format=format,
                                **kwargs)  
                                        
    # Rescale image using gaussian pyramid
    dims_raw = im_in.shape
    for ii, im in enumerate(im_filt):
        ratio = np.array(dims_raw)/np.array(im.shape)
        if ratio[0] > 1:
            im = transform.rescale(im, scale = ratio, mode='reflect')
        else:
            pass
        im_filt[ii] = im

    return im_filt
        


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
    
    kernels = []
    params = []
    for theta in range(ntheta):
        theta = theta/ntheta * np.pi
        for freq in frequency:
            params.append([freq, theta, bandwidth, gamma])
            kernel = np.real(_gabor_kernel_nodc(frequency=freq, theta=theta,
                                                bandwidth=bandwidth, gamma=gamma))
            kernels.append(kernel)
            
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


def shapes (im_filtlist, params, im_rois=None):
    """
    Computes the mean value obtained with each filter (shape) for each ROI
    
    Parameters
    ----------
    im_filtlist : list of 2D ndarray of scalars
        List of images filtered by each 2D Gabor kernel
    
    params: 2d structured array
        Parameters used to calculate 2D gabor kernels. 
        Params array has 4 fields (theta, freq, bandwidth, gamma)
        
    im_rois : 2D ndarray of int, optional, default is None
        image with labels as values

    Returns
    -------
    params_multires : 2d structured array
        Parameters used to calculate 2D gabor kernels. 
        Params array has 4 fields (theta, freq, bandwidth, gamma)
         
    shape_features : 2d nd array of scalars
        Each column corresponds to a shape (linked to a kernel filter) 
        Each row corresponds to a ROI
        
    """
    # Get mean intensity
    shapes = []
    if im_rois is None:
        for im in im_filtlist:
            shapes.append(np.mean(im))
    else:
        for im in im_filtlist:           
            rprops = measure.regionprops(im_rois, intensity_image=im)
            roi_mean = [roi.mean_intensity for roi in rprops]
            shapes.append(roi_mean)
        shapes = list(map(list, zip(*shapes)))  # transpose shape
        
    shape_features = np.asarray(shapes)
    
    # organise parameters
    npyr = int(len(im_filtlist) / len(params)) # number of pyramids
    params = np.asarray(params)
    orient = params[:,1]*180/np.pi
    orient = orient.tolist()*npyr
    pyr_level = np.sort(np.arange(npyr).tolist()*len(params))+1
    freq = params[:,0].tolist()*npyr
    bandwidth = params[:,2].tolist()*npyr 
    gamma = params[:,3].tolist()*npyr 
    nparams = len(params)*npyr
    
    params_multires = np.zeros(nparams,dtype={'names':('theta', 'freq', 
                                              'bandwidth','gamma','pyr_level'),
                                              'formats':('f8','f8','f8','f8','f8')})
    params_multires['theta'] = orient
    params_multires['freq'] = freq
    params_multires['bandwidth'] = bandwidth
    params_multires['gamma'] = gamma
    params_multires['pyr_level'] = pyr_level
    
    return params_multires, shape_features

def centroids (im, ext, date=None, im_rois=None):
    """
    Computes the intensity centroid (in time and frequency domain) of the 2D 
    signal 
    
    Parameters
    ----------
    im : list of 2D ndarray of scalars
        Reference spectrogram
        
    ext : list of scalars [left, right, bottom, top], optional, default: None
        The location, in data-coordinates, of the lower-left and
        upper-right corners. If `None`, the image is positioned such that
        the pixel centers fall on zero-based (row, column) indices. 
    
    date: datetime object, optional, default is None
        Position of the ROI in time (Year-Month-Day-Hour-Min-Sec). If date is 
        None, the default date : (1900,1,1,0,0,0,0) is taken.
        
    im_rois : 2D ndarray of int, optional, default is None
        image with labels as values

    Returns
    -------        
    centroid_features: 2d nd array of scalars (centroid in freq and time)
        Centroid of image. If labels provided, centroid for each ROI (rows)
        column 0 is 'cyear' 
        column 1 is 'cmonth' 
        column 2 is 'chour' 
        column 3 is 'cminute'
        column 4 is 'csecond' 
        column 5 is 'cfreq' 
            
    """    
    
    df = (ext[3]-ext[2])/(im.shape[0]-1)
    dt = (ext[1]-ext[0])/(im.shape[1]-1)
    
    centroids=[]
    if im_rois is None:
        centroids = ndi.center_of_mass(im)
    else:
        rprops = measure.regionprops(im_rois, intensity_image=im)
        centroids = [roi.weighted_centroid for roi in rprops]
          
    # Convert pixels in frequency Hz and time in s
    if date is None : date = datetime(1900,1,1,0,0,0,0)
    centroids = [((timedelta(seconds = int(t*dt))+date).year, 
                  (timedelta(seconds = int(t*dt))+date).month,
                  (timedelta(seconds = int(t*dt))+date).day,
                  (timedelta(seconds = int(t*dt))+date).hour,
                  (timedelta(seconds = int(t*dt))+date).minute,
                  (timedelta(seconds = int(t*dt))+date).second,
                  f*df) for f,t in centroids]   
    
    centroid_features = np.asarray(centroids)    
    
    return centroid_features

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
    params_shape, shape_features = shapes(im_filtlist=im_filtlist, 
                                          params = params, 
                                          im_rois=im_rois)
    # Extract centroids features for each roi
    centroid_features = centroids(im=im, ext=ext, date=date, im_rois=im_rois)
    
    if save_csv :
        table = save_csv(save_csv+'.csv', 
                         shape_features, centroid_features, label_features,
                         display=display)
    else:
        table = create_csv(shape_features, centroid_features, label_features,
                           display=display)
    
    return table, params_shape 
