#!/usr/bin/env python
""" Multiresolution Analysis of Acoustic Diversity
List of funtions that will endup into a Python package """
#
# Author:  Juan Sebastian ULLOA <lisofomia@gmail.com>
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
from ..util import plot2D,read_audacity_annot



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


def _plot_filter_bank(kernels,frequency, ntheta, **kwargs):
    """
    Display filter bank
    
    Parameters
    ----------
        kernels: list
            List of kernels from filter_bank_2d_nodc()
        m: int
            number of row subplots
        n: int
            number of column subplots
    Returns
    -------
        plt
    """

    params = []
    for theta in range(ntheta):
        theta = theta/ntheta * np.pi
        for freq in frequency:
            params.append([freq, theta])

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
    fontsize =kwargs.pop('fontsize',8*100/dpi) 
    
    fig.set_figwidth(figsize[0])        
    fig.set_figheight(figsize[1])
    
    w = np.asarray(w)/dpi
    h = np.asarray(h)/dpi
    wmax = np.max(w)*1.2
    hmax = np.max(h)*1.05

    params_label = []
    for param in params:
        params_label.append('theta=%d,\nf=%.2f' % (param[1] * 180 / np.pi, param[0]))
    
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
        plt
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
            Complex kernel (or filter)
            
    Returns
    -------
        im_out: Modulus operand on filtered image

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
        im_in: list of 2D arrays
            List of input images to process 
        kernels: list of 2D arrays
            List of 2D wavelets to filter the images
        npyr: int
            Number of pyramids to compute
            
    Returns
    -------
        im_list: list of 2D arrays
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
        


def filter_bank_2d_nodc(frequency, ntheta, bandwidth=1, gamma=1, display=False, savefig=None, **kwargs):
    """
    Build a Gabor filter bank with no offset component
    
    Parameters
    ----------
        frequency: 1d array
        ntheta: number of orientations
        bandwidth:

    Returns
    -------
        params: 2d structured array
            Corresponding parameters of the 2D fileters used to calculate gabor kernels
            params array has 4 fields (theta, freq, scale, gamma)
        kernels: 2d array
            Gabor kernel
    """
    
    kernels = []
    params = []
    for theta in range(ntheta):
        theta = theta/ntheta * np.pi
        for freq in frequency:
            params.append([freq, theta])
            kernel = np.real(_gabor_kernel_nodc(frequency=freq, theta=theta,
                                                bandwidth=bandwidth, gamma=gamma))
            kernels.append(kernel)
            
    if display: 
        _, fig = _plot_filter_bank(kernels, frequency, ntheta, **kwargs)
        if savefig is not None : 
            dpi   =kwargs.pop('dpi',96)
            format=kwargs.pop('format','png')               
            filename = savefig+'_filter_bank2D.'+format
            fig.savefig(filename, bbox_inches='tight', dpi=dpi, format=format,
                        **kwargs) 
            
    return params, kernels  


def shapes (im_list, params, im_rois=None):
    
    # Get mean intensity
    shapes = []
    if im_rois is None:
        for im in im_list:
            shapes.append(np.mean(im))
    else:
        for im in im_list:           
            rprops = measure.regionprops(im_rois, intensity_image=im)
            roi_mean = [roi.mean_intensity for roi in rprops]
            shapes.append(roi_mean)
        shapes = list(map(list, zip(*shapes)))  # transpose shape
    
    # organise parameters
    npyr = int(len(im_list) / len(params)) # number of pyramids
    params = np.asarray(params)
    orient = params[:,1]*180/np.pi
    orient = orient.tolist()*npyr
    pyr_level = np.sort(np.arange(npyr).tolist()*len(params))+1
    freq = params[:,0].tolist()*npyr
    nparams = len(params)*npyr
    
    params_multires = np.zeros(nparams, dtype={'names':('theta', 'freq', 'pyr_level'),
                                               'formats':('f8', 'f8', 'f8')})
    params_multires['theta'] = orient
    params_multires['freq'] = freq
    params_multires['pyr_level'] = pyr_level
    
    return params_multires, np.asarray(shapes)

def centroids (im, ext, date=None, im_rois=None):
    """
    Computes intensity centroid of the 2D signal (usually time-frequency representation) 
    
    Parameters
    ----------
        im: 2D array
            Input image to process 
        labels: 2D array, optional
            Optional array with label numbers for the region of interest and '0' otherwise            
    Returns
    -------
        centroids: 1D array of (centroid in freq, centroid in time)
            centroid of image. If labels provided, centroid for each region of interest
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
        
    return np.asarray(centroids)

def _create_csv(shape_features, centroid_features, label_features = None):
    
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
        
    return table

def save_csv(filename, shape_features, centroid_features, label_features=None):
    
    table = _create_csv(shape_features, centroid_features, label_features)
    table.to_csv(path_or_buf=filename,sep=',',mode='a',header=True, index=False)
    return table

def get_features_wrapper(im, ext, display=False, savefig=None, save_csv=None, **kwargs):
    """
    Computes shape of 2D signal (image or spectrogram) at multiple resolutions 
    using 2D Gabor filters
    
    Parameters
    ----------
        im: 2D array
            Input image to process 
        im_label: 2D array, optional
            Optional array with '1' or a number 'label' on the region of interest and '0' otherwise
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
            If im_rois provided, corresponding bounding box
    """   
       
    freq=kwargs.pop('freq',(0.75, 0.5))
    ntheta=kwargs.pop('ntheta',2)
    bandwidth=kwargs.pop('bandwidth', 1)
    gamma=kwargs.pop('gamma', 1)
    npyr=kwargs.pop('npyr', 4)  
    date=kwargs.pop('date', None) 
    im_rois=kwargs.pop('im_rois', None)  
       
    params, kernels = filter_bank_2d_nodc(frequency=freq, ntheta=ntheta, bandwidth=bandwidth,
                                          gamma=gamma, display=display, savefig=savefig)
    
    # multiresolution image filtering (Gaussian pyramids)
    im_filtlist = filter_multires(im, ext, kernels, params, npyr=npyr, 
                                  display=display, savefig=savefig)
    
    # Extract shape features for each roi
    params_shape, shape_features = shapes(im_list=im_filtlist, 
                                             params = params, 
                                             im_rois=im_rois)
    # Extract centroids features for each roi
    centroid_features = centroids(im=im, ext=ext, date=date, im_rois=im_rois)
    
    if save_csv :
        table = save_csv(save_csv+'.csv', 
                         shape_features, centroid_features, label_features=None)
    else:
        table = _create_csv(shape_features, centroid_features, label_features=None)
    
    return table, params_shape 
