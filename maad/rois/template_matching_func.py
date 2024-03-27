#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Template matching functions

New BSD License

TODO:
    - min_t and max_t should be 0 to len(Sxx_audio)
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from skimage.feature import match_template
import matplotlib.pyplot as plt
from matplotlib import patches
from maad import sound, util

#%%
def template_matching(
    Sxx, Sxx_template, tn, ext, peak_th, peak_distance=None, display=False, **kwargs
):
    """
    Use normalized spectrogram cross-correlation to detect the occurrence of a template
    sound in a target audio.

    The output is an array with values between -1.0 and 1.0. The value at a given
    position corresponds to the correlation coefficient between the spectrograms of
    the target audio and the template. The function also delivers the detection found
    as peaks and as regions of interest (rois).

    Parameters
    ----------
    Sxx : 2D array
        Spectrogram of audio signal.

    Sxx_template : TYPE
        Spectrogram of target sound.

    tn : 1d array
        Time vector of target audio, which results from the maad.sound.spectrogram function.

    fn : 1d array
        Frecuency vector of target audio, which results from the maad.sound.spectrogram function.

    ext : list of scalars [left, right, bottom, top]
        Extent keyword arguments controls the bounding box in data coordinates for the
        spectrogram of the target audio, which results from the maad.sound.spectrogram function.

    peak_th : float, optional
        Threshold applied to find peaks in the cross-correlation array.
        Should be a value between -1 and 1.

    peak_distance : float, optional
        Required minimal temporal distance (>= 0) in seconds between neighbouring
        peaks. If set to `None`, the minimum temporal resolution will be used.
        The minimal temporal resolution is given by the array tn and depends on the parameters
        used to compute the spectrogram.

    display : Boolean, optional
        display the results of the template matching. The default is False.

    **kwargs: keywords pair, optional
        Set aditional specificities to find peaks in xcorroelation array.
        Arguments are passed to the the function scipy.signal.find_peaks.


    Returns
    -------
    xcorrcoef : 1D array
        Correlation coefficients resulting from the cross-correlation between
        audio and target template.

    rois : pandas DataFrame
        Detections found based on cross-correlation coefficients. The result is
        presented as a DataFrame where each row represent a detection, with the
        peak time (peak_time), peak amplitude (xcorrcoef), minimum and maximum time
        (min_t, max_t), and minimum and maximum frequency (min_f, max_f).

    """

    # check inputs
    if Sxx.ndim < Sxx.ndim:
        raise ValueError(
            "Dimensionality of template must be less than or "
            "equal to the dimensionality of image."
        )
    if np.any(np.less(Sxx.shape, Sxx_template.shape)):
        raise ValueError("Target spectrogram must be larger than template.")

    if peak_distance is None:  # if not provided, set to minimum distance
        peak_distance = np.diff(tn)[0]

    # set temporal distance to spectrogram pixels
    peak_distance_pixel = peak_distance / np.diff(tn)[0]

    if peak_distance_pixel < 1:
        raise ValueError(
            f"`peak_distance` must be greater or equal to spectrogram resolution: {np.diff(tn)[0]}"
        )

    # Pad Sxx to have len(xcorrcoef) == Sxx.shape[1]
    # if Sxx_template.shape[1] is even substract 1 to time width
    time_width = np.floor(Sxx_template.shape[1] / 2).astype(int)
    if (Sxx_template.shape[1] % 2) == 1:
        pad_width = (
            (
                0,
                0,
            ),
            (time_width, time_width),
        )
    else:
        pad_width = (
            (
                0,
                0,
            ),
            (time_width, time_width - 1),
        )
    Sxx_pad = np.pad(Sxx, pad_width, mode="edge")

    # Compute normalized cross-correlation
    xcorrcoef = match_template(Sxx_pad, Sxx_template)

    # When flims from Sxx is larger than Sxx_template, take mean value
    xcorrcoef = np.mean(xcorrcoef, axis=0)

    ## Find peaks
    prominence = kwargs.pop("prominence", None)
    width = kwargs.pop("width", None)
    wlen = kwargs.pop("wlen", None)
    rel_height = kwargs.pop("rel_height", 0.5)
    plateau_size = kwargs.pop("plateau_size", None)
    threshold = kwargs.pop("threshold", None)

    peaks, peak_dict = find_peaks(
        xcorrcoef,
        peak_th,
        threshold,
        peak_distance_pixel,
        prominence,
        width,
        wlen,
        rel_height,
        plateau_size,
    )
    peaks_time = tn[peaks]

    # Build rois as pandas Dataframe
    # Create Dataframe and adjust extreme values for min_t and max_t
    template_len = tn[Sxx_template.shape[1]] - tn[0]
    rois = pd.DataFrame(
        {
            "peak_time": peaks_time,
            "xcorrcoef": xcorrcoef[peaks],
            "min_t": peaks_time - template_len / 2,
            "max_t": peaks_time + template_len / 2,
        }
    )
    rois.loc[rois.min_t < 0 , "min_t"] = tn[0]
    rois.loc[rois.max_t > tn[-1] , "max_t"] = tn[-1]

    if display == True:
        rois['min_f'] = ext[2]
        rois['max_f'] = ext[3]
        # plot spectrogram
        fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        util.plot_spectrogram(Sxx, ext, log_scale=False, ax=ax[0], colorbar=False)
        if not (rois.empty):
            for idx, _ in rois.iterrows():
                xy = (rois.min_t[idx], rois.min_f[idx])
                width = rois.max_t[idx] - rois.min_t[idx]
                height = rois.max_f[idx] - rois.min_f[idx]
                rect = patches.Rectangle(
                    xy, width, height, lw=1, edgecolor="yellow", facecolor="yellow", alpha=0.25
                )
                ax[0].add_patch(rect)

        # plot corr coef
        ax[1].plot(tn[0 : xcorrcoef.shape[0]], xcorrcoef)
        ax[1].plot(peaks_time, xcorrcoef[peaks], "x")
        ax[1].hlines(peak_th, 0, tn[-1], linestyle="dotted", color="0.75")
        ax[1].set_xlabel("Time [s]")
        ax[1].set_ylabel("Correlation coeficient")

    return xcorrcoef, rois
