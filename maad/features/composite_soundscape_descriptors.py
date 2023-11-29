""" 
Utility functions for computing composite soundscape descriptors using data from multiple files at the same sampling site.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from maad import sound, util, rois

#%% Function argument validation
def _input_validation(data_input):
    """ Validate dataframe or path input argument """
    if isinstance(data_input, pd.DataFrame):
        df = data_input
    elif isinstance(data_input, str):
        if os.path.isdir(data_input):
            print('Collecting metadata from directory path...')
            df = util.get_metadata_dir(data_input)
            print('Done!')
        elif os.path.isfile(data_input) and data_input.lower().endswith(".csv"):
            print('Loading metadata from csv file')
            try:
                # Attempt to read all wav data from the provided file path.
                df = pd.read_csv(data_input) 
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {data_input}")
            print('Done!')
    else:
        raise ValueError("Input 'data' must be either a Pandas DataFrame or a file path string")
    return df

#%%
def graphical_soundscape(
    data, threshold_abs, path_audio='path_audio', time='time', target_fs=48000, nperseg=256, noverlap=128, db_range=80, min_distance=1):
    """
    Computes a graphical soundscape from a given DataFrame of audio files.

    This function is a variant of the original graphical soundscapes introduced by Campos-Cerqueira et al. The peaks are detected on the spectrogram instead of detecting peaks on the spectrum. Results are similar but not equal to the ones computed using seewave in R.

    References:
        - Campos‐Cerqueira, M., et al., 2020. How does FSC forest certification affect the acoustically active fauna in Madre de Dios, Peru? Remote Sensing in Ecology and Conservation 6, 274–285. https://doi.org/10.1002/rse2.120
        - Furumo, P.R., Aide, T.M., 2019. Using soundscapes to assess biodiversity in Neotropical oil palm landscapes. Landscape Ecology 34, 911–923.
        - Campos-Cerqueira, M., Aide, T.M., 2017. Changes in the acoustic structure and composition along a tropical elevational gradient. JEA 1, 1–1. https://doi.org/10.22261/JEA.PNCO7I

    Parameters
    ----------
    data : pandas DataFrame
        A Pandas DataFrame containing information about the audio files.
    threshold_abs : float
        Minimum amplitude threshold for peak detection in decibels.
    path_audio : str
        Column name where the full path of audio is provided.
    time : str
        Column name where the time is provided as a string using the format 'HHMMSS'.
    target_fs : int
        The target sample rate to resample the audio signal if needed.
    nperseg : int
        Window length of each segment to compute the spectrogram.
    noverlap : int
        Number of samples to overlap between segments to compute the spectrogram.
    db_range : float
        Dynamic range of the computed spectrogram.
    min_distance : int
        Minimum number of indices separating peaks.

    Returns
    -------
    res : pandas DataFrame
        A Pandas DataFrame containing the graphical representation of the soundscape.

    Examples
    --------
    >>> from maad.util import get_metadata_dir
    >>> from maad.features import graphical_soundscape, plot_graph
    >>> df = get_metadata_dir('../../data/indices')
    >>> gs = graphical_soundscape(data=df, threshold_abs=-80)
    >>> plot_graph(gs)
    """

    df = _input_validation(data)
    df.sort_values(by=path_audio, inplace=True)
    total_files = len(df)
    print(f'{total_files} files found to process...')
    res = pd.DataFrame()
    for idx, df_aux in df.reset_index().iterrows():
        filename = os.path.basename(df_aux[path_audio])
        print(f'Processing file {idx+1} of {total_files}: {filename}', end='\r')
        
        # Load data
        s, fs = sound.load(df_aux[path_audio])
        s = sound.resample(s, fs, target_fs, res_type="scipy_poly")
        Sxx, tn, fn, ext = sound.spectrogram(s, target_fs, nperseg=nperseg, noverlap=noverlap)
        Sxx_db = util.power2dB(Sxx, db_range=db_range)

        # Compute local max
        peak_time, peak_freq = rois.spectrogram_local_max(
            Sxx_db, tn, fn, ext,
            min_distance, 
            threshold_abs)
        
        # Count number of peaks at each frequency bin
        freq_idx, count_freq = np.unique(peak_freq, return_counts=True)
        count_peak = np.zeros(fn.shape)
        bool_index = np.isin(fn, freq_idx)
        indices = np.where(bool_index)[0]
        count_peak[indices] = count_freq / len(tn)
        peak_density = pd.Series(index=fn, data=count_peak)

        # Normalize per time
        #peak_density = (peak_density > 0).astype(int)
        peak_density.name = os.path.basename(df_aux[path_audio])
        res = pd.concat([res, peak_density.to_frame().T])

    res["time"] = df[time].str[0:2].astype(int).to_numpy()
    print('\nComputation completed!')
    return res.groupby("time").mean()

#%%
def plot_graph(graph, ax=None, savefig=False, fname=None):
    """ Plots a graphical soundscape

    Parameters
    ----------
    graph : pandas.Dataframe
        A graphical soundscape as pandas dataframe with index as time and frequency as columns
    ax : matplotlib.axes, optional
        Axes for subplots. If not provided it creates a new figure, by default None.

    Returns
    -------
    ax
        Axes of the figure
    """
    if ax == None:
        fig, ax = plt.subplots()

    ax.imshow(graph.values.T, aspect='auto', origin='lower')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Frequency (Hz)')
    ytick_idx = np.arange(0, graph.shape[1], 20).astype(int)
    ax.set_yticks(ytick_idx)
    ax.set_yticklabels(graph.columns[ytick_idx].astype(int).values)

    if savefig:
        plt.savefig(fname, bbox_inches='tight')
    
    return ax
