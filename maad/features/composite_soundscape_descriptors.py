""" 
Utility functions for computing composite soundscape descriptors using data from multiple files at the same sampling site.
"""
import os
from concurrent.futures import ProcessPoolExecutor
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
            df['time'] = df.date.dt.hour
            print('Done!')
        elif os.path.isfile(data_input) and data_input.lower().endswith(".csv"):
            print('Loading metadata from csv file')
            try:
                # Attempt to read all wav data from the provided file path.
                df = pd.read_csv(data_input, dtype={'time': str}) 
                df['date'] = pd.to_datetime(df.date)
                df['time'] = df.date.dt.hour
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {data_input}")
            print('Done!')
    
    else:
        raise ValueError("Input 'data' must be either a Pandas DataFrame or a file path string")
    
    return df

def _validate_n_jobs(n_jobs):
    """ Validate number of jobs """
    if not isinstance(n_jobs, int):
        raise ValueError("n_jobs must be an integer.")
    
    if n_jobs <= 0 or n_jobs == -1:
        # Use all available CPUs
        n_jobs = os.cpu_count()
        print(f"Using all available CPUs: {n_jobs}")
    else:
        # Validate that the number is not larger than available CPUs
        available_cpus = os.cpu_count()
        if n_jobs > available_cpus:
            # Set n_jobs to the maximum number of available CPUs
            n_jobs = available_cpus
            print(f"Adjusted n_jobs to maximum available CPUs: {n_jobs}")
        else:
            print(f"Using specified number of CPUs: {n_jobs}")

    return n_jobs

#%%
def _spectral_peak_density(
        path_audio, target_fs, nperseg, noverlap, db_range, min_distance, threshold_abs):
    """
    Computes the spectral peak density for an audio file, representing the number of peaks per time step within each frequency bin.

    Parameters
    ----------
    path_audio : pandas DataFrame
        A Pandas DataFrame containing information about the audio files.
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
    threshold_abs : float
        Minimum amplitude threshold for peak detection in decibels.

    Returns
    -------
    peak_density : pandas Dataframe
        The peak density representation of the audio per frequency bin.

    """

    filename = os.path.basename(path_audio)
    print(f'Processing file {filename}', end='\r')

    # Load data
    s, fs = sound.load(path_audio)
    s = sound.resample(s, fs, target_fs, res_type="scipy_poly")
    Sxx, tn, fn, ext = sound.spectrogram(s, target_fs, nperseg=nperseg, noverlap=noverlap)
    Sxx_db = util.power2dB(Sxx, db_range=db_range)

    # Compute local max
    _, peak_freq = rois.spectrogram_local_max(
        Sxx_db, tn, fn, ext,
        min_distance, 
        threshold_abs)

    # Compute peak density (number of peaks / time steps)
    freq_idx, count_freq = np.unique(peak_freq, return_counts=True)
    count_peak = np.zeros(fn.shape)
    bool_index = np.isin(fn, freq_idx)
    indices = np.where(bool_index)[0]
    count_peak[indices] = count_freq / len(tn)
    peak_density = pd.Series(index=fn, data=count_peak)

    # Normalize per time
    peak_density.name = filename
    return peak_density.to_frame().T

#%%
def graphical_soundscape(
    data, threshold_abs, path_audio='path_audio', time='time', target_fs=48000, nperseg=256, noverlap=128, db_range=80, min_distance=1, n_jobs=1):
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
        If a string is passed with a directory location or a csv file, parameters 'path_audio' and 'time' will be set as default and can't be customized.
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
    n_jobs : int
        Number of processes to use for parallel computing. Default is 1.

    Returns
    -------
    res : pandas DataFrame
        A Pandas DataFrame containing the graphical representation of the soundscape.

    Examples
    --------
    >>> from maad.util import get_metadata_dir
    >>> from maad.features import graphical_soundscape, plot_graph
    >>> df = get_metadata_dir('../../data/indices')
    >>> df['hour'] = df.date.dt.hour
    >>> gs = graphical_soundscape(data=df, threshold_abs=-80, time='hour')
    >>> plot_graph(gs)
    """
    df = _input_validation(data)
    df.sort_values(by=path_audio, inplace=True)
    total_files = len(df)
    print(f'{total_files} files found to process...')
    flist = df[path_audio].to_list()

    if n_jobs==1:
        # Use sequential processing
        results = []
        for path_audio in flist:
            result = _spectral_peak_density(path_audio, target_fs, nperseg, noverlap, db_range, min_distance, threshold_abs)
            results.append(result)
    else:
        # Use parallel processing
        _validate_n_jobs(n_jobs)
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(_spectral_peak_density, path_audio, target_fs, nperseg, noverlap, db_range, min_distance, threshold_abs)
                       for path_audio in flist]

        # Wait for all tasks to complete
        results = [future.result() for future in futures]
        
    res = pd.concat(results)
    res['time'] = df[time].values
    print('\nComputation completed!')
    return res.groupby('time').mean()

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

    ax.imshow(graph.values.T, aspect='auto', origin='lower', 
              extent=[int(graph.index[0]), int(graph.index[-1]), 
                      graph.columns[0], graph.columns[-1]])
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Frequency (Hz)')

    if savefig:
        plt.savefig(fname, bbox_inches='tight')
    
    return ax
