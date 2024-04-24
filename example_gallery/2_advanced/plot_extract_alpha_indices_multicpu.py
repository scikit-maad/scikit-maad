#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Use multicpu functionality to compute indices
==============================================

Acoustic indices can summarize aspects of the acoustic energy distribution in
audio recordings and are widely used to characterize animal acoustic communities[1-3].
In this example, we will see how to eficiently compute multiple acoustic indices, 
and present basics post-processing posibilities. The audio recordings used in this 
example can be downloaded from the open GitHub repository 
(https://github.com/scikit-maad/scikit-maad/tree/production/data).

"""
# sphinx_gallery_thumbnail_path = '_auto_examples/2_advanced/images/sphx_glr_plot_extract_alpha_indices_multicpu_001.png'

#%%
import pandas as pd
import os
import time
import matplotlib.pyplot as plt

# Parallel processing packages
# from functools import partial
from tqdm import tqdm
from concurrent import futures

from maad import sound, features
from maad.util import date_parser
import multiprocessing as mp  

#%%
# Define the function that will be used for batch processing
# ----------------------------------------------------------

# This function should work by itself without waiting for another process
# Here we defined a function to process a single audio file. The input arguments
# are the path to the audio file and the recording date of the audio file. Both
# arguments are in the dataframe df given by the function date_parser.

def single_file_processing (audio_path,
                            date) :
    """
    Parameters
    ----------
    audio_path : string
        full path to the audio file (.wav) to process.
        The full path is in the dataframe given by the function date_parser
    date : datetime
        date of recording of the audio file. 
        The date is in the dataframe given by the function date_parser

    Returns
    -------
    df_indices : dataframe
        Dataframe containing all the temporal and spectral indices, as well as
        the audio path ('file' column) and the recording date ('Date' column)

    """
        
    # Load the original sound (16bits) and get the sampling frequency fs
    try :
        wave,fs = sound.load(filename=audio_path, 
                            channel='left', 
                            detrend=True, 
                            verbose=False)

        """ ===================================================================
                        Computation in the time domain 
        ====================================================================""" 
    
        # compute all the audio indices and store them into a DataFrame
        # dB_threshold and rejectDuration are used to select audio events.
        df_audio_ind = features.all_temporal_alpha_indices(
                                    wave, fs, 
                                    gain = G, sensibility = S,
                                    dB_threshold = 3, rejectDuration = 0.01,
                                    verbose = False, display = False)
        
        """ ===================================================================
                        Computation in the frequency domain 
        ===================================================================="""
    
        # Compute the Power Spectrogram Density (PSD) : Sxx_power
        Sxx_power,tn,fn,ext = sound.spectrogram (
                                        wave, fs, window='hann', 
                                        nperseg = 1024, noverlap=1024//2, 
                                        verbose = False, display = False, 
                                        savefig = None)   
        
        # compute all the spectral indices and store them into a DataFrame 
        # flim_low, flim_mid, flim_hi corresponds to the frequency limits in Hz 
        # that are required to compute somes indices (i.e. NDSI)
        # if R_compatible is set to 'soundecology', then the output are similar to 
        # soundecology R package.
        # mask_param1 and mask_param2 are two parameters to find the regions of 
        # interest (ROIs). These parameters need to be adapted to the dataset in 
        # order to select ROIs
        df_spec_ind, _ = features.all_spectral_alpha_indices(
                                                Sxx_power,
                                                tn,fn,
                                                flim_low = [0,1500], 
                                                flim_mid = [1500,8000], 
                                                flim_hi  = [8000,20000], 
                                                gain = G, sensitivity = S,
                                                verbose = False, 
                                                R_compatible = 'soundecology',
                                                mask_param1 = 6, 
                                                mask_param2=0.5,
                                                display = False)
        
        """ ===================================================================
                        Create a dataframe 
        ===================================================================="""        
        # add scalar indices into the df_indices dataframe
        df_indices = pd.concat([df_audio_ind,
                                df_spec_ind], axis=1)
    
        # add date and audio_path
        df_indices.insert(0, 'Date', date)
        df_indices.insert(1, 'file', audio_path)
    
    except:
        # if an error occur, send an empty output
        df_indices = pd.DataFrame()
        
    return df_indices

#%%
# Set Variables
# -------------
# We list all spectral and temporal acoustic indices that will be computed.

SPECTRAL_FEATURES=['MEANf','VARf','SKEWf','KURTf','NBPEAKS','LEQf', 
'ENRf','BGNf','SNRf','Hf', 'EAS','ECU','ECV','EPS','EPS_KURT','EPS_SKEW','ACI',
'NDSI','rBA','AnthroEnergy','BioEnergy','BI','ROU','ADI','AEI','LFC','MFC','HFC',
'ACTspFract','ACTspCount','ACTspMean', 'EVNspFract','EVNspMean','EVNspCount',
'TFSD','H_Havrda','H_Renyi','H_pairedShannon', 'H_gamma', 'H_GiniSimpson','RAOQ',
'AGI','ROItotal','ROIcover']

TEMPORAL_FEATURES=['ZCR','MEANt', 'VARt', 'SKEWt', 'KURTt',
            'LEQt','BGNt', 'SNRt','MED', 'Ht','ACTtFraction', 'ACTtCount', 
            'ACTtMean','EVNtFraction', 'EVNtMean', 'EVNtCount']

# Parameters of the audio recorder. This is not a mandatory but it allows
# to compute the sound pressure level of the audio file (dB SPL) as a 
# sonometer would do.
S = -35         # Sensbility microphone-35dBV (SM4) / -18dBV (Audiomoth)   
G = 26+16       # Amplification gain (26dB (SM4 preamplifier))

#%%
# We parse the directory were the audio dataset is located in order to get a df with date 
# and fullfilename. As the data were collected with a SM4 audio recording device
# we set the dateformat agument to 'SM4' in order to be able to parse the date
# from the filename. In case of Audiomoth, the date is coded as Hex in the 
# filename. The path to the audio dataset is "../../data/indices/".
    
if __name__ == '__main__':  # Multiprocessing should be declared under the main entry point
    # Check if the start method is already set
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("fork")  # or "spawn" or "forkserver" depending on your needs
    
    df = date_parser("../../data/indices/", dateformat='SM4', verbose=True)

    # Date is used as index. Reset the index in order to get back Date as column
    df.reset_index(inplace = True)

    #%%
    """ ===========================================================================
                    Mono CPU 
    ============================================================================"""
    # Only one cpu is used to process all data in dataframe, row by row.
    # This is the common way to process the data but it has some limitations :
    # - only 1 CPU is used even if the computer has more CPUs.
    # - data are sequentially processed which means each file will wait for the 
    # completion of the previous file in the list. If 1 file requires more time to
    # be processed, the time to complete the overall process will take longer.

    # create an empty dataframe. It will contain all ROIs found for each
    # audio file in the directory
    df_indices = pd.DataFrame()     
    
    tic = time.perf_counter()
    with tqdm(total=len(df), desc="unique cpu indices calculation...") as pbar:          
        for index, row in df.iterrows() :
            df_indices_temp = single_file_processing(row["file"], row["Date"])
            pbar.update(1)
            df_indices = pd.concat([df_indices, df_indices_temp])
    toc = time.perf_counter()

    # time duration of the process
    monocpu_duration = toc - tic

    print(f"Elapsed time is {monocpu_duration:0.1f} seconds")

    #%%%
    """ ===========================================================================
                    Multi CPU
    ============================================================================"""
    # At least 2 CPUs will be used in parallel and the files to process will be 
    # distributed on each CPU depending on their availability. This will speed up
    # the process.

    # create an empty dataframe. It will contain all ROIs found for each
    # audio file in the directory
    df_indices = pd.DataFrame()

    # Number of CPU used for the calculation. 
    nb_cpu = os.cpu_count()

    tic = time.perf_counter()
    # Multicpu process
    with tqdm(total=len(df), desc="multi cpu indices calculation...") as pbar:
        with futures.ProcessPoolExecutor(max_workers=nb_cpu) as pool:
            # give the function to map on several CPUs as well its arguments as 
            # as list
            for df_indices_temp in pool.map(
                single_file_processing, 
                df["file"].to_list(), 
                df["Date"].to_list()
            ):
                pbar.update(1)
                df_indices = pd.concat([df_indices, df_indices_temp])
    toc = time.perf_counter()

    # time duration of the process
    multicpu_duration = toc - tic

    print(f"Elapsed time is {multicpu_duration:0.1f} seconds")

    #%%
    # Display the comparison between to methods
    # -----------------------------------------

    plt.style.use('ggplot')
    fig, ax = plt.subplots(1,1,figsize=(6,2))

    # bar graphs
    width = 0.75
    x = ['sequential (mono CPU)', 'parallel (multi CPU)']
    y = [monocpu_duration, multicpu_duration]
    ax.barh(x, y, width, color=('tab:blue','tab:orange'))
    ax.set_xlabel('Elapsed time (s)')
    ax.set_title("Comparison between sequential\n and parallel processing")

    fig.tight_layout()


    # %%
