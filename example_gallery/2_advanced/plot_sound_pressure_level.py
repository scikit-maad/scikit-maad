#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estimate sound pressure level from audio recordings
===================================================

Sound pressure level (dB SPL) is a quantitative value that allows comparison of audio 
recordings coming from different datasets and environments. Sound pressure level 
corresponds to a quantitative measurement of the acoustic pressure energy. 
An Automated Recording Unit (ARU) can be converted into a pseudo sound meter level 
knowing few parameters: the sensitivity of the microphone, the amplification
gain, the bit depth and the voltage range of the analog to digital converter.
This is sufficient to convert a wav file (array of intergers) into pressure 
(Pa). Of course, as the frequency response of the ARUs's microphone is never flat, 
the result is an approximation of the real sound pressure level. In order to be
more precise, one should correct the frequency response of the microphone.

In this example, we will evaluate the variation of the sound pressure level 
of the ambient sound in a cold temperate forest in France during 24h. 

"""

#%%
# Load required modules
# ---------------------
# Load required packages
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from maad import sound, util, spl

#%%
# Set variables
# -------------
# It is very important to always keep the parameters of the ARU that was used 
# to collect the audio dataset. This is a mandatory to compute the sound 
# pressure level of the audio file (dB SPL) as a sound meter level would do.
# For this experiment, we used a Songmeter 4 (SM4, from Wildlife Acoustics) 
# with an amplification gain of 16dB.

# From Sam Lapp (thank you very much Sam for your help)
# in case of audiomoth the gain is
#              Gain (dB) 
# low gain     12.7     
#              16.9
# medium gain  23.5
#              28.0
# high gain    30.4

S = -35                 # Sensibility of the microphone -35dBV (SM4) / -38dBV (Audiomoth)  / -35dBV (SM Mini) / -35dBV (SM Mini 2)  
PREANPLIFIER = 26       # Preamplifier gain +26dB (SM4) / +20.8 (Audiomoth 1.2.0) / +20dB (Audiomoth 1.1.0) / +23dB (SM Mini) / +23dB (SM Mini 2)
G = PREANPLIFIER+16     # Total amplification gain in dB (preamplifier Gain + Gain)
VADC = 2                # Voltage range of the analog to digital converter (ADC)

#%%
# First, we parse the directory /indices in order to get a DataFrame with date 
# and fullfilename. As the data was collected with a SM4 audio recording device,
# we set the dateformat agument to 'SM4' in order to be able to parse the date
# from the filename. In case of Audiomoth, the date is coded as Hex in the 
# filename.
df = util.date_parser("../../data/indices/", dateformat='SM4', verbose=True)

#%%
# Load and preprocess audio
# -------------------------
# Then we process all the files found in the directory /indices.
# Initialisation of an empty dataframe df_spl to store all the dB SPL values 
# extracted from the whole audio dataset.

df_spl = pd.DataFrame()

# Main loop to go through all audio files
for index, row in df.iterrows() : 
    
    # initialisation of an empty list to store the dB SPL of the current
    # audio recording.
    leq_list = []
    
    # get the full filename of the corresponding row
    fullfilename = row['file']
    # Save file basename
    path, filename = os.path.split(fullfilename)
    print ('\n**************************************************************')
    print (filename)
    
    #### Load the original sound (16bits, only left channel) and get the sampling 
    # frequency fs
    try :
        wave,fs = sound.load(filename=fullfilename, channel='left', detrend=True, verbose=False)

    except:
        # Delete the row if the file does not exist or raise a value error (i.e. no EOF)
        df.drop(index, inplace=True)
        continue
    
    """ =======================================================================
                    Computation in the frequency domain 
    ========================================================================"""

    # Compute the Power Spectrogram Density (PSD) : Sxx_power
    Sxx_power,tn,fn,ext = sound.spectrogram (wave, fs, window='hann', 
                                            nperseg = 1024, noverlap=1024//2, 
                                            verbose = False, display = False, 
                                            savefig = None)   
    
    
    #### Average PSD (It's a mandatory to compute the mean on the PSD for 
    # energy conservation)
    mean_PSD = np.mean(Sxx_power, axis = 1)
    
    #### Compute the Leq of each frequency band (1kHz step)
    leq_list+=[spl.psd2leq(mean_PSD[util.index_bw(fn,(0,1000))], 
                            gain=G, 
                            sensitivity=S, 
                            Vadc=VADC)]
    leq_list+=[spl.psd2leq(mean_PSD[util.index_bw(fn,(1000,2000))], 
                            gain=G, 
                            sensitivity=S, 
                            Vadc=VADC)]
    leq_list+=[ spl.psd2leq(mean_PSD[util.index_bw(fn,(2000,3000))], 
                            gain=G, 
                            sensitivity=S, 
                            Vadc=VADC)]
    leq_list+=[spl.psd2leq(mean_PSD[util.index_bw(fn,(3000,4000))], 
                            gain=G, 
                            sensitivity=S, 
                            Vadc=VADC)]
    leq_list+=[spl.psd2leq(mean_PSD[util.index_bw(fn,(4000,5000))], 
                            gain=G, 
                            sensitivity=S, 
                            Vadc=VADC)]
    leq_list+=[spl.psd2leq(mean_PSD[util.index_bw(fn,(5000,6000))], 
                            gain=G, 
                            sensitivity=S, 
                            Vadc=VADC)]
    leq_list+=[spl.psd2leq(mean_PSD[util.index_bw(fn,(6000,7000))], 
                            gain=G, 
                            sensitivity=S, 
                            Vadc=VADC)]
    leq_list+=[spl.psd2leq(mean_PSD[util.index_bw(fn,(7000,8000))], 
                            gain=G, 
                            sensitivity=S, 
                            Vadc=VADC)]
    leq_list+=[spl.psd2leq(mean_PSD[util.index_bw(fn,(8000,9000))], 
                            gain=G, 
                            sensitivity=S, 
                            Vadc=VADC)]
    leq_list+=[spl.psd2leq(mean_PSD[util.index_bw(fn,(9000,10000))], 
                            gain=G, 
                            sensitivity=S, 
                            Vadc=VADC)]
    leq_list+=[spl.psd2leq(mean_PSD[util.index_bw(fn,(10000,11000))], 
                            gain=G, 
                            sensitivity=S, 
                            Vadc=VADC)]
    
    #### Create a dataframe from the list
    df_leq = pd.DataFrame([leq_list],
                        columns = ['0-1kHz', 
                                    '1-2kHz',
                                    '2-3kHz',
                                    '3-4kHz',
                                    '4-5kHz',
                                    '5-6kHz',
                                    '6-7kHz',
                                    '7-8kHz',
                                    '8-9kHz',
                                    '9-10kHz',
                                    '10-11kHz',
                                    ])

    """ =======================================================================
                    Create a dataframe 
    ========================================================================"""
    #### We create a dataframe from row that contains the date and the 
    # full filename. This is done by creating a DataFrame from row (ie. TimeSeries)
    # then transposing the DataFrame. 
    df_row = pd.DataFrame(row)
    df_row =df_row.T
    df_row.index.name = 'Date'
    df_row = df_row.reset_index()

    #### add Leq values into the df_spl dataframe
    df_spl = pd.concat([df_spl,pd.concat([df_row, df_leq], axis=1)], axis=0)

#### When the loop ends, set Date as index
df_spl = df_spl.set_index('Date')

#### remove the column file
df_spl = df_spl.drop(columns=['file'])

#%%
# Display results
# ---------------
# Display Leq (dB SPL) dynamics for each frequency band.
# One can observe that most of the acoustic energy is between 0-1kHz. Moreover
# the sound pressure level of the 0-1kHz frequency band is not constant but 
# increases during the day, from 4am to 22pm with a maximum between at 10am 
# This is mainly due to the airplanes flying over the forest during the day.
# One can also observe that the energy of the biophony is really lower than 
# the energy of the low frequency band. The sound pressure level of the biophony
# is the highest between 5am to 9am, which corresponds to the dawn chorus
plt.style.use('ggplot')
df_spl_rev = df_spl.iloc[:,::-1]
util.plot_features_map(df_spl_rev, norm=False, cmap='RdPu')
plt.tight_layout()

#%%
# Display the comparison between the dynamics of the Leq (dB SPL) corresponding 
# mainly to the anthropophony (frequency band 0-1kHz) and the biophony 
# (frequency band 1-12kHz)
plt.figure(figsize=[7,4])
df_spl.iloc[:,0].apply(util.add_dB, axis=1).plot()
df_spl.iloc[:,1:].apply(util.add_dB, axis=1).plot()
plt.xlabel('Hours')
plt.ylabel('Sound Pressure Level (dB SPL)')
plt.legend(['anthropophony', 'biophony'])
plt.tight_layout()
plt.show()

