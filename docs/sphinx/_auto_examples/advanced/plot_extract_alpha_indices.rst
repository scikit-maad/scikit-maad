.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download__auto_examples_advanced_plot_extract_alpha_indices.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr__auto_examples_advanced_plot_extract_alpha_indices.py:


Extract ecoacoustics alpha indices from audio recording
=======================================================

In ecoacoustics, acoustics diversity is measured by single values, the so-called
alpha indices, which compress a portion of audio into a single value. In this
example, we will see how to compute these indices and show basics post-processing


.. code-block:: default

    # sphinx_gallery_thumbnail_path = '../_images/sphx_glr_plot_extract_alpha_indices_002.png'

    import pandas as pd
    import os
    from maad import sound, features
    from maad.util import (date_parser, plot_correlation_map, 
                           plot_features_map, plot_features, false_Color_Spectro)









.. code-block:: default

    SPECTRAL_FEATURES=['SPEC_MEAN','SPEC_VAR','SPEC_SKEW','SPEC_KURT','NBPEAKS','LEQf', 
    'ENRf','BGNf','SNRf','Hf', 'EAS','ECU','ECV','EPS','EPS_KURT','EPS_SKEW','ACI',
    'NDSI','rBA','AnthroEnergy','BioEnergy','BI','ROU','ADI','AEI','LFC','MFC','HFC',
    'ACTspFract','ACTspCount','ACTspMean', 'EVNspFract','EVNspMean','EVNspCount',
    'TFSD','H_Havrda','H_Renyi','H_pairedShannon', 'H_gamma', 'H_GiniSimpson','RAOQ',
    'AGI','ROItotal','ROIcover']

    AUDIO_FEATURES=['ZCR','AUDIO_MEAN', 'AUDIO_VAR', 'AUDIO_SKEW', 'AUDIO_KURT',
                   'LEQt','BGNt', 'SNRt','MED', 'Ht','ACTtFraction', 'ACTtCount', 
                   'ACTtMean','EVNtFraction', 'EVNtMean', 'EVNtCount']








First, we parse the directory /indices in order to get a df with date 
and fullfilename. As the data were collected with a SM4 audio recording device
we set the dateformat agument to 'SM4' in order to be able to parse the date
from the filename. In case of Audiomoth, the date is coded as Hex in the 
filename.


.. code-block:: default

    df = date_parser("../../data/indices/", dateformat='SM4', verbose=True)

    # remove index => Date becomes a column instead of an index. This is
    # required as df_audio_ind, df_spec_ind and df_spec_ind_per_bin do not have 
    # date as index. Then we can concatenate all the dataframe.
    #df = df.reset_index()





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    indices.csv
    per_bin_indices.csv
    S4A03895_20190522_000000.wav
    S4A03895_20190522_001500.wav
    S4A03895_20190522_003000.wav
    S4A03895_20190522_004500.wav
    S4A03895_20190522_010000.wav
    S4A03895_20190522_011500.wav
    S4A03895_20190522_013000.wav
    S4A03895_20190522_014500.wav
    S4A03895_20190522_020000.wav
    S4A03895_20190522_021500.wav
    S4A03895_20190522_023000.wav
    S4A03895_20190522_024500.wav
    S4A03895_20190522_030000.wav
    S4A03895_20190522_031500.wav
    S4A03895_20190522_033000.wav
    S4A03895_20190522_034500.wav
    S4A03895_20190522_040000.wav
    S4A03895_20190522_041500.wav
    S4A03895_20190522_043000.wav
    S4A03895_20190522_044500.wav
    S4A03895_20190522_050000.wav
    S4A03895_20190522_051500.wav
    S4A03895_20190522_053000.wav
    S4A03895_20190522_054500.wav
    S4A03895_20190522_060000.wav
    S4A03895_20190522_061500.wav
    S4A03895_20190522_063000.wav
    S4A03895_20190522_064500.wav
    S4A03895_20190522_070000.wav
    S4A03895_20190522_071500.wav
    S4A03895_20190522_073000.wav
    S4A03895_20190522_074500.wav
    S4A03895_20190522_080000.wav
    S4A03895_20190522_081500.wav
    S4A03895_20190522_083000.wav
    S4A03895_20190522_084500.wav
    S4A03895_20190522_090000.wav
    S4A03895_20190522_091500.wav
    S4A03895_20190522_093000.wav
    S4A03895_20190522_094500.wav
    S4A03895_20190522_100000.wav
    S4A03895_20190522_101500.wav
    S4A03895_20190522_103000.wav
    S4A03895_20190522_104500.wav
    S4A03895_20190522_110000.wav
    S4A03895_20190522_111500.wav
    S4A03895_20190522_113000.wav
    S4A03895_20190522_114500.wav
    S4A03895_20190522_120000.wav
    S4A03895_20190522_121500.wav
    S4A03895_20190522_123000.wav
    S4A03895_20190522_124500.wav
    S4A03895_20190522_130000.wav
    S4A03895_20190522_131500.wav
    S4A03895_20190522_133000.wav
    S4A03895_20190522_134500.wav
    S4A03895_20190522_140000.wav
    S4A03895_20190522_141500.wav
    S4A03895_20190522_143000.wav
    S4A03895_20190522_144500.wav
    S4A03895_20190522_150000.wav
    S4A03895_20190522_151500.wav
    S4A03895_20190522_153000.wav
    S4A03895_20190522_154500.wav
    S4A03895_20190522_160000.wav
    S4A03895_20190522_161500.wav
    S4A03895_20190522_163000.wav
    S4A03895_20190522_164500.wav
    S4A03895_20190522_170000.wav
    S4A03895_20190522_171500.wav
    S4A03895_20190522_173000.wav
    S4A03895_20190522_174500.wav
    S4A03895_20190522_180000.wav
    S4A03895_20190522_181500.wav
    S4A03895_20190522_183000.wav
    S4A03895_20190522_184500.wav
    S4A03895_20190522_190000.wav
    S4A03895_20190522_191500.wav
    S4A03895_20190522_193000.wav
    S4A03895_20190522_194500.wav
    S4A03895_20190522_200000.wav
    S4A03895_20190522_201500.wav
    S4A03895_20190522_203000.wav
    S4A03895_20190522_204500.wav
    S4A03895_20190522_210000.wav
    S4A03895_20190522_211500.wav
    S4A03895_20190522_213000.wav
    S4A03895_20190522_214500.wav
    S4A03895_20190522_220000.wav
    S4A03895_20190522_221500.wav
    S4A03895_20190522_223000.wav
    S4A03895_20190522_224500.wav
    S4A03895_20190522_230000.wav
    S4A03895_20190522_231500.wav
    S4A03895_20190522_233000.wav
    S4A03895_20190522_234500.wav




LOAD SOUND AND PREPROCESS SOUND  


.. code-block:: default

    df_indices = pd.DataFrame()
    df_indices_per_bin = pd.DataFrame()
    
    for index, row in df.iterrows() : 
    
        # get the full filename of the corresponding row
        fullfilename = row['file']
        # Save file basename
        path, filename = os.path.split(fullfilename)
        print ('\n**************************************************************')
        print (filename)
    
        #### Load the original sound (16bits) and get the sampling frequency fs
        try :
            wave,fs = sound.load(filename=fullfilename, channel='left', detrend=True, verbose=False)

        except:
            # Delete the row if the file does not exist or raise a value error (i.e. no EOF)
            df.drop(index, inplace=True)
            continue
    
        """ =======================================================================
                         Computation in the time domain 
        ========================================================================""" 
    
        # Parameters of the audio recorder. This is not a mandatory but it allows
        # to compute the sound pressure level of the audio file (dB SPL) as a 
        # sonometer would do.
        S = -35         # Sensbility microphone-35dBV (SM4) / -18dBV (Audiomoth)   
        G = 26+16       # Amplification gain (26dB (SM4 preamplifier))

        # compute all the audio indices and store them into a DataFrame
        # dB_threshold and rejectDuration are used to select audio events.
        df_audio_ind = features.all_audio_alpha_indices(wave, fs, 
                                              gain = G, sensibility = S,
                                              dB_threshold = 3, rejectDuration = 0.01,
                                              verbose = False, display = False)
    
        """ =======================================================================
                         Computation in the frequency domain 
        ========================================================================"""
 
        # Compute the Power Spectrogram Density (PSD) : Sxx_power
        Sxx_power,tn,fn,ext = sound.spectrogram (wave, fs, window='hanning', 
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
        df_spec_ind, df_spec_ind_per_bin = features.all_spectral_alpha_indices(Sxx_power,
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
    
        """ =======================================================================
                         Create a dataframe 
        ========================================================================"""
        # First, we create a dataframe from row that contains the date and the 
        # full filename. This is done by creating a DataFrame from row (ie. TimeSeries)
        # then transposing the DataFrame. 
        df_row = pd.DataFrame(row)
        df_row =df_row.T
        df_row.index.name = 'Date'
        df_row = df_row.reset_index()

        # add scalar indices into the df_indices dataframe
        df_indices = df_indices.append(pd.concat([df_row,
                                                  df_audio_ind,
                                                  df_spec_ind], axis=1))
        # add vector indices into the df_indices_per_bin dataframe
        df_indices_per_bin = df_indices_per_bin.append(pd.concat([df_row, 
                                                                  df_spec_ind_per_bin], axis=1))
    # Set back Date as index
    df_indices = df_indices.set_index('Date')
    df_indices_per_bin = df_indices_per_bin.set_index('Date')





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


    **************************************************************
    S4A03895_20190522_000000.wav

    **************************************************************
    S4A03895_20190522_001500.wav

    **************************************************************
    S4A03895_20190522_003000.wav

    **************************************************************
    S4A03895_20190522_004500.wav

    **************************************************************
    S4A03895_20190522_010000.wav

    **************************************************************
    S4A03895_20190522_011500.wav

    **************************************************************
    S4A03895_20190522_013000.wav

    **************************************************************
    S4A03895_20190522_014500.wav

    **************************************************************
    S4A03895_20190522_020000.wav

    **************************************************************
    S4A03895_20190522_021500.wav

    **************************************************************
    S4A03895_20190522_023000.wav

    **************************************************************
    S4A03895_20190522_024500.wav

    **************************************************************
    S4A03895_20190522_030000.wav

    **************************************************************
    S4A03895_20190522_031500.wav

    **************************************************************
    S4A03895_20190522_033000.wav

    **************************************************************
    S4A03895_20190522_034500.wav

    **************************************************************
    S4A03895_20190522_040000.wav

    **************************************************************
    S4A03895_20190522_041500.wav

    **************************************************************
    S4A03895_20190522_043000.wav

    **************************************************************
    S4A03895_20190522_044500.wav

    **************************************************************
    S4A03895_20190522_050000.wav

    **************************************************************
    S4A03895_20190522_051500.wav

    **************************************************************
    S4A03895_20190522_053000.wav

    **************************************************************
    S4A03895_20190522_054500.wav

    **************************************************************
    S4A03895_20190522_060000.wav

    **************************************************************
    S4A03895_20190522_061500.wav

    **************************************************************
    S4A03895_20190522_063000.wav

    **************************************************************
    S4A03895_20190522_064500.wav

    **************************************************************
    S4A03895_20190522_070000.wav

    **************************************************************
    S4A03895_20190522_071500.wav

    **************************************************************
    S4A03895_20190522_073000.wav

    **************************************************************
    S4A03895_20190522_074500.wav

    **************************************************************
    S4A03895_20190522_080000.wav

    **************************************************************
    S4A03895_20190522_081500.wav

    **************************************************************
    S4A03895_20190522_083000.wav

    **************************************************************
    S4A03895_20190522_084500.wav

    **************************************************************
    S4A03895_20190522_090000.wav

    **************************************************************
    S4A03895_20190522_091500.wav

    **************************************************************
    S4A03895_20190522_093000.wav

    **************************************************************
    S4A03895_20190522_094500.wav

    **************************************************************
    S4A03895_20190522_100000.wav

    **************************************************************
    S4A03895_20190522_101500.wav

    **************************************************************
    S4A03895_20190522_103000.wav

    **************************************************************
    S4A03895_20190522_104500.wav

    **************************************************************
    S4A03895_20190522_110000.wav

    **************************************************************
    S4A03895_20190522_111500.wav

    **************************************************************
    S4A03895_20190522_113000.wav

    **************************************************************
    S4A03895_20190522_114500.wav

    **************************************************************
    S4A03895_20190522_120000.wav

    **************************************************************
    S4A03895_20190522_121500.wav

    **************************************************************
    S4A03895_20190522_123000.wav

    **************************************************************
    S4A03895_20190522_124500.wav

    **************************************************************
    S4A03895_20190522_130000.wav

    **************************************************************
    S4A03895_20190522_131500.wav

    **************************************************************
    S4A03895_20190522_133000.wav

    **************************************************************
    S4A03895_20190522_134500.wav

    **************************************************************
    S4A03895_20190522_140000.wav

    **************************************************************
    S4A03895_20190522_141500.wav

    **************************************************************
    S4A03895_20190522_143000.wav

    **************************************************************
    S4A03895_20190522_144500.wav

    **************************************************************
    S4A03895_20190522_150000.wav

    **************************************************************
    S4A03895_20190522_151500.wav

    **************************************************************
    S4A03895_20190522_153000.wav

    **************************************************************
    S4A03895_20190522_154500.wav

    **************************************************************
    S4A03895_20190522_160000.wav

    **************************************************************
    S4A03895_20190522_161500.wav

    **************************************************************
    S4A03895_20190522_163000.wav

    **************************************************************
    S4A03895_20190522_164500.wav

    **************************************************************
    S4A03895_20190522_170000.wav

    **************************************************************
    S4A03895_20190522_171500.wav

    **************************************************************
    S4A03895_20190522_173000.wav

    **************************************************************
    S4A03895_20190522_174500.wav

    **************************************************************
    S4A03895_20190522_180000.wav

    **************************************************************
    S4A03895_20190522_181500.wav

    **************************************************************
    S4A03895_20190522_183000.wav

    **************************************************************
    S4A03895_20190522_184500.wav

    **************************************************************
    S4A03895_20190522_190000.wav

    **************************************************************
    S4A03895_20190522_191500.wav

    **************************************************************
    S4A03895_20190522_193000.wav

    **************************************************************
    S4A03895_20190522_194500.wav

    **************************************************************
    S4A03895_20190522_200000.wav

    **************************************************************
    S4A03895_20190522_201500.wav

    **************************************************************
    S4A03895_20190522_203000.wav

    **************************************************************
    S4A03895_20190522_204500.wav

    **************************************************************
    S4A03895_20190522_210000.wav

    **************************************************************
    S4A03895_20190522_211500.wav

    **************************************************************
    S4A03895_20190522_213000.wav

    **************************************************************
    S4A03895_20190522_214500.wav

    **************************************************************
    S4A03895_20190522_220000.wav

    **************************************************************
    S4A03895_20190522_221500.wav

    **************************************************************
    S4A03895_20190522_223000.wav

    **************************************************************
    S4A03895_20190522_224500.wav

    **************************************************************
    S4A03895_20190522_230000.wav

    **************************************************************
    S4A03895_20190522_231500.wav

    **************************************************************
    S4A03895_20190522_233000.wav

    **************************************************************
    S4A03895_20190522_234500.wav




After calculating all alpha indices (in audio and spectral domain), let's 
have a look to the data. 
First, plot correlation map of all indices. We set the R threshold to 0 in
order to have everything. If you want to focus on highly correlated indices
set the threshold to 0.75 for instance.


.. code-block:: default

    fig, ax = plot_correlation_map(df_indices, R_threshold=0)




.. image:: /_auto_examples/advanced/images/sphx_glr_plot_extract_alpha_indices_001.png
    :alt: plot extract alpha indices
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Volumes/lacie_macosx/numerical_analysis_toolbox/scikit-maad/maad/util/visualization.py:789: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      plt.show()




A graphical way to have a quick overview of the indices variation during 
a 24h cycle consists in plotting heatmaps of indices 
For a better view, we seperate spectral and audio indices.


.. code-block:: default

    plot_features_map(df_indices[SPECTRAL_FEATURES], mode='24h')
    plot_features_map(df_indices[AUDIO_FEATURES], mode='24h')

    # A more classical way to analyse variations of indices consists in plotting
    # graphs. We choose to normalize rescale their value between 0 to 1 in order to
    # compare their trend during a 24h cycle 
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3,2, sharex=True, squeeze=True, figsize=(5,5))
   
    fig, ax[0,0] = plot_features(df_indices[['Hf']],norm=True,mode='24h', ax=ax[0,0])  
    fig, ax[0,1] = plot_features(df_indices[['AEI']],norm=True,mode='24h', ax=ax[0,1])
    fig, ax[1,0] = plot_features(df_indices[['NDSI']],norm=True,mode='24h', ax=ax[1,0])
    fig, ax[1,1] = plot_features(df_indices[['ACI']],norm=True,mode='24h', ax=ax[1,1])
    fig, ax[2,0] = plot_features(df_indices[['MED']],norm=True,mode='24h', ax=ax[2,0])
    fig, ax[2,1] = plot_features(df_indices[['ROItotal']],norm=True,mode='24h', ax=ax[2,1])




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /_auto_examples/advanced/images/sphx_glr_plot_extract_alpha_indices_002.png
          :alt: plot extract alpha indices
          :class: sphx-glr-multi-img

    *

      .. image:: /_auto_examples/advanced/images/sphx_glr_plot_extract_alpha_indices_003.png
          :alt: plot extract alpha indices
          :class: sphx-glr-multi-img

    *

      .. image:: /_auto_examples/advanced/images/sphx_glr_plot_extract_alpha_indices_004.png
          :alt: plot extract alpha indices
          :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Volumes/lacie_macosx/numerical_analysis_toolbox/scikit-maad/maad/util/visualization.py:567: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      plt.show()
    /Volumes/lacie_macosx/numerical_analysis_toolbox/scikit-maad/maad/util/visualization.py:686: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      plt.show()




Create false color spectrograms with 3 indices


.. code-block:: default

    fcs, triplet = false_Color_Spectro(df_indices_per_bin,
                                       indices = ['AUDIO_KURT_per_bin',
                                                 'EVNspCount_per_bin',
                                                 'AUDIO_MEAN_per_bin'],
                                       reverseLUT=False,
                                       unit='hours',
                                       permut=False,
                                       display=True,
                                       figsize=(5,9))




.. image:: /_auto_examples/advanced/images/sphx_glr_plot_extract_alpha_indices_005.png
    :alt: False Color Spectro   [R:AUDIO_KURT; G:EVNspCount; B:AUDIO_MEAN]
    :class: sphx-glr-single-img





# Save date as .CSV
# save df_indices
save_csv = 'indices.csv'
df_indices.to_csv(path_or_buf=os.path.join("../data/indices/",save_csv),sep=',',mode='w',header=True, index=True)
# save df_indices_per_bin (for future false color spectro)
df_indices_per_bin.to_csv(path_or_buf=os.path.join("../data/indices/",'per_bin_'+save_csv),sep=',',mode='w',header=True, index=True)


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  23.126 seconds)


.. _sphx_glr_download__auto_examples_advanced_plot_extract_alpha_indices.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_extract_alpha_indices.py <plot_extract_alpha_indices.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_extract_alpha_indices.ipynb <plot_extract_alpha_indices.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
