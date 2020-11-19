.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download__auto_examples_maad_examples.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr__auto_examples_maad_examples.py:


Indice - This script gives an example of how to use scikit-MAAD for ecoacoustics indices
========================================================================================

Created on Mon Aug  6 17:59:44 2018


.. code-block:: default

    #
    # Authors:  Juan Sebastian ULLOA <lisofomia@gmail.com>
    #           Sylvain HAUPERT <sylvain.haupert@mnhn.fr>
    #
    # License: New BSD License

    print(__doc__)

    # Clear all the variables 
    from IPython import get_ipython
    get_ipython().magic('reset -sf')
 
    # =============================================================================
    # Load the modules
    # =============================================================================
    import matplotlib.pyplot as plt
    import pandas as pd # for csv
    import numpy as np
    from numpy import sum, log, log10, min, max, abs, mean, median, sqrt, diff
    from skimage import filters

    # min value
    import sys
    _MIN_ = sys.float_info.min

    # =============================================================================
    # ############## Import MAAD module
    from pathlib import Path # in order to be wind/linux/MacOS compatible
    import os

    # change the path to the current path where the script is located
    # Get the current dir of the current file
    dir_path = os.path.dirname(os.path.realpath('__file__'))
    os.chdir(dir_path)

    maad_path = Path(dir_path).parents[0]
    os.sys.path.append(maad_path.as_posix())
    import maad

    # Close all the figures (like in Matlab)
    plt.close("all")


    """****************************************************************************
    # -------------- LOAD SOUND AND PREPROCESS SOUND    ---------------------------
    ****************************************************************************"""

    """****************************************************************************
    # -------------- LOAD SOUND    ---------------------------
    ****************************************************************************"""
                    
    #### Load the original sound
    s,fs = maad.sound.load(filename="../data/guyana_tropical_forest.wav", channel='left', detrend=True, verbose=False)

    """****************************************************************************
    # -------------- ENVELOPE   ---------------------------
    ****************************************************************************"""

    #### Envelope (mode fast => see TOWSEY)
    env_fast = maad.sound.envelope(s, mode='fast', N=512)
    #### Envelope (mode Hilbert)   
    env_hilbert = maad.sound.envelope(s, mode='hilbert')
    #### compute the time vector for the vector wave
    t = np.arange(0,len(s),1)/fs
    t_env_fast = np.arange(0,len(env_fast),1)*len(s)/fs/len(env_fast)
    #### plot 0.1 of the envelope and 0.1s of the abs(wave)
    fig1, ax1 = plt.subplots()
    ax1.plot(t[t<0.1], abs(s[t<0.1]), label='abs(s)')
    ax1.plot(t[t<0.1], env_hilbert[t<0.1], label='env(wave) - hilbert option')
    ax1.plot(t_env_fast[t_env_fast<0.1], env_fast[t_env_fast<0.1], label='env(s) - fast option')
    ax1.set_xlabel('Time [sec]')
    ax1.legend()


    """****************************************************************************
    # -------------- SPECTROGRAMS   ---------------------------
    ****************************************************************************"""
    #### Energy of signal s
    E1 = sum(s**2)
    maad.util.linear2dB(E1)
    #### Compute the spectrogram with 'psd' output
    N = 2048
    Pxx,tn,fn,_ = maad.sound.spectrogram (s, fs, window='hann', nperseg=N, noverlap=N//2, 
                                         fcrop=None, tcrop=None, 
                                         mode = 'psd',
                                         verbose=False, display=True, 
                                         savefig = None)   
    mean_power = mean(Pxx, axis = 1)
    E2 =sum(mean_power*len(s)) 
    maad.util.linear2dB(E2)
    #### Compute the spectrogram with 'amplitude' output
    Sxx,tn,fn,_ = maad.sound.spectrogram (s, fs, window='hann', nperseg=N, noverlap=N//20, 
                                         fcrop=None, tcrop=None, 
                                         mode = 'amplitude',
                                         verbose=False, display=True, 
                                         savefig = None)   
    ### For energy conservation => conversion of Sxx (amplitude) into power before doing the average.
    mean_power = mean(Sxx**2, axis = 1)
    E3=sum(mean_power*len(s)) 
    maad.util.linear2dB(E3)

    stop

    ## =============compute spectrogram
    #import scipy as sp
    #fn, tn, PSDxx = sp.signal.spectrogram(wave, fs, window='hanning', 
    #                                  nperseg=N, noverlap=N//2, nfft=N, 
    #                                  mode='psd',
    #                                  detrend='constant', 
    #                                  scaling='density', axis=-1)      
    #    
    ## Mutliply by the frequency resolution step (fs/nfft) to get the power
    #PSDxx = PSDxx * (fs/N) # fs/nfft is the frequency increment
    # 
    ##### Average Spectrum and PSD (better to compute the mean on the PSD)
    #mean_PSD = mean(PSDxx, axis = 1)
    #energy_PSD = sum(mean_PSD*len(wave)) 
    #print ('NRJ from mean(PSDxy) : %2.5f' % energy_PSD)
    #
    ## =============compute spectrogram
    #import scipy as sp
    #fn, tn, Sxx = sp.signal.spectrogram(wave, fs, window='hanning', 
    #                                  nperseg=N, noverlap=N//2, nfft=N, 
    #                                  mode='magnitude',
    #                                  detrend='constant', 
    #                                  scaling='density', axis=-1)      
    #    
    ## Mutliply by the sqrt of 2 times the frequency resolution step (fs/nfft) to get the amplitude
    #Sxx = Sxx*sqrt(2*fs/N) # fs/nfft is the frequency increment
    #
    #
    #PSDxx = (Sxx) **2 
    #
    ## Mutliply by the frequency resolution step (fs/nfft) to get the power
    ##PSDxx = PSDxx * (fs/N) # fs/nfft is the frequency increment
    # 
    ##### Average Spectrum and PSD (better to compute the mean on the PSD)
    #mean_PSD = mean(PSDxx, axis = 1)
    #energy_PSD = sum(mean_PSD*len(wave)) 
    #print ('NRJ from mean(PSDxy) : %2.5f' % energy_PSD)
    #
    #
    ## =============compute spectrogram
    #import scipy as sp
    #fn, tn, Sxx_complex = sp.signal.spectrogram(wave, fs, window='hanning', 
    #                                  nperseg=N, noverlap=N//2, nfft=N, 
    #                                  mode='complex',
    #                                  detrend='constant', 
    #                                  scaling='density', axis=-1)      
    #
    ## Mutliply by the frequency resolution step (fs/nfft) to get the power
    ## so multiply by the sqrt((fs/nfft)) to get the amplitude
    ## Also multiply by sqrt(2) in order to compensate 
    ## complex    
    #Sxx_complex = Sxx_complex*sqrt(2*(fs/N))
    ## magnitude
    #Sxx = abs(Sxx_complex)
    ## power
    #PSDxx = Sxx**2
    # 
    ##### Average Spectrum and PSD (better to compute the mean on the PSD)
    #mean_PSD = mean(PSDxx, axis = 1)
    #energy_PSD = sum(mean_PSD*len(wave)) 
    #print ('NRJ from mean(PSDxy) : %2.5f' % energy_PSD)












    #=========== compute spectrogram

    from scipy.signal.windows import get_window
    win = get_window('hanning', N) 
    scale = 1.0 / win.sum()**2

    import scipy as sp
    fn, tn, Sxx = sp.signal.spectrogram(wave, fs, window='hanning', 
                                      nperseg=N, noverlap=N//2, nfft=N, 
                                      mode='magnitude',
                                      detrend='constant', 
                                      scaling='spectrum', axis=-1)      

    Sxx = Sxx *2
    
    #### Average Spectrum and PSD (better to compute the mean on the PSD)
    mean_PSD = mean(Sxx**2, axis = 1)
 
    energy_PSD2 = sum(mean_PSD*len(wave)) 
    print ('NRJ from mean(PSDxy) : %2.5f' % energy_PSD2)  

    #=========== compute spectrogram
    import scipy as sp
    fn, tn, Sxx = sp.signal.spectrogram(wave, fs, window='hanning', 
                                      nperseg=N, noverlap=N//2, nfft=N, 
                                      mode='complex',
                                      detrend='constant', 
                                      scaling='spectrum', axis=-1)      
    
    #### Average Spectrum and PSD (better to compute the mean on the PSD)
    mean_PSD = mean(abs(Sxx)**2, axis = 1)

    energy_PSD2 = sum(mean_PSD*len(wave)) 
    print ('NRJ from mean(PSDxy) : %2.5f' % energy_PSD2)  

    #### Average Spectrum and PSD (better to compute the mean on the PSD)
    PSD = (np.conjugate(Sxx/scale)*Sxx/scale).real
    PSD = PSD* scale
    mean_PSD = mean(PSD, axis = 1)

    energy_PSD2 = sum(mean_PSD*len(wave)) 
    print ('NRJ from mean(PSDxy) : %2.5f' % energy_PSD2)  
 



































    envdB = maad.util.linear2dB(env, mode='amplitude')



    # time step
    if MODE_ENV == 'fast' : dt_env=1/fs*Nt
    if MODE_ENV == 'hilbert' : dt_env=1/fs

    # Time vector
    WAVE_DURATION = len(wave)/fs
    tn = np.arange(0,len(env),1)*WAVE_DURATION/len(env)

    #### Background noise estimation
    """BGNt [TOWSEY] """
    BKdB_t = maad.util.get_unimode (envdB, mode ='ale', axis=1, verbose=False, display=False)
    BK_t = maad.util.dB2linear(BKdB_t, db_gain=dB_GAIN, mode = 'amplitude') # transform bgn in dB back to amplitude
    BGNt = BKdB_t

    #### Signal to Noise ratio estimation
    """ SNRt [TOWSEY] """
    SNRt = max(envdB) - BGNt

    #### Env in dB without noise 
    envdBSansNoise = envdB - BKdB_t
    envdBSansNoise [ envdBSansNoise<0] = 0

    # Display the wave, the envelope and the background noise (red line), 
    # in linear and dB scale
    if DISPLAY :
        # linear representation
        fig1, ax1 = plt.subplots()
        ax1.plot(tn, env, lw=0.2, alpha=1)
        ax1.fill_between(tn,env,0, alpha=0.5)
        ax1.axhline(BK_t, color='red', lw=1, alpha=0.5)
        ax1.set_title('Waveform, envelope and background noise')
        ax1.set_xlabel('Time [sec]')
    
        # dB representation
        fig2, ax2 = plt.subplots()
        ax2.fill_between(tn,envdB,-50, alpha=0.75)
        ax2.axhline(BKdB_t, color='red', lw=1, alpha=0.5)
        ax2.set_title('Envelope in dB and background noise')
        ax2.set_xlabel('Time [sec]')
    
        # dB representation
        fig3, ax3 = plt.subplots()
        ax3.fill_between(tn,envdBSansNoise,0, alpha=0.75)
        ax3.set_title('Envelope in dB without background noise')
        ax3.set_xlabel('Time [sec]')

    #### Median
    """ 
        COMMENT : Result a bit different due to different Hilbert implementation
    """
    MED = median(env)
    print("median %2.5f" % MED)
 
    """ =======================================================================
    ENTROPY :  Entropy is a measure of ENERGY dispersal => square the amplitude.
        
    TEMPORAL ENTROPY => value<0.7 indicates a brief concentration of energy
                     (few seconds)
                     value close 1 indicates no peak events but rather 
                     smooth sound or noise.
    ======================================================================= """
    #### temporal entropy from the envelope's amplitude [SUEUR] or energy [TOWSEY]
    """ ENT [TOWSEY] """
    """ 
        COMMENT : Result a bit different due to different envelope estimation
        implementation
    """
    Ht = maad.ecoacoustics.entropy(env**2)
    ENT = 1 - Ht
    print("Ht %2.5f" % Ht)
 
    """**************************** Activity *******************************"""
    """ ACT & EVN [TOWSEY] """
 
    ACTtFraction, ACTtCount, ACTtMean = maad.ecoacoustics.acoustic_activity (envdBSansNoise, 
                                                            dB_threshold=6, axis=0)
    EVNtSsum, EVNtMean, EVNtCount, EVNt = maad.ecoacoustics.acoustic_events (envdBSansNoise,
                                                            dB_threshold=6,
                                                            dt=dt_env, rejectDuration=None)
    ACT = ACTtFraction
    EVN = EVNtMean
 
    # display a portion of the signal (0,20)
    if DISPLAY :
     
        fig3, ax3 = plt.subplots()
        ax3.plot(tn, env/max(abs(env)), lw=0.5, alpha=1)
        plt.fill_between(tn, 0, EVNt*1,color='red',alpha=0.5)
        ax3.set_title('Detected Events from the envelope without noise')
        ax3.set_xlabel('Time [sec]')   

    """ =======================================================================
    ===========================================================================
                     Computation in the frequency domain 
    ===========================================================================
    ========================================================================"""
 
    #### spectrogram => mode : 'amplitude' or 'psd'
    PSDxx,tn,fn,_ = maad.sound.spectrogram3 (wave, fs, 
                                         window=WIN, noverlap=NOVLP, nfft=N, 
                                         fcrop=None, tcrop=None, 
                                         verbose=False, display=DISPLAY, 
                                         savefig = None)   


    """ ********************  TO CHECK    """
    #### Smoothing of the spectrogram (like TOWSEY)
    #PSDxx = fir_filter(PSDxx,kernel=('boxcar',3), axis=1)
    #PSDxx = fir_filter(PSDxx,kernel=('boxcar',3), axis=0)
    """ ******************** END TO CHECK  """

    #### PSD spectrogram PSDxx to amplitude spectrogram Sxx
    Sxx = sqrt(PSDxx)
    #### Average Spectrum and PSD (better to compute the mean on the PSD)
    mean_PSD = mean(PSDxx, axis = 1)


    # index of the selected bandwidth
    iANTHRO_BAND = maad.ecoacoustics.index_bw(fn,ANTHRO_BAND)
    iBIO_BAND = maad.ecoacoustics.index_bw(fn,BIO_BAND)
    iINSECT_BAND = maad.ecoacoustics.index_bw(fn,INSECT_BAND) 

    #### convert into dB
    SxxdB = maad.util.linear2dB(Sxx, mode='amplitude')
    PSDxxdB = maad.util.linear2dB(PSDxx, mode ='power')
         
    # display MEAN PSD SPECTROGRAM in dB [anthropological and Biological bands]
    if DISPLAY :
        fig5, ax5 = plt.subplots()
        ax5.plot(fn[iANTHRO_BAND], 
                 maad.util.linear2dB(mean_PSD[iANTHRO_BAND], mode ='power'), 
                 color='#555555', lw=2, alpha=1)
        ax5.plot(fn[iBIO_BAND], 
                 maad.util.linear2dB(mean_PSD[iBIO_BAND], mode ='power'), 
                 color='#55DD00', lw=2, alpha=1)
        ax5.plot(fn[iINSECT_BAND], 
                 maad.util.linear2dB(mean_PSD[iINSECT_BAND], mode ='power'), 
                 color='#DDDC00', lw=2, alpha=1)                
    
    #### Noise estimation 
    """BGNf [TOWSEY] """
    """ 
        COMMENT : Result a bit different due to smoothing
    """
    BKdB_f = maad.util.get_unimode (PSDxxdB, mode ='ale', axis=1, 
                                    verbose=False, display=False)

    """ ********************  TO CHECK    """
    # smooth the noise profile
    #BKdB_f = fir_filter(BKdB_f,kernel=('boxcar', 3), axis=0)
    """ ******************** TO CHECK """
    BGNf = BKdB_f

    if DISPLAY :
        ax5.plot(fn[iANTHRO_BAND],
                    BKdB_f[iANTHRO_BAND], 'r--', lw=2, alpha=0.5)
        ax5.plot(fn[iBIO_BAND],
                    BKdB_f[iBIO_BAND], 'r--', lw=2, alpha=0.5)
        ax5.plot(fn[iINSECT_BAND],
                    BKdB_f[iINSECT_BAND], 'r--', lw=2, alpha=0.5)
        ax5.set_title('Mean PSD and uniform background noise (dash)')
        ax5.set_xlabel('Frequency [Hz]')
        ax5.axis('tight') 


    """ Parseval : energy conservation from temporal domain to frequency domain """
    print ('Parseval : energy conservation from temporal domain to frequency domain')
    print ('=> if N < 4096, the conservation is not preserved due to noise')
    energy_wav = sum(wave**2)
    print ('NRJ from wav : %2.5f' % energy_wav)
    energy_PSD = sum(PSDxx/PSDxx.shape[1]*len(wave))  
    print ('NRJ from PSDxy : %2.5f' % energy_PSD)   
    energy_PSD2 = sum(mean_PSD*len(wave)) 
    print ('NRJ from mean(PSDxy) : %2.5f' % energy_PSD2)  

    #### Signal to Noise ratio estimation
    """ SNRf [TOWSEY] """
    SNRf = max(PSDxxdB[iBIO_BAND]) - maad.util.linear2dB(mean(maad.util.dB2linear(BKdB_f[iBIO_BAND], mode='power')),mode='power')

    """ Spectrogram in dB without noise """
    # Remove background noise (BGNf is an estimation) and negative value to zero
    PSDxxdB_SansNoise =PSDxxdB - BKdB_f[..., np.newaxis]
    PSDxxdB_SansNoise[PSDxxdB_SansNoise<0] =0

    """ ********************  OPTION TO CHECK    """   
    #    # TOWSEY : smooth the spectro and set value lower than threshold (2dB in Towsey
    #    # here the threshold is evaluated as the background value) to 0
    #    SxxdB_SansNoise_smooth = maad.sound.fir_filter(SxxdB_SansNoise,kernel=('boxcar',3), axis=0)
    #    SxxdB_SansNoise_smooth = maad.sound.fir_filter(SxxdB_SansNoise_smooth,kernel=('boxcar',3), axis=1)
    #    thresh = filters.threshold_li(SxxdB_SansNoise_smooth)
    #    SxxdB_SansNoise[SxxdB_SansNoise_smooth<thresh] =_MIN_
    """ ********************  OPTION TO CHECK    """   

    # Conversion dB to linear
    PSDxx_SansNoise = maad.util.dB2linear(PSDxxdB_SansNoise, mode ='power')
    Sxx_SansNoise = sqrt(PSDxx_SansNoise)

    # Conversion linear to dB for the Amplitude Spectrum
    SxxdB_SansNoise = maad.util.linear2dB(Sxx_SansNoise, mode='amplitude')

    # display the MEAN spectrogram in dB without noise
    if DISPLAY :
        fig6, ax6 = plt.subplots()
        plt.plot(fn, mean(PSDxxdB_SansNoise,axis=1))
        ax6.set_title('Power Spectrum Density (PSD) in dB without uniform background noise')
        ax6.set_xlabel('Frequency [Hz]')
        ax6.axis('tight') 

    # display full SPECTROGRAM in dB without noise
    if DISPLAY :
        fig7, ax7 = plt.subplots()
        # set the paramteers of the figure
        fig7.set_facecolor('w')
        fig7.set_edgecolor('k')
        fig7.set_figheight(4)
        fig7.set_figwidth (13)
        # display image
        _im = ax7.imshow(PSDxxdB_SansNoise, extent=(tn[0], tn[-1], fn[0], fn[-1]), 
                         interpolation='none', origin='lower', 
                         vmin =0, vmax=max(PSDxxdB_SansNoise), cmap='gray')
        plt.colorbar(_im, ax=ax7)
        # set the parameters of the subplot
        ax7.set_title('Power Spectrum Density (PSD)')
        ax7.set_xlabel('Time [sec]')
        ax7.set_ylabel('Frequency [Hz]')
        ax7.axis('tight') 
        fig7.tight_layout()
        # Display the figure now
        plt.show()

    """******** Spectral indices from Spectrum (Amplitude or Energy) *******"""

    """
    FREQUENCY ENTROPY => low value indicates concentration of energy around a
                narrow frequency band. 
                WARNING : if the DC value is not removed before processing
                the large peak at f=0Hz (DC) will lower the entropy...
    """

    #### Entropy of spectral  
    """ EAS, ECU, ECV, EPS, KURT, SKEW [TOWSEY]  """
    X = PSDxx_SansNoise
    EAS, ECU, ECV, EPS, KURT, SKEW = maad.ecoacoustics.spectral_entropy (X,fn,frange=None,display=False)

    #### temporal entropy per frequency bin
    """ ENTsp [Towsey] """
    X = PSDxx_SansNoise
    Ht_perFreqBin = maad.ecoacoustics.entropy(X, axis=1) 
    ENTsp = 1 - Ht_perFreqBin
  
    """ Hf and H """
    Hf = maad.ecoacoustics.entropy(mean_PSD)
    print("Hf %2.5f" % Hf)

    H = Hf * Ht

    """=============================================================
    ECOLOGICAL INDICES :
            ACI
            NDSI 
            rBA 
            Bioacoustics Index
    ============================================================="""

    #### Acoustic complexity index => 1st derivative of the spectrogram
    # BUXTON use SxxdB...
    # TOWSEY and BUXTON : ACI (bioBand)
    """ ACIsp [Towsey] """
    """ 
        COMMENT : 
            ACI_sum gives same result as in Seewave R package when norm is 'per_bin'
            ACI_sum gives same result as in SoundEcology R package when norm is 'global'
    """
    X = Sxx
    ACI_xx,ACI_per_bin,ACI_sum = maad.ecoacoustics.acousticComplexityIndex(X, norm='per_bin' )
    ACI=ACI_sum
    print("ACI {seewave} %2.5f" %ACI)
   
    #### energy repartition in the frequency bins
    ###### energy based the spectrum converted into freq bins if step is different to df [soundecology, SUEUR]
    # NDSI is borned between [-1, 1]. Why not /2 and add 1 in order to be borned between [0,1] ?
    """ NDSI & rBA """
    X = PSDxx
    NDSI, rBA, AnthroEnergy, BioEnergy = maad.ecoacoustics.soundscapeIndex(X, fn, frange_bioPh=BIO_BAND,
                                                         frange_antroPh=FREQ_ANTHRO_MAX, step=1000) 
    print("NDSI {seewave} %2.5f" %NDSI)
    
    ###### Bioacoustics Index : the calculation in R from soundecology is weird...
    """ BI """
    """ VALIDATION : almost OK (difference due to different spectrogram values...)
    """
    X = Sxx
    BI = maad.ecoacoustics.bioacousticsIndex(X, fn, frange=BIO_BAND, R_compatible='soundecology') 
    print("BI %2.5f" %BI)

    #### roughness
    """ ROU """
    X = Sxx
    rough = maad.ecoacoustics.roughness(X, norm='per_bin', axis=1)
    ROU = sum(rough) 
    print("roughness {seewave} %2.2f" % ROU)

    """*********** Spectral indices from the decibel spectrogram ***********"""
    #### Score
    """ ADI & AEI """ 
    """ 
        COMMENT :
                - threshold : -50dB when norm by the max (as soundecology)
                              3dB if SxxdB_SansNoise
    """    
    X = Sxx
    #    X = SxxdB_SansNoise
    ADI = maad.ecoacoustics.acousticDiversityIndex(X, fn, fmin=0, fmax=FREQ_BIO_MAX, bin_step=1000, 
                                dB_threshold=-50, index="shannon", R_compatible='soundecology') 
    AEI = maad.ecoacoustics.acousticEvenessIndex  (X, fn, fmin=0, fmax=FREQ_BIO_MAX, bin_step=1000, 
                                dB_threshold=-50, R_compatible='soundecology') 

    print("ADI %2.5f" %ADI)
    print("AEI %2.5f" %AEI)

    """************************** SPECTRAL COVER ***************************"""
    #### Low frequency cover (LFC)
    """ LFC [TOWSEY] """
    X = SxxdB_SansNoise[iANTHRO_BAND]
    lowFreqCover, _, _ = maad.ecoacoustics.acoustic_activity (X, dB_threshold=3,axis=1)
    LFC = mean(lowFreqCover)

    #### Low frequency cover (MFC)
    """ MFC [TOWSEY] """
    X = SxxdB_SansNoise[iBIO_BAND]
    medFreqCover, _, _ = maad.ecoacoustics.acoustic_activity (X, dB_threshold=3,axis=1)
    MFC = mean(medFreqCover)

    #### Low frequency cover (LFC)
    """ HFC [TOWSEY] """
    X = SxxdB_SansNoise[iINSECT_BAND]
    HighFreqCover, _, _ = maad.ecoacoustics.acoustic_activity (X, dB_threshold=3,axis=1)
    HFC = mean(HighFreqCover)


    """**************************** Activity *******************************"""
    # Time resolution (in s)
    DELTA_T = tn[1]-tn[0]
    # Minimum time duration of an event (in s)
    MIN_EVENT_DUR = DELTA_T * 3
    # amplitude threshold in dB. 6dB is choosen because it corresponds to a 
    # signal that is 2x higher than the background
    THRESH = 6

    X = SxxdB_SansNoise
    ACTspFraction, ACTspCount, ACTspMean = maad.ecoacoustics.acoustic_activity (X, 
                                                              dB_threshold=THRESH,
                                                              axis=1)

    EVNspSum, EVNspMean, EVNspCount, EVNsp = maad.ecoacoustics.acoustic_events (X, 
                                                              dB_threshold=THRESH,
                                                              dt=DELTA_T, 
                                                              rejectDuration=MIN_EVENT_DUR)
    ### fraction EVN over the total duration for each frequency bin
    EVNspSum = np.asarray(EVNspSum)/WAVE_DURATION
    EVNspSum = EVNspSum.tolist() # IMPORTANT : to be able to eval from csv

    # display Number of events/s / frequency
    if DISPLAY :
        fig8, ax8 = plt.subplots()
        plt.plot(fn, EVNspCount)
        ax8.set_xlabel('Frequency [Hz]')
        ax8.set_title('EVNspCount : Number of events/s')
    
    # display EVENTS detected in the spectrogram
    if DISPLAY :

        fig9, ax9 = plt.subplots()
        # set the paramteers of the figure
        fig9.set_facecolor('w')
        fig9.set_edgecolor('k')
        fig9.set_figheight(4)
        fig9.set_figwidth (13)

        # display image
        _im = ax9.imshow(EVNsp*1, extent=(tn[0], tn[-1], fn[0], fn[-1]), 
                         interpolation='none', origin='lower', 
                         vmin =0, vmax=1, cmap='gray')
        plt.colorbar(_im, ax=ax9)

        # set the parameters of the subplot
        ax9.set_title('Events detected')
        ax9.set_xlabel('Time [sec]')
        ax9.set_ylabel('Frequency [Hz]')
        ax9.axis('tight') 

        fig9.tight_layout()
    
        # Display the figure now
        plt.show()            



.. code-block:: default

    c_clipping.append(sum(abs(wave)>=1))
    c_BGNt.append(BGNt)
    c_SNRt.append(SNRt)
    c_M.append(MED)
    c_ENT.append(ENT)
    c_ACTtFraction.append(ACTtFraction)
    c_ACTtCount.append(ACTtCount)
    c_ACTtMean.append(ACTtMean)
    c_EVNtSsum.append(EVNtSsum)
    c_EVNtMean.append(EVNtMean)
    c_EVNtCount.append(EVNtCount)
    c_BGNf.append(BGNf)
    c_SNRf.append(SNRf)
    c_EAS.append(EAS)
    c_ECU.append(ECU)
    c_ECV.append(ECV)
    c_EPS.append(EPS)
    c_H.append(H)
    c_ACI.append(ACI)
    c_NDSI.append(NDSI)
    c_rBA.append(rBA)
    c_BI.append(BI)
    c_ADI.append(ADI)      
    c_AEI.append(AEI)       
    c_ROU.append(ROU)
    c_LFC.append(LFC)
    c_MFC.append(MFC)
    c_HFC.append(HFC)
    c_ACTspFraction.append(ACTspFraction)
    c_ACTspCount.append(ACTspCount)
    c_ACTspMean.append(ACTspMean)
    c_EVNspSum.append(EVNspSum)
    c_EVNspMean.append(EVNspMean)
    c_EVNspCount.append(EVNspCount)

    # Average the spectro along the axis of time
    c_LTR.append(np.mean(SxxdB_SansNoise,1).tolist())
    
    # =============================================================================
 
    ####### Create the dataframe
    # add new columns to the pd dataframe 
 
    sub_df.loc[:,'clipping'] = pd.Series(c_clipping, index=sub_df.index)      
    sub_df.loc[:,'BGNt'] = pd.Series(c_BGNt, index=sub_df.index)  
    sub_df.loc[:,'SNRt'] = pd.Series(c_SNRt, index=sub_df.index)   
    sub_df.loc[:,'M'] = pd.Series(c_M, index=sub_df.index)   
    sub_df.loc[:,'ENT'] = pd.Series(c_ENT, index=sub_df.index)   
    sub_df.loc[:,'ACTtFraction'] = pd.Series(c_ACTtFraction, index=sub_df.index)   
    sub_df.loc[:,'ACTtCount'] = pd.Series(c_ACTtCount, index=sub_df.index)   
    sub_df.loc[:,'ACTtMean'] = pd.Series(c_ACTtMean, index=sub_df.index)   
    sub_df.loc[:,'EVNtSsum'] = pd.Series(c_EVNtSsum, index=sub_df.index)   
    sub_df.loc[:,'EVNtMean'] = pd.Series(c_EVNtMean, index=sub_df.index)   
    sub_df.loc[:,'EVNtCount'] = pd.Series(c_EVNtCount, index=sub_df.index)   
    sub_df.loc[:,'BGNf'] = pd.Series(c_BGNf, index=sub_df.index)   
    sub_df.loc[:,'SNRf'] = pd.Series(c_SNRf, index=sub_df.index)   
    sub_df.loc[:,'EAS'] = pd.Series(c_EAS, index=sub_df.index)   
    sub_df.loc[:,'ECU'] = pd.Series(c_ECU, index=sub_df.index)   
    sub_df.loc[:,'ECV'] = pd.Series(c_ECV, index=sub_df.index)   
    sub_df.loc[:,'EPS'] = pd.Series(c_EPS, index=sub_df.index)   
    sub_df.loc[:,'H'] = pd.Series(c_H, index=sub_df.index)   
    sub_df.loc[:,'ACI'] = pd.Series(c_ACI, index=sub_df.index)   
    sub_df.loc[:,'NDSI'] = pd.Series(c_NDSI, index=sub_df.index) 
    sub_df.loc[:,'rBA'] = pd.Series(c_rBA, index=sub_df.index) 
    sub_df.loc[:,'BI'] = pd.Series(c_BI, index=sub_df.index) 
    sub_df.loc[:,'ADI'] = pd.Series(c_ADI, index=sub_df.index)   
    sub_df.loc[:,'AEI'] = pd.Series(c_AEI, index=sub_df.index)   
    sub_df.loc[:,'ROU'] = pd.Series(c_ROU, index=sub_df.index) 
    sub_df.loc[:,'LFC'] = pd.Series(c_LFC, index=sub_df.index)   
    sub_df.loc[:,'MFC'] = pd.Series(c_MFC, index=sub_df.index)   
    sub_df.loc[:,'HFC'] = pd.Series(c_HFC, index=sub_df.index)   
    sub_df.loc[:,'ACTspFraction'] = pd.Series(c_ACTspFraction, index=sub_df.index)   
    sub_df.loc[:,'ACTspCount'] = pd.Series(c_ACTspCount, index=sub_df.index)   
    sub_df.loc[:,'ACTspMean'] = pd.Series(c_ACTspMean, index=sub_df.index)   
    sub_df.loc[:,'EVNspSum'] = pd.Series(c_EVNspSum, index=sub_df.index)   
    sub_df.loc[:,'EVNspMean'] = pd.Series(c_EVNspMean, index=sub_df.index)   
    sub_df.loc[:,'EVNspCount'] = pd.Series(c_EVNspCount, index=sub_df.index)   
    sub_df.loc[:,'LTR'] = pd.Series(c_LTR, index=sub_df.index)
 

    ######## Save .CSV
    sub_df.to_csv(path_or_buf=os.path.join(savedir,save_csv),sep=',',mode='w',header=True, index=True)



.. code-block:: default


    # =============================================================================
    # Data vizualization with pandas
    # ============================================================================
    df_indices = pd.read_csv(os.path.join(savedir,save_csv))

    # table with a summray of the indices value
    df_indices.describe()




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.000 seconds)


.. _sphx_glr_download__auto_examples_maad_examples.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: maad_examples.py <maad_examples.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: maad_examples.ipynb <maad_examples.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
