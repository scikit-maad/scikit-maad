.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download__auto_examples_indices_calculation_indice.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr__auto_examples_indices_calculation_indice.py:


Created on Mon Aug  6 17:59:44 2018

Updated Thu 28 May 2020

This script gives an example of how to use scikit-MAAD for ecoacoustics indices


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

    import scipy as sp
    from scipy.signal import hilbert, hann, sosfilt, convolve, iirfilter, get_window, resample, tukey
    from scipy.ndimage.morphology import binary_erosion, binary_dilation, binary_closing, binary_opening
    from scipy.interpolate import interp1d 
    from scipy.stats import rankdata
    from scipy import ndimage as ndi

    from skimage import filters,transform, measure, morphology
    from skimage.morphology import disk, grey

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

    #maad_path = Path(dir_path).parents[0]
    maad_path = Path('/home/haupert/DATA/mes_projets/_TOOLBOX/Python/maad_project/scikit-maad/')
    os.sys.path.append(maad_path.as_posix())
    import maad

    # Close all the figures (like in Matlab)
    plt.close("all")

    """****************************************************************************
    # -------------------          options              ---------------------------
    ****************************************************************************"""
    # root directory of the files
    #datadir = '/media/haupert/EXT_DATA/FRANCE/dB@RISOUX/'
    #datadir='/media/haupert/Seagate Expansion Drive/Jura_dB@risoux_Jul-Oct_2020/'
    datadir = '/media/haupert/Transcend/dB@RISOUX/2018/'

    datadir = '/home/haupert/DATA/mes_projets_data/FRANCE/dB@RISOUX/TRAINING_DATA_ANNOTATION_INDICE/MAGNETO_01_S4A03895/Data'
    # root directory of the save file
    savedir = '//home/haupert/DATA/mes_projets/01_FRANCE/dB@RISOUX/DATA_PROCESSING/INDICES_2018-2019-2020'
    save_csv = 'jura_indices_Juillet_2018_allmagneto.csv'

    CHANNEL = 'left'
    MODE_ENV = 'fast'  # 'fast' #'hilbert'

    Nt = 512           # frame size (in points)
    N  = 1024        # fft size (in points)
    NOVLP = N//2    # Number of overlaping points during spectrogram calculation
    WIN = 'hanning'      #'boxcar' hanning'

    dB_RANGE = 120
    dB_GAIN = 0

    # Sensbility microphone
    S = -35  #-35dBV (SM4) / -18dBV (Audiomoth)
    # Amplification gain
    G = 16+26 # total amplification gain (16dB Petchora)
    # Reference pressure (in the air : 20µPa)
    P_REF = 20e-6
    # Leq integration time dt in s
    deltaT = 1
    #
    VADC = 2
    # bits
    bit = 16

    LOW_FREQ_MIN = 0
    LOW_FREQ_MAX = 1000
    MID_FREQ_MIN = 1000
    MID_FREQ_MAX = 10000
    HIGH_FREQ_MIN = 10000
    HIGH_FREQ_MAX = 20000

    LOW_BAND = (LOW_FREQ_MIN, LOW_FREQ_MAX)
    MID_BAND = (MID_FREQ_MIN,MID_FREQ_MAX)
    HIGH_BAND = (HIGH_FREQ_MIN,HIGH_FREQ_MAX)

    DISPLAY = False
    DISPLAY_PRIORITY = True
    SAVE = False

    """****************************************************************************
    # -------------------          end options          ---------------------------
    ****************************************************************************"""

    """****************************************************************************
    # -------------- LOAD SOUND AND PREPROCESS SOUND    ---------------------------
    ****************************************************************************"""
    # parse a directory in order to get a df with date and fullfilename
    df = maad.util.date_parser(datadir, dateformat='SM4', verbose=True)

    # select the files

    # =============================================================================
    # ##### EXAMPLES
    # #see https://pandas.pydata.org/pandas-docs/sdf/timeseries.html
    # # Returning an array containing the hours for each row in your dataframe
    # df.index.hour
    # # grab all rows where the time is between 12h and 13h,
    # df.between_time('12:00:00','13:00:00')
    # # Increment the time by 1 microsecond
    # df.index = df.index+ pd.Timedelta(microseconds=1) 
    # date_selec = pd.date_range('2018-07-14', '2018-07-20')
    # =============================================================================

    ## Select data between date range
    T0 = '2019-06-01 00:00:00'
    T1 = '2019-07-01 00:00:00'
    sub_df = df[T0:T1]

    # or keep all data
    #sub_df = df

    # or select a row (= a single file)
    #sub_df = df.iloc[0:(6*24*7)]

    # define the indices' lists
    # c_ for column_
    # lists of scalar
    N_FILE = len(sub_df)

    """****************************************************************************
    # -------------- LOAD SOUND AND PREPROCESS SOUND    ---------------------------
    ****************************************************************************"""
    # define the indices' lists
    # c_ for column_
    # lists of scalar
    c_file =  []
    c_clipping =  []
    c_LEQt =  []
    c_BGNt =  []
    c_SNRt =  []
    c_M =  []
    c_ENT =  []
    c_ACTtFraction=  []
    c_ACTtCount=  []
    c_ACTtMean=  []
    c_EVNtSsum=  []
    c_EVNtMean=  []
    c_EVNtCount=  []
    c_LEQf=  []
    c_BGNf=  []
    c_RAIN=  []
    c_SNRf=  []
    c_EAS=  []
    c_ECU=  []
    c_ECV=  []
    c_EPS=  []
    c_KURT=  []
    c_SKEW=  []
    c_H=  []
    c_ACI=  []
    c_NDSI=  []
    c_rBA=  []
    c_BI=  []
    c_ADI=  []
    c_AEI=  []
    c_ROU=  []
    c_LFC=  []
    c_MFC=  []
    c_HFC =  []
    c_AGI23ms=  []
    c_AGI46ms=  []
    c_AGI93ms=  []
    c_AGI186ms=  []
    c_AGI371ms=  []
    c_AGI743ms=  []
    c_AGI1487ms=  []
    c_AGI2972ms=  []
    c_AGI5944ms=  []
    c_AGIdtmax=  []
    c_RAOQ =  []
    c_ROUsurf=  []
    c_ROItotal = []
    c_ROIcover = []
    c_ROIunique = []

    # lists of vectors
    c_frequency = []
    c_BGNf_per_bin = []
    c_AGI=  []
    c_AGIdt=  []
    c_ACTspFraction=  []
    c_ACTspCount=  []
    c_ACTspMean =  []
    c_EVNspSum=  []
    c_EVNspMean=  []
    c_EVNspCount=  []
    c_Rq_per_bin=  []
    c_Ra_per_bin=  []
    c_TFSD=  []
    c_TFSD_per_bin = []
    c_AGIperbin=  []
    c_ENTsp=  []
    c_ACI_per_bin=  []
    c_LTS =  []

    for index, row in sub_df.iterrows() : 
    
        # get the full filename of the corresponding row
        fullfilename = row['file']
        # Save file basename
        path, filename = os.path.split(fullfilename)
        savefile_base = filename[0:-4]
        savefile = os.path.join(savedir,savefile_base)
    
        print ('\n***********************')
        print (filename)
    
        """========================================================================
        ===========================================================================
                         Computation in the time domain 
        ===========================================================================
        ======================================================================= """                    
        #### Load the original sound (16bits)
        try :
            wave,fs = maad.sound.load(filename=fullfilename, channel=CHANNEL, detrend=True, verbose=False)
        except:
            # Delete the row if the file does not exist or raise a value error (i.e. no EOF)
            sub_df.drop(index, inplace=True)
            continue
    
        #### wave -> pressure -> Leq
        p = maad.util.wav2pressure(wave, gain=G, Vadc=VADC, sensitivity=S, dBref=94)
        LEQt = maad.util.mean_dBSPL(maad.util.pressure2Leq(p, f=fs, dt=deltaT), axis=1)
        print("LEQt %2.5f" % LEQt)

    #    #### Highpass signal (200Hz)
    #    wave = maad.sound.iir_filter1d(wave,fs,fcut=200,forder=1,fname='butter',ftype='highpass')

        #### Envelope (mode fast => see TOWSEY)
        env = maad.sound.envelope(p, mode=MODE_ENV, N=Nt)
        envdB = maad.util.pressure2dBSPL(env)
    
        # time step
        if MODE_ENV == 'fast' : dt_env=1/fs*Nt
        if MODE_ENV == 'hilbert' : dt_env=1/fs
    
        # Time vector
        WAVE_DURATION = len(wave)/fs
        tn = np.arange(0,len(env),1)*WAVE_DURATION/len(env)

        #### Background noise estimation
        """BGNt [TOWSEY] """
        BGNt = maad.util.get_unimode (envdB, mode ='ale')
   
        #### Signal to Noise ratio estimation
        """ SNRt [TOWSEY] """
        SNRt = maad.util.mean_dBSPL(envdB, axis=1) - BGNt
        print("SNRt %2.5f" % SNRt)
    
        #### Env in dB without noise 
        envdBSansNoise = envdB - BGNt
        envdBSansNoise [ envdBSansNoise<0] = 0
    
        # Display the wave, the envelope and the background noise (red line), 
        # in linear and dB scale
        if DISPLAY :
            # linear representation
            fig1, ax1 = plt.subplots()
            ax1.plot(tn, env, lw=0.2, alpha=1)
            ax1.fill_between(tn,env,0, alpha=0.5)
            ax1.axhline(maad.util.dBSPL2pressure(BGNt), color='red', lw=1, alpha=0.5)
            ax1.set_title('Waveform, envelope and background noise')
            ax1.set_xlabel('Time [sec]')
        
            # dB representation
            fig2, ax2 = plt.subplots()
            ax2.fill_between(tn,envdB,-50, alpha=0.75)
            ax2.axhline(BGNt, color='red', lw=1, alpha=0.5)
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
        Ht = maad.features.entropy(env**2)
        ENT = 1 - Ht
        print("Ht %2.5f" % Ht)
 
        """**************************** Activity *******************************"""
        """ ACT & EVN [TOWSEY] """
 
        # dB_threshold=6dB as envdBSansNoise is amplitude not energy
        ACTtFraction, ACTtCount, ACTtMean = maad.features.acoustic_activity (envdBSansNoise, 
                                                                dB_threshold=6, axis=0)
        EVNtSsum, EVNtMean, EVNtCount, EVNt = maad.features.acoustic_events (envdBSansNoise,
                                                                dB_threshold=6,
                                                                dt=dt_env, rejectDuration=None)

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
 
        #### Power Spectrogram Density (PSD) : Pxx
        Pxx,tn,fn,_ = maad.sound.spectrogram (p, fs, 
                                             window=WIN, noverlap=NOVLP, nperseg=N, 
                                             fcrop=None, tcrop=None, 
                                             verbose=False, display=DISPLAY, 
                                             savefig = None)   
     
        # index of the selected bandwidth
        iLOW_BAND = maad.util.index_bw(fn,LOW_BAND)
        iMID_BAND = maad.util.index_bw(fn,MID_BAND)
        iHIGH_BAND = maad.util.index_bw(fn,HIGH_BAND) 
       
        """ ********************  TO CHECK    """
        #### Smoothing of the spectrogram (like TOWSEY)
        #Pxx = fir_filter(PSDxx,kernel=('boxcar',3), axis=1)
        #Pxx = fir_filter(PSDxx,kernel=('boxcar',3), axis=0)
        """ ******************** END TO CHECK  """

        #### power spectrogram Pxx to amplitude spectrogram Sxx
        Sxx = sqrt(Pxx)
        #### convert into dB
        dBxx = maad.util.pressure2dBSPL(Sxx)
        #### Average Pxx in time direction (Better to work with PSD for energy conservation)
        P = mean(Pxx, axis =1)
        #### get the Leq
        LEQf = maad.util.PSD2Leq(P)
        print("LEQf %2.5f" % LEQf)
        #### get the Leq per bin
        LEQf_per_bin = maad.util.power2dBSPL(P)
    
        # display full SPECTROGRAM in dB 
        if DISPLAY or DISPLAY_PRIORITY :
            fig_kwargs = {'vmax': max(dBxx),
                          'vmin':0,
                          'extent':(tn[0], tn[-1], fn[0], fn[-1]),
                          'figsize':(4,13),
                          'title':'Power Spectrogram',
                          'xlabel':'Time [sec]',
                          'ylabel':'Frequency [Hz]',
                          }
            fig4, ax4 = maad.util.plot2D(dBxx,**fig_kwargs)

        # display MEAN PSD SPECTROGRAM in dB [anthropological and Biological bands]
        if DISPLAY :
            fig5, (ax5_1, ax5_2) = plt.subplots(nrows=1, ncols=2) 
        
            # mean PSD in along frequency
            ax5_1.plot(fn[iLOW_BAND], 
                     maad.util.power2dBSPL(P[iLOW_BAND]),
                     color='#555555', lw=2, alpha=1)
            ax5_1.plot(fn[iMID_BAND], 
                     maad.util.power2dBSPL(P[iMID_BAND]), 
                     color='#55DD00', lw=2, alpha=1)
            ax5_1.plot(fn[iHIGH_BAND], 
                     maad.util.power2dBSPL(P[iHIGH_BAND]), 
                     color='#DDDC00', lw=2, alpha=1)
                 
            # mean PSD in along time        
            ax5_2.plot(tn, 
                     maad.util.power2dBSPL(mean(Pxx, axis=0)), 
                     color='#005555', lw=2, alpha=1)
        
        
        """ Parseval : energy conservation from temporal domain to frequency domain """
        print ('Parseval : energy conservation from temporal domain to frequency domain')
        print ('=> if N < 4096, the conservation is not fully preserved due to noise')
        energy_wav = sum(p**2)
        print ('NRJ from wav : %2.5f' % energy_wav)
        energy_PSD = sum(Pxx/Pxx.shape[1]*len(p))  
        print ('NRJ from PSDxy : %2.5f' % energy_PSD)   
        energy_PSD2 = sum(P*len(p)) 
        print ('NRJ from mean(PSDxy) : %2.5f' % energy_PSD2)  

        #### Noise estimation 
        """BGNf [TOWSEY] """
        """ 
            COMMENT : Result a bit different due to smoothing
        """   
        """ Get the stationnary noise """    
    #    BGN_dBxx = morphology.reconstruction(dBxx-np.quantile(dBxx, 0.999),dBxx)
    #    # Remove stationnary background noise
    #    dBxx_NoStationnaryNoise =dBxx - BGN_dBxx 
    #    if DISPLAY :
    #        fig_kwargs['title'] = 'PSDxxdB without stationnary noise'
    #        maad.util.plot2D(dBxx_NoStationnaryNoise,**fig_kwargs)  
        
        dBxx_NoStationnaryNoise = dBxx
    
        """ Get the horizontal noisy profile along freq axis"""   
        BGN_HorizontalNoise = maad.util.get_unimode (dBxx_NoStationnaryNoise, mode ='ale', axis=1)
    
        # smooth the profile by removing spurious thin peaks (less than 5 pixels wide)
        # => keep buzzing and long monochromatic calls
    #    BGN_HorizontalNoise = morphology.opening(BGN_HorizontalNoise, selem=np.array((1,1,1,1,1)))
        BGN_HorizontalNoise = maad.util.running_mean(BGN_HorizontalNoise,N=5)
    
        # Remove horizontal noisy peaks profile (BGN_VerticalNoise is an estimation) and negative value to zero
        dBxx_NoHorizontalNoise =dBxx_NoStationnaryNoise - BGN_HorizontalNoise[..., np.newaxis]
    
        if DISPLAY :
            fig_kwargs['title'] = 'Power Spectrogram without horizontal noisy profile'
            maad.util.plot2D(dBxx_NoHorizontalNoise,**fig_kwargs)
        
        """ Get the vertical noisy profile along time axis"""       
        BGN_VerticalNoise = maad.util.get_unimode (dBxx_NoHorizontalNoise, mode ='ale', axis=0)
    
        # smooth the profile by removing spurious thin peaks (less than 9 pixels wide)
        BGN_VerticalNoise = maad.util.running_mean(BGN_VerticalNoise,N=9)
    
        # Remove vertical noisy peaks profile noise (BGN_VerticalNoise is an estimation) and negative value to zero
        dBxx_noNoise = dBxx_NoHorizontalNoise - BGN_VerticalNoise[np.newaxis, ...]

        # Set negative value to 0
        dBxx_noNoise[dBxx_noNoise<0] = 0        

        # display full SPECTROGRAM in dB without noise
        if DISPLAY :
            fig_kwargs['title'] = 'Power spectrogram without noise'
            maad.util.plot2D(dBxx_noNoise,**fig_kwargs)

        if DISPLAY :           
            # mean PSD in along frequency
            ax5_1.plot(fn[iLOW_BAND], 
                     BGN_HorizontalNoise[iLOW_BAND],
                     color='#FF0000', lw=2, alpha=1)
            ax5_1.plot(fn[iMID_BAND], 
                     BGN_HorizontalNoise[iMID_BAND],
                     color='#FF0000', lw=2, alpha=1)
            ax5_1.plot(fn[iHIGH_BAND], 
                     BGN_HorizontalNoise[iHIGH_BAND],
                     color='#FF0000', lw=2, alpha=1)
            # mean PSD in along time        
            ax5_2.plot(tn, 
                     BGN_VerticalNoise,
                     color='#FF0000', lw=2, alpha=1)
    
        #### Signal to Noise ratio estimation
        """ SNRf [TOWSEY] """
        BGNf_per_bin = BGN_HorizontalNoise
        SNRf_per_bin = LEQf_per_bin - BGNf_per_bin
        BGNf = maad.util.add_dBSPL(BGNf_per_bin, axis=1)
        SNRf = LEQf - BGNf
        print("SNRf %2.5f" % SNRf)
    
        # Conversion dB to linear
        Pxx_noNoise = maad.util.dBSPL2power(dBxx_noNoise)
        Sxx_noNoise = sqrt(Pxx_noNoise)
    
        # display the MEAN spectrogram in dB without noise
        if DISPLAY :
            fig6, ax6 = plt.subplots()
            plt.plot(fn, maad.util.mean_dBSPL(dBxx_noNoise,axis=2).transpose())
            ax6.set_title('Power spectrogram without uniform background noise')
            ax6.set_xlabel('Frequency [Hz]')
            ax6.set_ylabel('Amplitude [dB]')
            ax6.axis('tight') 

                    
        """******** Spectral indices from Spectrum (Amplitude or Energy) *******"""
    
        """
        FREQUENCY ENTROPY => low value indicates concentration of energy around a
                    narrow frequency band. 
                    WARNING : if the DC value is not removed before processing
                    the large peak at f=0Hz (DC) will lower the entropy...
        """
    
        #### Entropy of spectral  
        """ EAS, ECU, ECV, EPS, KURT, SKEW [TOWSEY]  """
        X = Pxx_noNoise
        EAS, ECU, ECV, EPS, KURT, SKEW = maad.features.spectral_entropy (X,fn,flim=None,display=False)
    
        #### temporal entropy per frequency bin
        """ ENTsp [Towsey] """
        X = Pxx_noNoise
        Ht_perFreqBin = maad.features.entropy(X, axis=1) 
        ENTsp = 1 - Ht_perFreqBin
  
        """ Hf and H """
        Hf = maad.features.entropy(P)
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
        ACI_xx,ACI_per_bin,ACI_sum = maad.features.acousticComplexityIndex(X, norm='global' )
        ACI=ACI_sum
        print("ACI {SoundEcology} %2.5f" %ACI)
       
        #### energy repartition in the frequency bins
        ###### energy based the spectrum converted into freq bins if step is different to df [soundecology, SUEUR]
        # NDSI is borned between [-1, 1]. Why not /2 and add 1 in order to be borned between [0,1] ?
        """ NDSI & rBA """
        X = Pxx
        NDSI, rBA, AnthroEnergy, BioEnergy = maad.features.soundscapeIndex(X, fn, flim_bioPh=MID_BAND,
                                                                           flim_antroPh=LOW_FREQ_MAX, step=1000) 
        print("NDSI {seewave} %2.5f" %NDSI)
        
        ###### Bioacoustics Index : the calculation in R from soundecology is weird...
        """ BI """
        """ VALIDATION : almost OK (difference due to different spectrogram values...)
        """
        X = Sxx
        BI = maad.features.bioacousticsIndex(X, fn, flim=MID_BAND, R_compatible=True) 
        print("BI  {SoundEcology} %2.5f" %BI)

        #### roughness
        """ ROU """
        X = Sxx
        rough = maad.features.roughness(X, norm='per_bin', axis=1)
        ROU = sum(rough) 
        print("roughness {seewave} %2.2f" % ROU)

        """*********** Spectral indices from the decibel spectrogram ***********"""
        #### Score
        """ ADI & AEI """ 
        """ 
            COMMENT :
                    - threshold : -50dB when norm by the max (as soundecology)
                                  6dB if PSDxxdB_SansNoise
        """    
        X = Sxx
        ADI = maad.features.acousticDiversityIndex(X, fn, fmin=0, fmax=MID_FREQ_MAX, bin_step=1000, 
                                    dB_threshold=-50, index="shannon", R_compatible='soundecology') 
        AEI = maad.features.acousticEvenessIndex  (X, fn, fmin=0, fmax=MID_FREQ_MAX, bin_step=1000, 
                                    dB_threshold=-50, R_compatible='soundecology') 
    
        print("ADI %2.5f" %ADI)
        print("AEI %2.5f" %AEI)
    
        """************************** SPECTRAL COVER ***************************"""
        #### Low frequency cover (LFC)
        """ LFC [TOWSEY] """
        X = dBxx_noNoise[iLOW_BAND]
        lowFreqCover, _, _ = maad.features.acoustic_activity (X, dB_threshold=3,axis=1)
        LFC = mean(lowFreqCover)
    
        #### Low frequency cover (MFC)
        """ MFC [TOWSEY] """
        X = dBxx_noNoise[iMID_BAND]
        medFreqCover, _, _ = maad.features.acoustic_activity (X, dB_threshold=3,axis=1)
        MFC = mean(medFreqCover)
    
        #### Low frequency cover (LFC)
        """ HFC [TOWSEY] """
        X = dBxx_noNoise[iHIGH_BAND]
        HighFreqCover, _, _ = maad.features.acoustic_activity (X, dB_threshold=3,axis=1)
        HFC = mean(HighFreqCover)
    
        """**************************** Activity *******************************"""
        # Time resolution (in s)
        DELTA_T = tn[1]-tn[0]
        # Minimum time duration of an event (in s)
        MIN_EVENT_DUR = DELTA_T * 3
        # amplitude threshold in dB. 3dB is choosen because it corresponds to a 
        # signal that is 2x higher than the background
        THRESH = 3
    
        X = dBxx_noNoise
        ACTspFraction, ACTspCount, ACTspMean = maad.features.acoustic_activity (X, 
                                                                  dB_threshold=THRESH,
                                                                  axis=1)

        EVNspSum, EVNspMean, EVNspCount, EVNsp = maad.features.acoustic_events (X, 
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
      
        """**************************** New indices*****************************""" 
   
        #### Time frequency derivation (TFSD)
        """ TFSD [AUMOND] """
        X = Pxx
        TFSD, TFSD_per_bin = maad.features.tfsd(X, fn, flim=(2000,8000), 
                                                thirdOctave = None, display=DISPLAY)
    
        print("TFSD %2.5f" % mean(TFSD))
    
        #### Surface roughness as defined for material
        """ 
            ROUsurf 
        """
        BAND = iMID_BAND +iHIGH_BAND
        X = Pxx[BAND]
        # mean deviation from PSD in SPL unit
        Ra_per_bin, Rq_per_bin, Ra, Rq = maad.features.surfaceRoughness(X,norm='per_bin')   
        # roughness in term of dB SPL
        ROUsurf =maad.util.PSD2Leq(Ra_per_bin)
    
        print("ROUsurf %2.2fdB" % ROUsurf)
    
        #### raoQ
        """ RAOQ """
        BAND = iMID_BAND +iHIGH_BAND
        X = Sxx[BAND]
        f= fn[BAND]
        X_bins, bins = maad.features.intoBins(X, f, bin_step=1000, axis=0) 
        RAOQ = maad.features.raoQ(np.mean(X_bins,1), bins=bins)
    
        print("RAOQ %2.2fdB" % RAOQ)
    
        #### Acoustic gradient index => real 1st derivative of the spectrogram
        """ AGI """
        X = Sxx
        MAX_dt =10  # Last pyramid should have time resolution less than 10s
        # spectrogram resolutions
        df_Axx = fn[1] - fn[0]
        dt_Axx = tn[1] - tn[0]
        N_PYR = int(np.floor((log(MAX_dt)-log(dt_Axx))/log(2))) +2 # number of pyramids
        AGIxx, AGI_per_bin, AGI_mean, AGI_sum, AGI_dt = maad.features.acousticGradientIndex(X,dt=dt_Axx,
                                                                                 order=1, 
                                                                                 norm='global',
                                                                                 n_pyr=N_PYR,
                                                                                 display=False) 
        AGI = AGI_mean
        AGI23ms = AGI[0]
        AGI46ms = AGI[1]
        AGI93ms = AGI[2]
        AGI186ms= AGI[3]
        AGI371ms= AGI[4]
        AGI743ms= AGI[5]
        AGI1487ms= AGI[6]
        AGI2972ms= AGI[7]
        AGI5944ms= AGI[8]
        AGIdtmax = AGI_dt[np.argmax(AGI)]
        AGIdt = AGI_dt
        AGIperbin = np.asarray(AGI_per_bin).tolist()   

        print("AGI23ms %2.2f" % AGI23ms)
        print("max AGI %2.2f" % AGIdtmax)
    
        # display AGI SPECTROGRAM 
        if DISPLAY :
            index = 2
            fig7, ax7 = plt.subplots()
            # set the paramteers of the figure
            fig7.set_facecolor('w')
            fig7.set_edgecolor('k')
            fig7.set_figheight(4)
            fig7.set_figwidth (13)
            # display image
            _im = ax7.imshow(AGIxx[index], extent=(tn[0], tn[-1], fn[0], fn[-1]), 
                             interpolation='none', origin='lower', 
                             vmax=max(AGIxx[4])*0.5,
                             cmap='gray')
            plt.colorbar(_im, ax=ax7)
            # set the parameters of the subplot
            title = 'AGI with dt=%.2fms' %AGI_dt[index]
            ax7.set_title(title)
            ax7.set_xlabel('Time [sec]')
            ax7.set_ylabel('Frequency [Hz]')
            ax7.axis('tight') 
            fig7.tight_layout()
            # Display the figure now
            plt.show()
        
        ##### Rain Quantification
        """ Rain """  
        X = dBxx_noNoise
    
        # Extract the noise (vertical lines with one pixel width)
        rain = morphology.white_tophat(X, selem=np.ones((1,3)))
        rain = morphology.opening(rain, selem=np.ones((20,1)))
        rain[rain>0] = 1
    
        # indice RAIN
        RAIN = mean(rain)
   
        if DISPLAY : 
            fig_kwargs['title'] = 'Rain noise-like'
            fig_kwargs['vmax'] =max(rain)
            maad.util.plot2D(rain,**fig_kwargs)
    
        ##### number of Events
        """ ROI """        
        # get the mask
        # grey closing (similar to blurr) + Rain as mask to remove rain pixels
    #    X = morphology.closing(PSDxxdB_SansNoise * (1-rain) , selem=disk(3))
    #    X = morphology.closing(PSDxxdB_SansNoise, selem=disk(3))
    #    X = PSDxxdB_SansNoise * (1-rain)
    #    X = filters.gaussian(PSDxxdB_SansNoise * (1-rain), sigma=3, mode="nearest")
    #    X = filters.sobel(X)
    #    
    #    X = filters.sobel(PSDxxdB_SansNoise * (1-rain))
    #    X = filters.gaussian(X, sigma=2, mode="nearest")
    
        X = maad.sound.fir_filter(dBxx_noNoise * (1-rain),kernel=('boxcar',7), axis=1)
    
        if DISPLAY or DISPLAY_PRIORITY :
            fig_kwargs = {'vmax': max(X),
                          'vmin':min(X)}
            maad.util.plot2D(X,**fig_kwargs)
    
        im_mask1 = maad.rois.create_mask(im=X, ext=(0,WAVE_DURATION,0,fs/2), mode_bin = 'relative', 
                                display=DISPLAY, savefig=None, bin_std=4, bin_per=0.5)
    
    #    # binary closing
    #    im_mask2 = morphology.closing(im_mask1, selem=np.ones((3,3))) 
    #    if DISPLAY :
    #        fig_kwargs['title'] = 'MASK2'
    #        fig_kwargs['vmax'] = 1
    #        fig_kwargs['vmin'] = 0
    #        maad.util.plot2D(im_mask2,**fig_kwargs) 
        
    #    im_mask3 = im_mask2 - morphology.white_tophat(im_mask2, selem=np.ones((1,9)))  
    #    if DISPLAY or DISPLAY_PRIORITY :
    #        fig_kwargs['title'] = 'MASK3'
    #        fig_kwargs['vmax'] = 1
    #        fig_kwargs['vmin'] = 0
    #        maad.util.plot2D(im_mask3,**fig_kwargs) 
    #        
    #    # opening
    #    im_mask4 = morphology.opening(im_mask2, selem=np.ones((5,5))) 
    #    if DISPLAY or DISPLAY_PRIORITY :
    #        fig_kwargs['title'] = 'MASK4'
    #        fig_kwargs['vmax'] = 1
    #        fig_kwargs['vmin'] = 0
    #        maad.util.plot2D(im_mask4,**fig_kwargs) 
        
        im_mask = im_mask1   
                 
        # get the mask with rois (im_rois), the bounding bow for each rois (rois_bbox) and the unique index for each rois
        im_rois, rois_bbox, rois_index  = maad.rois.select_rois_auto(im_mask,min_roi=25, ext=(0,WAVE_DURATION,0,fs/2), display= DISPLAY)

        ##### Extract centroids features for each roi
        X = Pxx_noNoise
        rois_bbox, centroid = maad.features.centroid(im=X, im_blobs=im_rois)
    
        if DISPLAY or DISPLAY_PRIORITY :
            X = dBxx_noNoise
            fig_kwargs = {'vmax': max(X),
                          'vmin':min(X)}
            maad.rois.overlay_rois(X, (0,WAVE_DURATION,0,fs/2), rois_bbox.values, **fig_kwargs)

        #ROItotal
        ROItotal = len(centroid)
    
        ##### calcul the area of each roi
        # rectangular area (overestimation)
        area = (rois_bbox.max_y -rois_bbox.min_y) * (rois_bbox.max_x -rois_bbox.min_x)
        # duration in time
        duration = (rois_bbox.max_x -rois_bbox.min_x)
        # bandwith in frequency
        bandwith = (rois_bbox.max_y -rois_bbox.min_y)
        # real area
        area = []
        ind = []    
        for index, label in rois_index :
            area.append(sum(im_rois ==index))
            ind.append(index)
        area = pd.DataFrame({'duration':duration, 'bandwith':bandwith, 'area': area, 'label_num' :ind}) 
    
        # size of im_rois => whole spectrogram
        x,y = im_rois.shape
        total_area = x*y
        # Pourcentage of EVN over the total area
        ROIcover = sum(area.area) / total_area *100
    
        #### create a dataframe with 2 dataframes: centroid and a serie: area
        df_roi = pd.concat([centroid, area], axis=1)
        df_roi.set_index(['label_num'])
    
        if len(centroid)>0 :
            ######### K-mean ##########
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import MeanShift, estimate_bandwidth
        
            # normalize the data (centroid y-coordinate (frequency, not the time) and area)
            X = StandardScaler().fit_transform(df_roi[['y','area','duration', 'bandwith']])
        
            # The following bandwidth can be automatically detected using
            bandwidth = estimate_bandwidth(X, quantile=0.1)
            if bandwidth == 0:
                #ROIunique
                ROIunique = 0 
            else :
                try :
                    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                    ms.fit(X)
                    labels = ms.labels_
                    cluster_centers = ms.cluster_centers_
                    labels_unique = np.unique(labels)
                    #ROIunique
                    ROIunique = len(labels_unique)
                    # add labels to dataframe
                    rois_index = pd.DataFrame(rois_index)
                    rois_index.rename(columns={0:'label_num', 1:'label'}, inplace=True)
                    rois_index.label = labels+1 # +1 to start label at 1 in order to keep 0 for background
                
                    im_label = im_rois
                    for num in rois_index.label_num:
                        im_label[im_label == num] = rois_index.loc[rois_index.label_num == num].label.values[0]
                
                
                    if DISPLAY == True :
                        fig_kwargs = {'ylabel': 'Frequency [Hz]',
                                      'xlabel': 'Time [sec]',
                                      'title':'clusters',
                                      'figsize':(4, 13)}
                        X = im_rois    
                        randcmap = maad.util.rand_cmap(len(labels_unique),first_color_black=True, last_color_black = False)
                        _, fig = maad.util.plot2D (im_label, extent=(tn[0], tn[-1], fn[0], fn[-1]),
                                                   cmap=randcmap, **fig_kwargs)
                except :
                    #ROIunique
                    ROIunique = 0        
        else:
            #ROIunique
            ROIunique = 0 
        
        ####    
    
        df_roi = pd.concat([pd.DataFrame(rois_index, columns =['index', 'label']),df_roi])
    
        ########################################    
        c_clipping.append(sum(abs(wave)>=1))
        c_file.append(filename)
        c_LEQt.append(LEQt)
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
        c_LEQf.append(LEQf)
        c_SNRf.append(SNRf)
        c_BGNf.append(BGNf)
        c_RAIN.append(RAIN)
        c_EAS.append(EAS)
        c_ECU.append(ECU)
        c_ECV.append(ECV)
        c_EPS.append(EPS)
        c_KURT.append(KURT)
        c_SKEW.append(SKEW)
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

        c_AGI.append(AGI)
        c_AGIdt.append(AGIdt)
        c_AGI23ms.append(AGI23ms)
        c_AGI46ms.append(AGI46ms)
        c_AGI93ms.append(AGI93ms)
        c_AGI186ms.append(AGI186ms)
        c_AGI371ms.append(AGI371ms)
        c_AGI743ms.append(AGI743ms)
        c_AGI1487ms.append(AGI1487ms)
        c_AGI2972ms.append(AGI2972ms)
        c_AGI5944ms.append(AGI5944ms)
        c_AGIdtmax.append(AGIdtmax)
    
        c_RAOQ.append(RAOQ)
        c_ROUsurf.append(ROUsurf)
        c_ROItotal.append(ROItotal)
        c_ROIcover.append(ROIcover)
        c_ROIunique.append(ROIunique)
     
        ##### Vector along the frequency axis
        c_frequency.append(fn)
        c_BGNf_per_bin.append(BGNf_per_bin)
        c_Rq_per_bin.append(Rq_per_bin)
        c_Ra_per_bin.append(Ra_per_bin)
        c_ENTsp.append(ENTsp)
        c_ACI_per_bin.append(ACI_per_bin)
        c_ACTspFraction.append(ACTspFraction)
        c_ACTspCount.append(ACTspCount)
        c_ACTspMean.append(ACTspMean)
        c_EVNspSum.append(EVNspSum)
        c_EVNspMean.append(EVNspMean)
        c_EVNspCount.append(EVNspCount)    
        c_AGIperbin.append(AGIperbin)
        c_TFSD_per_bin.append(TFSD_per_bin)
        c_LTS.append(np.mean(Pxx,1))     # Average the Pxx along the axis of time
        
    # =============================================================================
 
    ####### Create the dataframe
    # add new columns to the pd dataframe 
 
    sub_df.loc[:,'filename'] = pd.Series(c_file, index=sub_df.index)   
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
    sub_df.loc[:,'LEQf'] = pd.Series(c_LEQf, index=sub_df.index)   
    sub_df.loc[:,'BGNf'] = pd.Series(c_BGNf, index=sub_df.index)   
    sub_df.loc[:,'RAIN'] = pd.Series(c_RAIN, index=sub_df.index)  
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

    sub_df.loc[:,'AGI23ms'] = pd.Series(c_AGI23ms, index=sub_df.index)   
    sub_df.loc[:,'AGI46ms'] = pd.Series(c_AGI46ms, index=sub_df.index)   
    sub_df.loc[:,'AGI93ms'] = pd.Series(c_AGI93ms, index=sub_df.index)   
    sub_df.loc[:,'AGI186ms'] = pd.Series(c_AGI186ms, index=sub_df.index)   
    sub_df.loc[:,'AGI371ms'] = pd.Series(c_AGI371ms, index=sub_df.index)  
    sub_df.loc[:,'AGI743ms'] = pd.Series(c_AGI743ms, index=sub_df.index)   
    sub_df.loc[:,'AGI1487ms'] = pd.Series(c_AGI1487ms, index=sub_df.index)   
    sub_df.loc[:,'AGI2972ms'] = pd.Series(c_AGI2972ms, index=sub_df.index)
    sub_df.loc[:,'AGI5944ms'] = pd.Series(c_AGI5944ms, index=sub_df.index)
    sub_df.loc[:,'AGIdtmax'] = pd.Series(c_AGIdtmax, index=sub_df.index)
    sub_df.loc[:,'AGI'] = pd.Series(c_AGI, index=sub_df.index)   
    sub_df.loc[:,'AGIdt'] = pd.Series(c_AGIdt, index=sub_df.index)   
    sub_df.loc[:,'RAOQ'] = pd.Series(c_RAOQ, index=sub_df.index)   
    sub_df.loc[:,'ROUsurf'] = pd.Series(c_ROUsurf, index=sub_df.index)   
    sub_df.loc[:,'KURT'] = pd.Series(c_KURT, index=sub_df.index)   
    sub_df.loc[:,'SKEW'] = pd.Series(c_SKEW, index=sub_df.index)   
    sub_df.loc[:,'LEQf'] = pd.Series(c_LEQf, index=sub_df.index)   
    sub_df.loc[:,'ROItotal'] = pd.Series(c_ROItotal, index=sub_df.index)  
    sub_df.loc[:,'ROIcover'] = pd.Series(c_ROIcover, index=sub_df.index)  
    sub_df.loc[:,'ROIunique'] = pd.Series(c_ROIunique, index=sub_df.index)   

    ##### Vector along the frequency axis
    sub_df.loc[:,'frequency'] = pd.Series(c_frequency, index=sub_df.index)
    sub_df.loc[:,'ACTspFraction'] = pd.Series(c_ACTspFraction, index=sub_df.index)   
    sub_df.loc[:,'ACTspCount'] = pd.Series(c_ACTspCount, index=sub_df.index)   
    sub_df.loc[:,'ACTspMean'] = pd.Series(c_ACTspMean, index=sub_df.index)   
    sub_df.loc[:,'EVNspSum'] = pd.Series(c_EVNspSum, index=sub_df.index)   
    sub_df.loc[:,'EVNspMean'] = pd.Series(c_EVNspMean, index=sub_df.index)   
    sub_df.loc[:,'EVNspCount'] = pd.Series(c_EVNspCount, index=sub_df.index)
    sub_df.loc[:,'Rq_per_bin'] = pd.Series(c_Rq_per_bin, index=sub_df.index)
    sub_df.loc[:,'Ra_per_bin'] = pd.Series(c_Ra_per_bin, index=sub_df.index)
    sub_df.loc[:,'ENTsp'] = pd.Series(c_ENTsp, index=sub_df.index)
    sub_df.loc[:,'ACI_per_bin'] = pd.Series(c_ACI_per_bin, index=sub_df.index)
    sub_df.loc[:,'BGNf_per_bin'] = pd.Series(c_BGNf_per_bin, index=sub_df.index)
    sub_df.loc[:,'AGIperbin'] = pd.Series(c_AGIperbin, index=sub_df.index)
    sub_df.loc[:,'TFSD_per_bin'] = pd.Series(c_TFSD_per_bin, index=sub_df.index)
    sub_df.loc[:,'LTS'] = pd.Series(c_LTS, index=sub_df.index)
 

    ######## Save .CSV
    if SAVE == True :
        # First save the vectors (per frequency bins) => Data for False Color Spectro
        sub_df[['file','frequency', 'LTS', 'ACTspFraction','ACTspCount','ACTspMean','EVNspSum', 
                'EVNspMean', 'EVNspCount', 'Rq_per_bin', 'Ra_per_bin', 'BGNf_per_bin',
                'ENTsp','ACI_per_bin','AGIperbin','TFSD_per_bin']].to_csv(path_or_buf=os.path.join(savedir,'per_bin_'+save_csv),sep=',',mode='w',header=True, index=True)
    
        # Then, save scalars
        sub_df.drop(columns=['frequency', 'LTS', 'ACTspFraction','ACTspCount','ACTspMean','EVNspSum', 
                'EVNspMean', 'EVNspCount', 'Rq_per_bin', 'Ra_per_bin', 'BGNf_per_bin',
                'ENTsp','ACI_per_bin','AGIperbin','TFSD']).to_csv(path_or_buf=os.path.join(savedir,save_csv),sep=',',mode='w',header=True, index=True)



.. code-block:: default


    # =============================================================================
    # Data vizualization with pandas
    # ============================================================================
    ######## LOAD CSV
    df_indices = pd.read_csv(os.path.join(savedir,save_csv), sep=',', index_col=0)
    # sort dataframe by date
    df_indices = df_indices.sort_index(axis=0)
    # set index as DatetimeIndex
    df_indices = df_indices.set_index(pd.DatetimeIndex(df_indices.index))

    # table with a summray of the indices value
    df_indices.describe()

    # =============================================================================
    # DISPLAY
    # ============================================================================
    # PLOTLY
    import plotly.io as pio
    import plotly.graph_objects as go
    pio.renderers.default = "browser"

    fig_kwargs = {'figsize': (5, 4),
                  'tight_layout':'tight_layout'}

    colors=['C0',       
            'C1']

    # ROItotal
    fig, ax = plt.subplots(**fig_kwargs)
    ax.plot(df_indices.index, df_indices.ROItotal)
    ax.set_title('Phenology of ROItotal)')
    ax.set_xlabel('Day time')
    ax.set_ylabel('ROItotal [au]')
    ax.grid()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.17), ncol=1)
    fig.tight_layout()

    # ROIcover
    fig, ax = plt.subplots(**fig_kwargs)
    ax.plot(df_indices.index, df_indices.ROIcover)
    ax.set_title('Phenology of ROIcover')
    ax.set_xlabel('Day time')
    ax.set_ylabel('ROIcover [%]')
    ax.grid()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.17), ncol=1)
    fig.tight_layout()

    # LEQf
    fig, ax = plt.subplots(**fig_kwargs)
    ax.plot(df_indices.index, df_indices.LEQf)
    ax.set_title('Phenology of LEQf')
    ax.set_xlabel('Day time')
    ax.set_ylabel('LEQf [dB]')
    ax.grid()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.17), ncol=1)
    fig.tight_layout()

    # BGNf
    fig, ax = plt.subplots(**fig_kwargs)
    ax.plot(df_indices.index, df_indices.BGNf)
    ax.set_title('Phenology of BGNf')
    ax.set_xlabel('Day time')
    ax.set_ylabel('BGNf [dB]')
    ax.grid()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.17), ncol=1)
    fig.tight_layout()

    # RAIN
    fig, ax = plt.subplots(**fig_kwargs)
    ax.plot(df_indices.index, df_indices.RAIN)
    ax.set_title('Phenology of RAIN')
    ax.set_xlabel('Day time')
    ax.set_ylabel('RAIN [%]')
    ax.grid()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.17), ncol=1)
    fig.tight_layout()

    ## SNRf
    fig, ax = plt.subplots(**fig_kwargs)
    ax.plot(df_indices.index, df_indices.SNRf)
    ax.set_title('Phenology of SNRf')
    ax.set_xlabel('Day time')
    ax.set_ylabel('SNRf [dB]')
    ax.grid()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.17), ncol=1)
    fig.tight_layout()




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.000 seconds)


.. _sphx_glr_download__auto_examples_indices_calculation_indice.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: indices_calculation_indice.py <indices_calculation_indice.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: indices_calculation_indice.ipynb <indices_calculation_indice.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
