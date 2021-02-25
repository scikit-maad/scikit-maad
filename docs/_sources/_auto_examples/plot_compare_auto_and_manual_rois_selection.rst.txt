.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download__auto_examples_plot_compare_auto_and_manual_rois_selection.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr__auto_examples_plot_compare_auto_and_manual_rois_selection.py:


Find Regions of interest (ROIs) in a spectrogram
================================================

A spectrogram is a time-frequency (2d) representation of a audio recording. 
Each acoustic event nested in the audio recording is represented by an acoustic
signature. When sounds does not overlap in time and frequency, it is possible
to extract automatically the acoustic signature as a region of interest (ROI) 
by different image processing tools such as binarization, double thresholding,
mathematical morphology tools...

Dependencies: To execute this example you will need to have installed the 
scikit-image, scikit-learn and pandas Python packages.


.. code-block:: default

    # sphinx_gallery_thumbnail_path = '../_images/sphx_glr_compare_auto_and_manual_rois_selection.png'

    import numpy as np
    import pandas as pd
    from maad import sound, rois, features
    from maad.util import power2dB, plot2D, format_features, read_audacity_annot








First, load and audio file and compute the power spectrogram.


.. code-block:: default

    s, fs = sound.load('../data/cold_forest_daylight.wav')

    t0 = 0
    t1 = 20
    f0 = 100
    f1 = 10000
    dB_max = 96

    Sxx_power, tn, fn, ext = sound.spectrogram(s, fs, nperseg=1024, noverlap=1024//2,
                                         fcrop=(f0,f1), tcrop=(t0,t1))

    # Convert the power spectrogram into dB, add dB_max which is the maximum decibel
    # range when quantification bit is 16bits and display the result
    Sxx_db = power2dB(Sxx_power) + dB_max
    plot2D(Sxx_db, **{'vmin':0, 'vmax':dB_max, 'extent':ext})




.. image:: /_auto_examples/images/sphx_glr_plot_compare_auto_and_manual_rois_selection_001.png
    :alt: Spectrogram
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Volumes/lacie_macosx/numerical_analysis_toolbox/scikit-maad/maad/util/visualization.py:280: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      if now: plt.show()




Then, relevant acoustic events are extracted directly from the power 
spectrogram based on a double thresholding technique. The result is binary
image called a mask. Double thresholding technique is more sophisticated than
basic thresholding based on a single value. First, a threshold selects pixels
with high value (i.e. high acoustic energy). They should belong to an acoustic
event. They are called seeds. From these seeds, we aggregate pixels connected
to the seed with value higher than the second threslhold. These new pixels 
become seed and the aggregating process continue until no more new pixels are
aggregated, meaning that there is no more connected pixels with value upper 
than the second threshold value.


.. code-block:: default


    # First we remove the stationary background in order to increase the contrast [1]
    # Then we convert the spectrogram into dB
    Sxx_power_noNoise= sound.median_equalizer(Sxx_power, display=True, **{'extent':ext})
    Sxx_db_noNoise = power2dB(Sxx_power_noNoise)

    # Then we smooth the spectrogram in order to facilitate the creation of masks as
    # small sparse details are merged if they are close to each other
    Sxx_db_noNoise_smooth = sound.smooth(Sxx_db_noNoise, std=0.5, 
                             display=True, savefig=None, 
                             **{'vmin':0, 'vmax':dB_max, 'extent':ext})

    # Then we create a mask (i.e. binarization of the spectrogram) by using the 
    # double thresholding technique
    im_mask = rois.create_mask(im=Sxx_db_noNoise_smooth, mode_bin ='relative', 
                               bin_std=8, bin_per=0.5,
                               verbose=False, display=False)

    # Finaly, we put together pixels that belong to the same acoustic event, and 
    # remove very small events (<=25 pixel²)
    im_rois, df_rois = rois.select_rois(im_mask, min_roi=25, max_roi=None, 
                                     display= True,
                                     **{'extent':ext})
    
    # format dataframe df_rois in order to convert pixels into time and frequency
    df_rois = format_features(df_rois, tn, fn)

    # overlay bounding box on the original spectrogram
    ax0, fig0 = rois.overlay_rois(Sxx_db, df_rois, **{'vmin':0, 'vmax':dB_max, 'extent':ext})

    # Compute and visualize centroids
    df_centroid = features.centroid_features(Sxx_db, df_rois, im_rois)
    df_centroid = format_features(df_centroid, tn, fn)
    ax0, fig0 = features.overlay_centroid(Sxx_db, df_centroid, savefig=None,
                                          **{'vmin':0,'vmax':dB_max,'extent':ext,'ms':4, 
                                             'marker':'+', 'fig':fig0, 'ax':ax0})





.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /_auto_examples/images/sphx_glr_plot_compare_auto_and_manual_rois_selection_002.png
          :alt: Spectrogram without stationnary noise
          :class: sphx-glr-multi-img

    *

      .. image:: /_auto_examples/images/sphx_glr_plot_compare_auto_and_manual_rois_selection_003.png
          :alt: Orignal Spectrogram, Blurred Spectrogram (std=0.5)
          :class: sphx-glr-multi-img

    *

      .. image:: /_auto_examples/images/sphx_glr_plot_compare_auto_and_manual_rois_selection_004.png
          :alt: Selected ROIs
          :class: sphx-glr-multi-img

    *

      .. image:: /_auto_examples/images/sphx_glr_plot_compare_auto_and_manual_rois_selection_005.png
          :alt: ROIs Overlay
          :class: sphx-glr-multi-img





Let's compare with the manual annotation (Ground Truth GT) obtained with 
Audacity software.
Each acoustic signature is manually selected and labeled. All similar acoustic 
signatures are labeled with the same name


.. code-block:: default

    df_rois_GT = read_audacity_annot('../data/cold_forest_daylight_label.txt')  ## annotations using Audacity

    # drop rows with frequency and time outside of tn and fn
    df_rois_GT = df_rois_GT[(df_rois_GT.min_t >= tn.min()) & 
                            (df_rois_GT.max_t <= tn.max()) & 
                            (df_rois_GT.min_f >= fn.min()) & 
                            (df_rois_GT.max_f <= fn.max())]

    # format dataframe df_rois in order to convert time and frequency into pixels
    df_rois_GT = format_features(df_rois_GT, tn, fn)

    # overlay bounding box on the original spectrogram
    ax1, fig1 = rois.overlay_rois(Sxx_db, df_rois_GT, **{'vmin':0,'vmax':dB_max,'extent':ext})
    
    # Compute and visualize centroids
    df_centroid_GT = features.centroid_features(Sxx_db, df_rois_GT)
    df_centroid_GT = format_features(df_centroid_GT, tn, fn)
    ax1, fig1 = features.overlay_centroid(Sxx_db, df_centroid_GT, savefig=None, 
                                          **{'vmin':0,'vmax':dB_max,'extent':ext,
                                             'ms':2, 'marker':'+','color':'blue',
                                             'fig':fig1, 'ax':ax1})

    # print informations about the rois
    print ('Total number of ROIs : %2.0f' %len(df_rois_GT))
    print ('Number of different ROIs : %2.0f' %len(np.unique(df_rois_GT['label'])))




.. image:: /_auto_examples/images/sphx_glr_plot_compare_auto_and_manual_rois_selection_006.png
    :alt: ROIs Overlay
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Total number of ROIs : 49
    Number of different ROIs :  5




Now we cluster the ROIS depending on 3 ROIS features :
- centroid_f : frequency position of the roi centroid 
- duration_t : duration of the roi
- bandwidth_f : frequency bandwidth of the roi
The clustering is done by the so-called KMeans clustering algorithm.
The number of attended clustering is the number of clusters found with 
manual annotation.
Finally, each rois is labeled with the corresponding cluster number predicted
by KMeans


.. code-block:: default

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # select features to perform KMeans clustering
    FEATURES = ['centroid_f','duration_t','bandwidth_f','area_tf']

    # Prepare the features in order to have zero mean and same variance
    X = StandardScaler().fit_transform(df_centroid[FEATURES])

    # perform KMeans with the same number of clusters as with the manual annotation  
    NN_CLUSTERS = len(np.unique(df_rois_GT['label'])) 
    labels = KMeans(n_clusters=NN_CLUSTERS, random_state=0).fit_predict(X)

    # Replace the unknow label by the cluster number predicted by KMeans
    df_centroid['label'] = [str(i) for i in labels] 

    # overlay color bounding box corresponding to the label, and centroids
    # on the original spectrogram
    ax2, fig2 = rois.overlay_rois(Sxx_db, df_centroid, **{'vmin':0,'vmax':dB_max,'extent':ext})
    ax2, fig2 = features.overlay_centroid(Sxx_db, df_centroid, savefig=None, 
                                          **{'vmin':0,'vmax':dB_max,'extent':ext,'ms':2, 
                                             'fig':fig2, 'ax':ax2})




.. image:: /_auto_examples/images/sphx_glr_plot_compare_auto_and_manual_rois_selection_007.png
    :alt: ROIs Overlay
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    //miniconda3/lib/python3.7/importlib/_bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
      return f(*args, **kwds)




It is possible to extract Rois directly from the audio waveform without 
computing the spectrogram. This works well if there is no big overlap between
each acoustic signature and you 
First, we have to define the frequency bandwidth where to find acoustic events
In our example, there are clearly 3 frequency bandwidths (low : l, medium:m
and high : h). 
We know that we have mostly short (ie. s) acoustic events in low, med and high
frequency bandwidths but also a long (ie l) acoustic events in med.
To extract 


.. code-block:: default

       
    df_rois_sh = rois.find_rois_cwt(s, fs, flims=[7000, 8000], tlen=0.2, th=0.000001)
    df_rois_sm = rois.find_rois_cwt(s, fs, flims=[3500, 5500], tlen=0.2, th=0.000001)
    df_rois_lm = rois.find_rois_cwt(s, fs, flims=[2000, 7500], tlen=2,   th=0.0001)
    df_rois_sl = rois.find_rois_cwt(s, fs, flims=[1800, 3000], tlen=0.2, th=0.000001)

    ## concat df
    df_rois_WAV =pd.concat([df_rois_sh, df_rois_sm, df_rois_lm, df_rois_sl], ignore_index=True)

    # drop rows with frequency and time outside of tn and fn
    df_rois_WAV = df_rois_WAV[(df_rois_WAV.min_t >= tn.min()) & 
                                          (df_rois_WAV.max_t <= tn.max()) & 
                                          (df_rois_WAV.min_f >= fn.min()) & 
                                          (df_rois_WAV.max_f <= fn.max())]
    
    # get features: centroid, 
    df_rois_WAV = format_features(df_rois_WAV, tn, fn)
    df_centroid_WAV = features.centroid_features(Sxx_db, df_rois_WAV)

    ax3, fig3 = rois.overlay_rois(Sxx_db, df_rois_WAV, **{'vmin':0,'vmax':dB_max,
                                                          'extent':ext})
    df_centroid_WAV = format_features(df_centroid_WAV, tn, fn)
    ax3, fig3 = features.overlay_centroid(Sxx_db, df_centroid_WAV, savefig=None, 
                                          **{'vmin':0,'vmax':dB_max,'extent':ext,
                                             'ms':2, 'fig':fig3, 'ax':ax3})




.. image:: /_auto_examples/images/sphx_glr_plot_compare_auto_and_manual_rois_selection_008.png
    :alt: ROIs Overlay
    :class: sphx-glr-single-img





Prepare the features in order to have zero mean and same variance


.. code-block:: default

    X = StandardScaler().fit_transform(df_centroid_WAV[FEATURES])

    # perform KMeans with the same number of clusters as with the manual annotation  
    labels = KMeans(n_clusters=NN_CLUSTERS, random_state=0).fit_predict(X)

    # Replace the unknow label by the cluster number predicted by KMeans
    df_centroid_WAV['label'] = [str(i) for i in labels] 

    # overlay color bounding box corresponding to the label, and centroids
    # on the original spectrogram
    ax4, fig4 = rois.overlay_rois(Sxx_db, df_centroid_WAV, **{'vmin':0,'vmax':dB_max,
                                                              'extent':ext})
    ax4, fig4 = features.overlay_centroid(Sxx_db, df_centroid_WAV, savefig=None, 
                                          **{'vmin':0,'vmax':dB_max,'extent':ext,
                                             'ms':2,'fig':fig4, 'ax':ax4})




.. image:: /_auto_examples/images/sphx_glr_plot_compare_auto_and_manual_rois_selection_009.png
    :alt: ROIs Overlay
    :class: sphx-glr-single-img





References
-----------
1.Towsey, M., 2013b. Noise Removal from Wave-forms and Spectrograms Derived from
  Natural Recordings of the Environment. Queensland University of Technology,
  Brisbane


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  3.589 seconds)


.. _sphx_glr_download__auto_examples_plot_compare_auto_and_manual_rois_selection.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_compare_auto_and_manual_rois_selection.py <plot_compare_auto_and_manual_rois_selection.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_compare_auto_and_manual_rois_selection.ipynb <plot_compare_auto_and_manual_rois_selection.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
