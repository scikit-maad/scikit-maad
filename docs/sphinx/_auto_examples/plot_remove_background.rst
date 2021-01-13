.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download__auto_examples_plot_remove_background.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr__auto_examples_plot_remove_background.py:


Remove background noise from audio with signal processing tools
===============================================================

This example shows different ways to remove background noise directly from
the spectrogram.
We use the sharpness metric to have a quantitative estimation of how well is 
the noise reduction. This metric gives partial information. Other metrics 
should be use in complement.


.. code-block:: default

    # sphinx_gallery_thumbnail_path = '../_images/sphx_glr_remove_background.png'

    from maad.util import plot2D, power2dB
    from maad.sound import (load, spectrogram, 
                           remove_background, median_equalizer, 
                           remove_background_morpho, 
                           remove_background_along_axis, sharpness)
    import numpy as np

    from timeit import default_timer as timer

    import matplotlib.pyplot as plt








First, we load the audio file and take its spectrogram.
The linear spectrogram is then transformed into dB. The dB range is  96dB 
which is the maximum dB range value for a 16bits audio recording. We add
96dB in order to get have only positive values in the spectrogram


.. code-block:: default

    s, fs = load('../data/tropical_forest_morning.wav')
    #s, fs = load('../data/cold_forest_night.wav')
    Sxx, tn, fn, ext = spectrogram(s, fs, fcrop=[0,20000], tcrop=[0,60])
    Sxx_dB = power2dB(Sxx, db_range=96) + 96





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Volumes/lacie_macosx/numerical_analysis_toolbox/scikit-maad/maad/sound/io_.py:109: WavFileWarning: Chunk (non-data) not understood, skipping it.
      fs, s = wavfile.read(filename)




We plot the original spectrogram.


.. code-block:: default

    plot2D(Sxx_dB, extent=ext, title='original',
           vmin=np.median(Sxx_dB), vmax=np.median(Sxx_dB)+40)

    print ("Original sharpness : %2.3f" % sharpness(Sxx_dB))




.. image:: /_auto_examples/images/sphx_glr_plot_remove_background_001.png
    :alt: original
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Volumes/lacie_macosx/numerical_analysis_toolbox/scikit-maad/maad/util/visualization.py:280: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      if now: plt.show()
    Original sharpness : 2.247




Test the function "remove_background"


.. code-block:: default

    start = timer()
    X1, noise_profile1, _ = remove_background(Sxx_dB)
    elapsed_time = timer() - start
    print("---- test remove_background -----")
    print("duration %2.3f s" % elapsed_time)
    print ("sharpness : %2.3f" % sharpness(X1))

    plot2D(X1, extent=ext, title='remove_background',
           vmin=np.median(X1), vmax=np.median(X1)+40)




.. image:: /_auto_examples/images/sphx_glr_plot_remove_background_002.png
    :alt: remove_background
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    ---- test remove_background -----
    duration 0.144 s
    sharpness : 1.706




Test the function "median_equalizer"


.. code-block:: default

    start = timer()
    X2 = median_equalizer(Sxx)
    X2 = power2dB(X2)
    elapsed_time = timer() - start
    print("---- test median_equalizer -----")
    print("duration %2.3f s" % elapsed_time)
    print ("sharpness : %2.3f" %sharpness(X2))

    plot2D(X2,extent=ext, title='median_equalizer',
           vmin=np.median(X2), vmax=np.median(X2)+40)




.. image:: /_auto_examples/images/sphx_glr_plot_remove_background_003.png
    :alt: median_equalizer
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    ---- test median_equalizer -----
    duration 0.085 s
    sharpness : 1.502




Test the function "remove_background_morpho"


.. code-block:: default

    start = timer()
    X3, noise_profile3,_ = remove_background_morpho(Sxx_dB, q=0.95) 
    elapsed_time = timer() - start
    print("---- test remove_background_morpho -----")
    print("duration %2.3f s" % elapsed_time)
    print ("sharpness : %2.3f" %sharpness(X3))

    plot2D(X3, extent=ext, title='remove_background_morpho',
           vmin=np.median(X3), vmax=np.median(X3)+40)




.. image:: /_auto_examples/images/sphx_glr_plot_remove_background_004.png
    :alt: remove_background_morpho
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    //miniconda3/lib/python3.7/importlib/_bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
      return f(*args, **kwds)
    ---- test remove_background_morpho -----
    duration 1.247 s
    sharpness : 1.093




Test the function "remove_background_along_axis"


.. code-block:: default

    start = timer()
    X4, noise_profile4 = remove_background_along_axis(Sxx_dB,mode='median', axis=1) 
    #X4 = power2dB(X4) 
    elapsed_time = timer() - start
    print("---- test remove_background_along_axis -----")
    print("duration %2.3f s" % elapsed_time)
    print ("sharpness : %2.3f" %sharpness(X4))

    plot2D(X4,  extent=ext, title='remove_background_along_axis',
           vmin=np.median(X4), vmax=np.median(X4)+40)

    plt.tight_layout()



.. image:: /_auto_examples/images/sphx_glr_plot_remove_background_005.png
    :alt: remove_background_along_axis
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    ---- test remove_background_along_axis -----
    duration 0.029 s
    sharpness : 1.166





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  4.200 seconds)


.. _sphx_glr_download__auto_examples_plot_remove_background.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_remove_background.py <plot_remove_background.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_remove_background.ipynb <plot_remove_background.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
