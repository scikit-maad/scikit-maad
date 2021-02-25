.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download__auto_examples_plot_nmf_and_false_color_spectrogram.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr__auto_examples_plot_nmf_and_false_color_spectrogram.py:


Signal decomposition and false-color spectrograms
===============================================

Soundscapes result from a combination of multiple signals that are mixed-down
into a single time-series. Unmixing these signals can be regarded as an 
important preprocessing step for further analyses of individual components.
Here, we will combine the robust characterization capabilities of 
the bidimensional wavelets [1] with an advanced signal decomposition tool, the 
non-negative-matrix factorization (NMF)[2]. NMF is a widely used tool to analyse
high-dimensional that automatically extracts sparse and meaningfull components
of non-negative matrices. Audio spectrograms are in essence sparse and 
non-negative matrices, and hence well suited to be decomposed with NMF. This 
decomposition can be further used to generate false-color spectrograms to 
rapidly identify patterns in soundscapes and increase the interpretability of 
the signal [3]. This example shows how to use the scikit-maad package to easily 
decompose audio signals and visualize false-colour spectrograms.

Dependencies: To execute this example you will need to have instaled the 
scikit-image and scikit-learn Python packages.


.. code-block:: default

    # sphinx_gallery_thumbnail_path = '../_images/sphx_glr_plot_nmf_and_false_color_spectrogram_003.png'
    import numpy as np
    import matplotlib.pyplot as plt
    from maad import sound, features
    from maad.util import power2dB, plot2D
    from skimage import transform
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import NMF





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    //miniconda3/lib/python3.7/importlib/_bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
      return f(*args, **kwds)
    //miniconda3/lib/python3.7/importlib/_bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
      return f(*args, **kwds)
    //miniconda3/lib/python3.7/importlib/_bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
      return f(*args, **kwds)
    //miniconda3/lib/python3.7/importlib/_bootstrap.py:219: RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
      return f(*args, **kwds)




First, load and audio file and compute the spectrogram.


.. code-block:: default

    s, fs = sound.load('../data/spinetail.wav')
    Sxx, tn, fn, ext = sound.spectrogram(s, fs, nperseg=1024, noverlap=512)

    Sxx_db = power2dB(Sxx, db_range=70)
    Sxx_db = transform.rescale(Sxx_db, 0.5, anti_aliasing=True, multichannel=False)
    plot2D(Sxx_db, **{'figsize':(4,10),'extent':(tn[0], tn[-1], fn[0], fn[-1])})




.. image:: /_auto_examples/images/sphx_glr_plot_nmf_and_false_color_spectrogram_001.png
    :alt: Spectrogram
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /Volumes/lacie_macosx/numerical_analysis_toolbox/scikit-maad/maad/util/visualization.py:303: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      if now: plt.show()




Then, compute feature with ``shape_features_raw`` to get the raw output of the 
spectrogram filtered by the filterbank composed of 2D Gabor wavelets. This
raw output can be fed to the NMF algorithm to decompose the spectrogram into
elementary basis spectrograms.


.. code-block:: default


    shape_im, params = features.shape_features_raw(Sxx_db, resolution='low')

    # Format the output as an array for decomposition
    X = np.array(shape_im).reshape([len(shape_im), Sxx_db.size]).transpose()

    # Decompose signal using non-negative matrix factorization
    Y = NMF(n_components=3, init='random', random_state=0).fit_transform(X)

    # Normalize the data and combine the three NMF basis spectrograms and the
    # intensity spectrogram into a single array to fit the RGBA color model. RGBA
    # stands for Red, Green, Blue and Alpha, where alpha indicates how opaque each
    # pixel is.

    Y = MinMaxScaler(feature_range=(0,1)).fit_transform(Y)
    intensity = 1 - (Sxx_db - Sxx_db.min()) / (Sxx_db.max() - Sxx_db.min())
    plt_data = Y.reshape([Sxx_db.shape[0], Sxx_db.shape[1], 3])
    plt_data = np.dstack((plt_data, intensity))








Finally, plot the resulting basis spectrogram as separate elements and 
combine them to produce a false-colour spectrogram using the RGBA color 
model.


.. code-block:: default


    fig, axes = plt.subplots(3,1, figsize=(10,8))
    for idx, ax in enumerate(axes):
        ax.imshow(plt_data[:,:,idx], origin='lower', aspect='auto', 
                  interpolation='bilinear')
        ax.set_axis_off()
        ax.set_title('Basis ' + str(idx+1))




.. image:: /_auto_examples/images/sphx_glr_plot_nmf_and_false_color_spectrogram_002.png
    :alt: Basis 1, Basis 2, Basis 3
    :class: sphx-glr-single-img





The first basis spectrogram shows fine and rapid modulations of the signal.
Both signals have these features and hence both are delineated in this
basis. The second basis highlights the short calls on the background, and the 
third component highlights the longer vocalizations of the spinetail. 
The three components can be mixed up to compose a false-colour spectrogram
where it can be easily distinguished the different sound sources by color.


.. code-block:: default


    fig, ax = plt.subplots(2,1, figsize=(10,6))
    ax[0].imshow(Sxx_db, origin='lower', aspect='auto', interpolation='bilinear', cmap='gray')
    ax[0].set_axis_off()
    ax[0].set_title('Spectrogram')
    ax[1].imshow(plt_data, origin='lower', aspect='auto', interpolation='bilinear')
    ax[1].set_axis_off()
    ax[1].set_title('False-color spectrogram')




.. image:: /_auto_examples/images/sphx_glr_plot_nmf_and_false_color_spectrogram_003.png
    :alt: Spectrogram, False-color spectrogram
    :class: sphx-glr-single-img





References
-----------
1. Sifre, L., & Mallat, S. (2013). Rotation, scaling and deformation invariant scattering for texture discrimination. Computer Vision and Pattern Recognition (CVPR), 2013 IEEE Conference On, 1233–1240. http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=6619007
2. Lee, D., & Sueng, S. (1999). Learning the parts of objects by non-negative matrix factorization. Nature, 401, 788–791. https://doi.org/10.1038/44565
3. Towsey, M., Znidersic, E., Broken-Brow, J., Indraswari, K., Watson, D. M., Phillips, Y., Truskinger, A., & Roe, P. (2018). Long-duration, false-colour spectrograms for detecting species in large audio data-sets. Journal of Ecoacoustics, 2(1), 1–1. https://doi.org/10.22261/JEA.IUSWUI


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  1.737 seconds)


.. _sphx_glr_download__auto_examples_plot_nmf_and_false_color_spectrogram.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_nmf_and_false_color_spectrogram.py <plot_nmf_and_false_color_spectrogram.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_nmf_and_false_color_spectrogram.ipynb <plot_nmf_and_false_color_spectrogram.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
