.. _audio-dataset:

Audio dataset
^^^^^^^^^^^^^

*scikit-maad* provides a diversified audio dataset with samples from tropical (French Guiana and Colombia) and temperate habitats (France). All files were recorded using passive acoustic sensors. More sounds from other ecosystems will be added soon.

======================= ========== ========== ========================= =======================================================
Code name               Date       Time       Location                  Habitat
======================= ========== ========== ========================= =======================================================
spinetail               2018-09-19 11:00      Valle del Cauca, Colombia Scrub vegetation near cloud forest edge
cold_forest_daylight    2019-04-12 11:00      Jura, France              Temperate cold mountain forest covered by spruce forest
cold_forest_night       2019-08-25 04:00      Jura, France              Temperate cold mountain forest covered by spruce forest
rock_savanna_night      2014-12-05 22:17      French Guiana             Tropical rock savanna on a granite hill
tropical_forest_morning 2019-06-29 6:15       French Guiana             Lowland tropical rainforest
indices                 2019-05-22 24-h cycle Jura, France              Temperate cold mountain forest covered by spruce forest
======================= ========== ========== ========================= =======================================================

Downloading the audio dataset
-----------------------------

Most of the examples found in modules and in the example gallery use these audio recordings. To test and run the scripts, we recommend downloading these files to your local hard drive. The files can be downloaded from the GitHub repository (https://github.com/scikit-maad/scikit-maad). Once in your local hard drive, the audio examples can be loaded by adding the path to these files:

>>> from maad import sound
>>> s, fs = sound.load('<PATH TO AUDIO DATA>/spinetail.wav')


Downloading a single audio example
----------------------------------

*scikit-maad* also provide a simple function to load audio examples using their URL. Note however, that the function will download the file each time you run the script.

>>> from maad import sound
>>> s, fs = sound.load_url('spinetail')
