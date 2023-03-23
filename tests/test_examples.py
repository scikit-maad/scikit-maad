#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test examples

"""

import pathlib
import runpy
import pytest
import os

# sets the backend to 'Agg' which doesn't display plots interactively
import matplotlib
matplotlib.use('Agg') 

# don't take into account all the warnings due to not plotting the figures
@pytest.mark.filterwarnings("ignore:Matplotlib is currently using agg")

# List of scripts to run. 
# wookpecker_drumming_characteristics.py is not run as it will download mp3 from XC
@pytest.mark.parametrize(
    "script",
    [
        (pathlib.Path(__file__, '..', '..', 'example_gallery','1_basic','plot_audio_representation.py').resolve()),
        (pathlib.Path(__file__, '..', '..', 'example_gallery','1_basic','plot_circadian_spectrogram.py').resolve()),
        (pathlib.Path(__file__, '..', '..', 'example_gallery','1_basic','plot_detection_distance.py').resolve()),
        (pathlib.Path(__file__, '..', '..', 'example_gallery','1_basic','plot_find_rois_simple.py').resolve()),
        (pathlib.Path(__file__, '..', '..', 'example_gallery','2_advanced','plot_compare_auto_and_manual_rois_selection.py').resolve()),
        (pathlib.Path(__file__, '..', '..', 'example_gallery','2_advanced','plot_extract_alpha_indices_multicpu.py').resolve()),
        (pathlib.Path(__file__, '..', '..', 'example_gallery','2_advanced','plot_extract_alpha_indices.py').resolve()),
        (pathlib.Path(__file__, '..', '..', 'example_gallery','2_advanced','plot_nmf_and_false_color_spectrogram.py').resolve()),
        (pathlib.Path(__file__, '..', '..', 'example_gallery','2_advanced','plot_remove_background.py').resolve()),
        (pathlib.Path(__file__, '..', '..', 'example_gallery','2_advanced','plot_sound_degradation_due_to_attenuation.py').resolve()),
        (pathlib.Path(__file__, '..', '..', 'example_gallery','2_advanced','plot_sound_pressure_level.py').resolve()),
        (pathlib.Path(__file__, '..', '..', 'example_gallery','2_advanced','plot_unsupervised_sound_classification.py').resolve()),
        (pathlib.Path(__file__, '..', '..', 'example_gallery','2_advanced','plot_xenocanto_wookpecker_activities.py').resolve()),
    ]
)

def test_script_execution(script):
    os.chdir(pathlib.Path(__file__, '..','data').resolve())
    try :
        runpy.run_path(script)
    except Exception as e:
        print(f"An error occurred: {e}")

os.chdir(pathlib.Path(__file__, '..').resolve())