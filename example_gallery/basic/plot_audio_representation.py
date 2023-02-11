#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio representation
====================

An audio signal can be represented in both, temporal and spectral domains.
These representations are complementary and fundamental to understand the audio
signal characteristics. In this introductory example we will load an audio
signal, apply basic transformations to better understand its features in
time and frequency.
"""

import matplotlib.pyplot as plt

from example_gallery._paths import DATA_PATH
from maad import sound
from maad import util


def plot_audio_representation(filename):
    # Load an audio file
    signal, sample_rate = sound.load(filename)
    # Plot and show the waveform
    util.plot_wave(signal, sample_rate)
    plt.show()

    # It can be noticed that in this audio there are four consecutive songs
    # of the spinetail
    # *Cranioleuca erythorps*, every song lasting of approximately two
    # seconds.

    # Trim the signal to zoom in on the details of the song.
    signal_trimmed = sound.trim(signal, sample_rate, 5, 8)

    # Compute the envelope of the trimmed signal
    envelope = sound.envelope(signal_trimmed, mode='fast', Nt=128)

    # Compute the Fourier and short - time Fourier transforms
    power_spectral_density_estimate, sample_frequency_indices = \
        sound.spectrum(
            signal,
            sample_rate,
            nperseg=1024,
            method='welch',
        )
    Sxx_power_or_amplitude, time_vector_of_the_horizontal_x_axis,\
        frequency_vector_of_the_vertical_y_axis, box_extend = \
        sound.spectrogram(
            signal_trimmed,
            sample_rate,
            window='hann',
            nperseg=1024,
            noverlap=512,
        )

    # Visualize the signal characteristics in the temporal and spectral domains
    figure, axes = plt.subplots(
        4,
        1,
        figsize=(8, 10),
    )
    util.plot_wave(
        signal_trimmed,
        sample_rate,
        ax=axes[0],
    )
    util.plot_wave(
        envelope,
        sample_rate,
        ax=axes[1],
    )
    util.plot_spectrum(
        power_spectral_density_estimate,
        sample_frequency_indices,
        ax=axes[2],
    )
    util.plot_spectrogram(
        Sxx_power_or_amplitude,
        extent=box_extend,
        ax=axes[3],
        colorbar=False,
    )
    plt.show()


def main():
    filename = str(DATA_PATH / 'spinetail.wav')
    plot_audio_representation(filename=filename)


if __name__ == '__main__':
    main()
