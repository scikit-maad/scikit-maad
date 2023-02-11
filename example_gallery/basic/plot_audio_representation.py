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
    util.plot_wave(
        s=signal,
        fs=sample_rate,
    )
    plt.show()

    # It can be noticed that in this audio there are four consecutive songs
    # of the spinetail
    # *Cranioleuca erythorps*, every song lasting of approximately two
    # seconds.

    # Trim the signal to zoom in on the details of the song.
    signal_trimmed = sound.trim(
        s=signal,
        fs=sample_rate,
        min_t=5,
        max_t=8,
    )

    # Compute the envelope of the trimmed signal
    envelope_transformation_mode = 'fast'
    frame_size = 128
    envelope = sound.envelope(
        s=signal_trimmed,
        mode=envelope_transformation_mode,
        Nt=frame_size,
    )

    # Compute the Fourier and short-time Fourier transforms
    sequence_size_in_samples = 1024
    spectrum_power_density_estimation_method = 'welch'
    power_spectral_density_estimate, sample_frequency_indices = \
        sound.spectrum(
            s=signal,
            fs=sample_rate,
            nperseg=sequence_size_in_samples,
            method=spectrum_power_density_estimation_method,
        )

    # Compute the spectrogram
    spectrogram_window_type = 'hann'
    spectrogram_fft_window_size_in_samples = 1024
    spectrogram_segment_overlap_in_samples = 512
    Sxx_power_or_amplitude, time_vector_of_the_horizontal_x_axis,\
        frequency_vector_of_the_vertical_y_axis, box_extend = \
        sound.spectrogram(
            x=signal_trimmed,
            fs=sample_rate,
            window=spectrogram_window_type,
            nperseg=spectrogram_fft_window_size_in_samples,
            noverlap=spectrogram_segment_overlap_in_samples,
        )

    # Visualize the signal characteristics in the temporal and spectral domains
    figure, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(8, 10),
    )
    util.plot_wave(
        s=signal_trimmed,
        fs=sample_rate,
        ax=axes[0],
    )
    util.plot_wave(
        s=envelope,
        fs=sample_rate,
        ax=axes[1],
    )
    util.plot_spectrum(
        pxx=power_spectral_density_estimate,
        f_idx=sample_frequency_indices,
        ax=axes[2],
    )
    util.plot_spectrogram(
        Sxx=Sxx_power_or_amplitude,
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
