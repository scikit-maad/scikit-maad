#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Circadian soundscape
====================

When dealing with large amounts of audio recordings, visualization plays a key
role to evidence the main patterns in the data. In this example we show how
to easily combine 96 audio files to build a visual representation of a 24-hour
natural soundscape.
"""
import matplotlib.pyplot as plt

from maad import sound, util

from example_gallery._paths import DATA_PATH


def plot_circadian_spectrogram():
    # Collect a list of trimmed signals from the sound files
    signals = []
    for wav_filename in sorted((DATA_PATH / 'indices').glob('*.wav')):
        signal, sample_rate = sound.load(wav_filename)
        signals.append(signal)

    # Combine all audio recordings into one
    signal = util.crossfade_list(
        s_list=signals,
        fs=sample_rate,
        fade_len=0.0,
    )

    # Compute spectrogram of the mixed audio
    Sxx, _, _, _ = sound.spectrogram(
        x=signal,
        fs=sample_rate,
        window='hann',
        nperseg=1024,
        noverlap=512,
    )

    # Plot the spectrogram
    _, axes = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(10, 3),
    )
    util.plot_spectrogram(
        Sxx=Sxx,
        extent=[0, 24, 0, 11],
        ax=axes,
        db_range=80,
        gain=25,
        colorbar=False,
    )
    axes.set_xlabel('Time [Hours]')
    axes.set_xticks(range(0, 25, 4))
    plt.show()

    # At dawn (5-10 h) and dusk (19-22 h), we can see clearly the bird chorus.
    # At low frequencies, we can see wind noise and propeller airplane sounds.


if __name__ == '__main__':
    plot_circadian_spectrogram()
