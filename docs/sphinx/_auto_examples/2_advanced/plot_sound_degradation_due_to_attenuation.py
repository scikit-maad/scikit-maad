#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simulation of sound degradation due to geometric, atmospheric and habitat attenuation
=====================================================================================

When sound travels in air (or in water), initial acoustic signature may change 
regarding distances due to attenuation. Sound attenuation in natural environment 
is very complex to model. We propose to reduce the complexity by decomposing 
the sound attenuation to three main sources of attenuation : 
- the geometric attenuation (Ageo) also known as spreading loss or geometric 
dispersion, 
- the atmospheric attenuation (Aatm) 
- and the habitat attenuation (Ahab). The later encompasses several sources of 
attenuation and might be seen as a proxy.

"""
# sphinx_gallery_thumbnail_path = '../_images/sphx_glr_plot_sound_degradation_due_to_attenuation.png'

from maad import sound, spl, util

#%%
# When working with sound attenuation in real environment, it is essential to
# know the sound pressure level at a certain distance (generally 1m). Here
# we used spinetail sound. We assume that its sound level @1m = 85dB SPL.
# It is also essential to be able to tranform the signal recorded by the
# recorder into sound pressure level. Here, the audio recorder is a SM4 
# (Wildlife Acoustics) which is an autonomous recording unit (ARU) from
# Wildlife. The sensitivity of the internal microphone is -35dBV and the 
# maximal voltage converted by the analog to digital convertor (ADC) is 2Vpp 
# (peak to peak). The gain used for the recording is a combination of
# the internal pre-amplifier of the SM4, which is 26dB and the adjustable gain
# which was 16dB. So the total gain applied to the signal is : 42dB

# We load the sound
w, fs = sound.load('../../data/spinetail.wav') 
# We convert the sound into sound pressure level (Pa)
p0 = spl.wav2pressure(wave=w, gain=42, Vadc=2, sensitivity=-35)
# We select part of the sound with the spinetail signal
p0_sig = p0[int(5.68*fs):int(7.48*fs)] 
# We select part of the sound with background
p0_noise = p0[int(8.32*fs):int(10.12*fs)] 
# We convert both signals into spectrograms
Sxx_power, tn, fn, ext = sound.spectrogram(p0_sig ,fs)
Sxx_power_noise, tn, fn, ext = sound.spectrogram(p0_noise ,fs)
# We convert both spectrograms into dB. We choose a dB range of 96dB which
# is the maximal range for a 16 bits signal.
Sxx_dB = util.power2dB(Sxx_power, db_range=96) + 96
Sxx_dB_noise = util.power2dB(Sxx_power_noise, db_range=96) + 96

#%%
# Before simulating the attenuation of the acoustic signature depending on 
# the distance, we need to evaluate the distance at which the signal of the 
# spinetail was recordered.
# First, we estimate the sound level L of the spinetail song in the recording 
# by selected the sound between 4900-7500 Hz.
p0_sig_4900_7500 = sound.select_bandwidth(p0_sig,fs,fcut=[4900,7300],
                                          forder=10, ftype='bandpass')
L = spl.pressure2leq(p0_sig_4900_7500, fs) 
print ('Sound Level measured : %2.2fdB SPL' %L)
  
# Then, knowing the sound level L at the position of the ARU, we can estimate 
# the maximum distance between the ARU and the position of the spinetail. 
# This estimation takes into account the geometric, atmospheric and habitat
# attenuation (here, we use the default values to define atmospheric and 
# habitat attenuation). This distance will be the reference distance r0
f,r0 = spl.active_distance(L, 85, f=(7500+4900)/2) 
print ('Max distance between ARU and spinetail is estimated to be%2.0fm' %r0)  
  
#%%
# knowing the distance r0 at which the signal was recorded from the source,
# we can simulate the attenuation of the acoustic signature depending on
# different distance of propagation.
# Here, we will simulate the attenuation of the signal after propagating 10m, 
# 20m, 40m, 80m

# plot original spectrogram
import matplotlib.pyplot as plt
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, sharex=True, figsize=(15,3))
util.plot2d(Sxx_dB, title="original", ax=ax1, extent=ext, vmin=0, vmax=70, figsize=[3,3])

# Compute the attenuation of the recorded spinetail song at 10m.
p_att = spl.apply_attenuation(p0_sig, fs, r0=5, r = 10)
Sxx_power_att, tn, fn, ext = sound.spectrogram(p_att,fs)
Sxx_dB_att_10m = util.power2dB(Sxx_power_att,db_range=96) + 96 

# Compute the attenuation of the recorded spinetail song at 20m.
p_att = spl.apply_attenuation(p0_sig, fs, r0=5, r = 20)
Sxx_power_att, tn, fn, ext = sound.spectrogram(p_att,fs)
Sxx_dB_att_20m = util.power2dB(Sxx_power_att,db_range=96) + 96 

# Compute the attenuation of the recorded spinetail song at 40m.
p_att = spl.apply_attenuation(p0_sig, fs, r0=5, r = 40)
Sxx_power_att, tn, fn, ext = sound.spectrogram(p_att,fs)
Sxx_dB_att_40m = util.power2dB(Sxx_power_att,db_range=96) + 96 

# Compute the attenuation of the recorded spinetail song at 80m.
p_att = spl.apply_attenuation(p0_sig, fs, r0=5, r = 80)
Sxx_power_att, tn, fn, ext = sound.spectrogram(p_att,fs)
Sxx_dB_att_80m = util.power2dB(Sxx_power_att,db_range=96) + 96 

# Add noise to the signal.
# We add real noise recorded just after the song of the spinetail.
# We subtracted 3dB in order to take the mean between the noise and the signal

Sxx_dB_att_10m = util.add_dB(Sxx_dB_att_10m,Sxx_dB_noise) - 3 
Sxx_dB_att_20m = util.add_dB(Sxx_dB_att_20m,Sxx_dB_noise) - 3 
Sxx_dB_att_40m = util.add_dB(Sxx_dB_att_40m,Sxx_dB_noise) - 3 
Sxx_dB_att_80m = util.add_dB(Sxx_dB_att_80m,Sxx_dB_noise) - 3 
  
# Plot attenuated spectrogram.

util.plot2d(Sxx_dB_att_10m, title='10m', ax=ax2, extent=ext, vmin=0, vmax=70, figsize=[3,3])
util.plot2d(Sxx_dB_att_20m, title='20m', ax=ax3, extent=ext, vmin=0, vmax=70, figsize=[3,3])
util.plot2d(Sxx_dB_att_40m, title='40m', ax=ax4, extent=ext, vmin=0, vmax=70, figsize=[3,3])
util.plot2d(Sxx_dB_att_80m, title='80m', ax=ax5, extent=ext, vmin=0, vmax=70, figsize=[3,3])   