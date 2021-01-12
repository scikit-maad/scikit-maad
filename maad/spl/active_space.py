#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Collection of functions to convert audio and voltage data to Sound Pressure 
Level (SPL in Pascal) and Leq (Continuous Equivalent SPL).
"""   
#
# Authors:  Juan Sebastian ULLOA <lisofomia@gmail.com>
#           Sylvain HAUPERT <sylvain.haupert@mnhn.fr>
#
# License: New BSD License 


"""****************************************************************************
# -------------------       Load modules            ---------------------------
****************************************************************************"""

# Import external modules
import numpy as np 
from numpy import sum, log10, abs, mean, sqrt, exp
import pandas as pd

# min value
import sys
_MIN_ = sys.float_info.min

# import internal modules
from maad.spl import dBSPL2pressure, pressure2dBSPL

"""****************************************************************************
# -------------------       Public functions        ---------------------------
****************************************************************************"""
#******** geometrical attenuation
def geometrical_att_factor (r,r0) :
  """
  Get the attenuation in dB due to spreading loss (also known as geometrical 
  attenuation). Usually the source is considered to be ponctual which creates
  a spherical spreading loss.
  
  INPUTS :
  r : propagation distances in m [SCALAR or VECTOR]
  r0 : reference distance in m [SCALAR]
  
  OUTPUT
  return the geometrical attenuation of an acoustic pressure in dB

  """ 
  # make sure it's array
  r = np.asarray(r)  
  
  Ageo_factor = r0/r
  
  return Ageo_factor

#******** geometrical (or spherical) attenuation
def geometrical_att_dB (r,r0) :
  """
  Get the attenuation in dB due to spreading loss (also known as spherical or geometrical attenuation)
  
  INPUTS :
  r : propagation distances in m [SCALAR or VECTOR]
  r0 : reference distance in m [SCALAR]
  
  OUTPUT
  return the geometrical (or spherical) attenuation of an acoustic pressure in dB
  => subtract this value to the reference acoustic pressure in dB (or sound pressure level (SPL))
  """ 
  # make sure it's array
  r = np.asarray(r)    
  
  Ageo_dB = -20*log10(geometrical_att_factor(r,r0))
  
  return Ageo_dB


#********  atmospheric attenuation
def atmospheric_att_coef_dB (f, t=20, rh=60, pa=101325):
   
  """ 
  Get the atmospheric attenuation coefficient in dB/m  
  
  INPUTS :
  f: frequency in Hz [SCALAR]
  t: temperature in °C [SCALAR]
  rh: relative humidity in % [SCALAR]
  pa: atmospheric pressure in Pa [SCALAR]
  
  OUTPUT
  Aatm_coef_dB : in dB/m
  
  Partially from http://www.sengpielaudio.com/AirdampingFormula.htm
  """
  # make sure it's array
  f = np.asarray(f)  
  
  pr = 101.325e3 # reference ambient atmospheric pressure: 101.325 kPa
  To1 = 273.16 # triple-point isotherm temp: 273.16 K
  To = 293.15 #  reference temperature in K: 293.15 K (20°C)
  t = t+273.15 # celcius to farenheit
  
  psat = pr*10**(-6.8346 * (To1/t)**1.261 + 4.6151) #saturation vapor pressure equals
  h = rh * (psat / pa) # molar concentration of water vapor, as a percentage
  frO = (pa / pr) * (24 + 4.04e4 * h * ((0.02 + h) / (0.391 + h))) # oxygen relaxation frequency
  frN = (pa / pr) * sqrt(t / To) * (9 + 280 * h * exp(-4.170*((t/To)**(-1/3) -1))) # nitrogen relaxation frequency
  
  z = 0.1068 * exp (-3352/t) / (frN+f**2 /frN)
  y = (t/To)**(-5/2) * (0.01275 * exp(-2239.1/t) * 1/(frO+f**2/frO) + z)
  Aatm_coef_dB = 8.686 * f**2 * ((1.84e-11 * 1/(pa/pr) * sqrt(t/To)) + y)
  
  return Aatm_coef_dB 


#********  atmospheric attenuation
def atmospheric_att_coef (f, t=20, rh=60, pa=101325):
  """ 
  Get the atmospheric attenuation coefficient in dB/m  
  
  INPUTS :
  f: frequency in Hz [SCALAR]
  t: temperature in °C [SCALAR]
  rh: relative humidity in % [SCALAR]
  pa: atmospheric pressure in Pa [SCALAR]
  
  OUTPUT
  Aatm_coef: in dB/m
  
  Partially from http://www.sengpielaudio.com/AirdampingFormula.htm
  """  
  # make sure it's array
  f = np.asarray(f)  
    
  Aatm_coef = atmospheric_att_coef_dB (f, t, rh, pa)/(20*log10(exp(1)))
  
  return Aatm_coef 
    
def atmospheric_att_factor (f, r, r0, t=20, rh=60, pa=101325):

  """ 
  Get the atmospheric attenuation in dB
  
  INPUTS :
  f: frequency in kHz [SCALAR]
  t: temperature in °C [SCALAR]
  rh: relative humidity in % [SCALAR]
  pa: atmospheric pressure in Pa [SCALAR]
  
  OUTPUT :
  Aatm : 
  atmospheric attenuation in dB depending on the frequency, the temperature, the atmospheric pressure, the relative humidity and the distance [MATRIX] 
  => subtract this value to the reference acoustic pressure in dB (or sound pressure level (SPL)) for each frequency and distance
  """
  # make sure it's array
  r = np.asarray(r)
  f = np.asarray(f) 
  # test if r is a single value
  if r.ndim == 0 : Nr= 1
  else: Nr = len(r)
  # test if f is a single value
  if f.ndim == 0 : Nf= 1
  else: Nf = len(f)

  Aatm_coef = atmospheric_att_coef(f, t, rh, pa)
  Aatm_factor = exp(-Aatm_coef.reshape(Nf,1) @ (r.reshape(1,Nr)-r0))
  
  # reshape the array when dimensions are 1
  if Aatm_factor.shape[0] == 1:
      Aatm_factor = Aatm_factor[0][:]
  if Aatm_factor.shape[1] == 1:
      Aatm_factor = Aatm_factor[:,0]
  
  return Aatm_factor

def atmospheric_att_dB (f, r, r0, t=20, rh=60, pa=101325):

  """ 
  Get the atmospheric attenuation in dB
  
  INPUTS :
  f: frequency in kHz [SCALAR]
  t: temperature in °C [SCALAR]
  rh: relative humidity in % [SCALAR]
  pa: atmospheric pressure in Pa [SCALAR]
  
  OUTPUT :
  Aatm.dB : 
  atmospheric attenuation in dB depending on the frequency, the temperature, the atmospheric pressure, the relative humidity and the distance [MATRIX] 
  => subtract this value to the reference acoustic pressure in dB (or sound pressure level (SPL)) for each frequency and distance
  
  row => frequency
  column => distances
  
  """
  # make sure it's array
  r = np.asarray(r)
  f = np.asarray(f) 
  # test if r is a single value
  if r.ndim == 0 : Nr= 1
  else: Nr = len(r)
  # test if f is a single value
  if f.ndim == 0 : Nf= 1
  else: Nf = len(f)
  
  Aatm_coef_dB = atmospheric_att_coef_dB(f, t, rh, pa)
  Aatm_dB = Aatm_coef_dB.reshape(Nf,1) @ (r.reshape(1,Nr)-r0)
  
  # reshape the array when dimensions are 1
  if Aatm_dB.shape[0] == 1:
      Aatm_dB = Aatm_dB[0][:]
  if Aatm_dB.shape[1] == 1:
      Aatm_dB = Aatm_dB[:,0]

  return Aatm_dB

#********  Habitat attenuation
def habitat_att_factor (f, r, r0, a0=0.002):
  """ 
  Get the habitat attenuation factor
  
  INPUTS :
  f: frequency in Hz [SCALAR]
  r : propagation distances in m [SCALAR or VECTOR]
  r0 : reference distance in m [SCALAR]
  a0 : attenuation coefficient of the habitat in Neper/Hz/m [SCALAR]
  
  OUTPUT :
  Ahab : habitat attenuation depending on the frequency and the distance [MATRIX] 
  => Multiply this value with the effective reference acoustic pressure p0 measured at r0 to estimate the pressure after attenuation
  """
  # make sure it's array
  r = np.asarray(r)
  f = np.asarray(f) 
  # test if r is a single value
  if r.ndim == 0 : Nr= 1
  else: Nr = len(r)
  # test if f is a single value
  if f.ndim == 0 : Nf= 1
  else: Nf = len(f)
  
  Ahab_coef = a0*f/1000
  Ahab_factor = exp(-Ahab_coef.reshape(Nf,1) @ (r.reshape(1,Nr)-r0))
  
  # reshape the array when dimensions are 1
  if Ahab_factor.shape[0] == 1:
      Ahab_factor = Ahab_factor[0][:]
  if Ahab_factor.shape[1] == 1:
      Ahab_factor = Ahab_factor[:,0]
  
  return Ahab_factor

def habitat_att_dB(f, r, r0, a0=0.002) :
  """ 
  Get the habitat attenuation in dB
  
  INPUTS :
  f: frequency in kHz [SCALAR]
  r : propagation distances in m [SCALAR or VECTOR]
  r0 : reference distance in m [SCALAR]
  a0 : attenuation coefficient of the habitat in Neper/kHz/m [SCALAR]
  
  OUTPUT :
  Ahab_dB : habitat attenuation in dB depending on the frequency and the distance [MATRIX] 
  => subtract this value to the reference acoustic pressure in dB (or sound pressure level (SPL)) for each frequency and distance
  """
  Ahab_dB = -20*log10(habitat_att_factor(f,r,r0,a0))
    
  return (Ahab_dB)

def habitat_att_coeff_dB (f,a0=0.002):
  """ 
  get the habitat attenuation coefficient in dB/m for the frequency f knowning the habitat attenuation parameter a0 (in Neper/kHz/m)
  
  INPUTS:
  f: frequency in kHz [SCALAR]
  a0 : attenuation coefficient of the habitat in Neper/kHz/m [SCALAR]
  
  OUTPUT:
  coeff.Ahab.dB  : return the habitat attenuation of an acoustic pressure in dB/m
  => subtract this value to the reference acoustic pressure in dB (or sound pressure level (SPL))
  """
  # make sure it's array
  f = np.asarray(f)  
  
  Ahab_coeff_dB = a0*f * 20*log10(exp(1))
  return Ahab_coeff_dB

def habitat_att_coeff (f,a0=0.002):
  """ 
  get the habitat attenuation factor for the frequency f knowning the habitat attenuation parameter a0 (in Neper/kHz/m)
  
  INPUTS:
  f: frequency in kHz [SCALAR]
  a0 : attenuation coefficient of the habitat in Neper/kHz/m [SCALAR]
  
  OUTPUT:
  coeff.Ahab : return the habitat attenuation factor of an acoustic pressure [Pa]
  => Multiply this value with the effective reference acoustic pressure p0 measured at r0 to estimate the pressure after attenuation
  """
  # make sure it's array
  f = np.asarray(f)  
  
  Ahab_coeff = a0*f 
  return Ahab_coeff

#************ full attenuation
def full_attenuation_factor (f, r, r0, t=20, rh=60, pa=101325, a0=0.002) :
  """ 
  get full attenuation factor taking into account the geometric, atmospheric and habitat attenuation contributions
  
  INPUTS:
  f: frequency in kHz [SCALAR]
  r : propagation distances in m [SCALAR or VECTOR]
  r0 : reference distance in m [SCALAR]
  t: temperature in °C [SCALAR]
  rh: relative humidity in % [SCALAR]
  pa: atmospheric pressure in Pa [SCALAR]
  a0 : attenuation coefficient of the habitat in Neper/kHz/m [SCALAR]
  
  OUTPUT:
  Atotal : return the total attenuation factor of an acoustic pressure [Pa]
  => Multiply this value with the effective reference acoustic pressure p0 measured at r0 to estimate the pressure after attenuation
  """
  # make sure it's array
  f = np.asarray(f)  
  r = np.asarray(r)
  
  Ageo_factor = geometrical_att_factor(r,r0)
  Aatm_factor = atmospheric_att_factor(f,r,r0,t,rh,pa)
  Ahab_factor = habitat_att_factor(f,r,r0,a0)
  
  # make sure it's array
  Ageo_factor = np.asarray(Ageo_factor)  
  Aatm_factor = np.asarray(Aatm_factor)
  Ahab_factor = np.asarray(Ahab_factor)

  Afull_factor = Ageo_factor[np.newaxis, ...] * Aatm_factor * Ahab_factor
  
  return Afull_factor

def full_attenuation_dB (f, r, r0, t=20, rh=60, pa=101325, a0=0.002):
  """ 
  get full attenuation factor taking into account the geometric, atmospheric and habitat attenuation contributions
  
  INPUTS:
  f: frequency in kHz [SCALAR]
  r : propagation distances in m [SCALAR or VECTOR]
  r0 : reference distance in m [SCALAR]
  t: temperature in °C [SCALAR]
  rh: relative humidity in % [SCALAR]
  pa: atmospheric pressure in Pa [SCALAR]
  a0 : attenuation coefficient of the habitat in Neper/kHz/m [SCALAR]
  
  OUTPUT:
  Atotal.dB : return the total attenuation in dB of an acoustic pressure [Pa]
  => subtract this value with the effective reference acoustic pressure p0 measured at r0 to estimate the pressure after attenuation
  """
  # make sure it's array
  f = np.asarray(f)  
  r = np.asarray(r)
  
  Ageo_dB = geometrical_att_dB(r,r0)
  Aatm_dB = atmospheric_att_dB(f,r,r0,t,rh,pa)
  Ahab_dB = habitat_att_dB(f,r,r0,a0)
  
  # make sure it's array
  Ageo_dB = np.asarray(Ageo_dB)  
  Aatm_dB = np.asarray(Aatm_dB)
  Ahab_dB = np.asarray(Ahab_dB)
    
  Afull_dB = Ageo_dB[np.newaxis,...] + Aatm_dB  + Ahab_dB
  
  return Afull_dB


#************** Repartition of energy along the frequencies
def dBSPL_per_bin (L, f) :
  """
  Function to spread the sound pressure level (Energy in dB) along a frequency vector (bins)   
  
  INPUTS:
  L : Sound Pressure Level in dB
  f: frequency in kHz [SCALAR or VECTOR]
  OUTPUT :
  Two vectors : 1st column is the frequency vector and the 2nd column is the sound pressure level corresponding to the frequency number of bins 
  """
  # test if f is a scalar
  if not hasattr(f, "__len__") : 
      L_per_bin = L 
  # if f is a vector
  else:
      # force to be ndarray
      f = np.asarray(f)   
      # init
      L_per_bin = np.ones(len(f)) * L
      nb_bin= len(f)
      # dB SPL for the frequency bandwidth
      L_per_bin = L_per_bin - 10*log10(nb_bin) 

  return f, L_per_bin

#************** Active distance
def active_distance (L_bkg, L0, f, r0= 1, delta_r=1, t=20, rh=60, pa=101325, 
                     a0=0.002):
  """ 
  get full attenuation factor taking into account the geometric, atmospheric and habitat attenuation contributions
  
  INPUTS:
  L_bkg : sound pressure level of the background in dB SPL [SCALAR, VECTOR]
  L0 : sound pressure level of the sound that is propagated
  f : frequency vector in kHz [VECTOR]
  r0 : distance at which L0 was measured (generally @1m)
  delta_r : distance resolution in m [SCALAR]
  t: temperature in °C [SCALAR]
  rh: relative humidity in % [SCALAR]
  pa: atmospheric pressure in Pa [SCALAR]
  a0 : attenuation coefficient of the habitat in Neper/kHz/m [SCALAR]
  
  OUTPUT:
  distance_max : maximum distance of propagation before the sound pressure level is below the background [SCALAR or VECTOR
  """
    
  # test if f is a scalar
  if not hasattr(f, "__len__") : f = np.array([f])
  if not hasattr(L_bkg, "__len__") : L_bkg = np.array([L_bkg])
  if not hasattr(L0, "__len__") : L0 = np.array([L0])

  # test if f, L_bkg and L0 have the same length
  if not (len(f)==len(L_bkg) & len(f)==len(L_bkg) & len(L0)==len(L_bkg)) : 
      raise TypeError ('L_bkg, L0 and f must have the same length')    

  # set the distance vector
  r = np.arange(1,10000,delta_r) 

  # number of frequencies
  Nf = len(f)     
  
  # set the distance max vector to store the result   
  distance_max = np.zeros(Nf)
  
  # get the initial pressure
  p0 = dBSPL2pressure(L0)
  
  # get the background pressure 
  p_bkg = dBSPL2pressure(L_bkg)
  
  # test for each frequency when the simulated pressure at distance r is below the background pressure
  for ii in np.arange(Nf) :
    # Get the pressure from the full attenuation model knowing the pressure p0 at r0
    p_simu = p0[ii] * full_attenuation_factor(f[ii], r, r0, t, rh, pa, a0)
    # distance max
    if sum((p_simu - p_bkg[ii])>0) >1 :
      distance_max[ii] = r[np.argmin((p_simu - p_bkg[ii])[(p_simu - p_bkg[ii])>0])] 
    else :
      distance_max[ii] = 0
      
  # test if f and distance_max are scalars
  if len(f) == 1 : f = f[0]
  if len(distance_max) == 1 : distance_max = distance_max[0]
  
  # return the frequency vector associated with the distance max
  return f,distance_max

#************************** Get the pressure at r0 from simulation
def pressure_at_r0 (f, r, p, r0=1, t=20, rh=60, pa=101325, a0=0.002) :
  """ 
    get the pressure at distance r0 from experimental values of pressure p measured at distance r.
    This function takes into account the geometric, atmospheric and habitat attenuations

  INPUTS:
    f : frequency vector in kHz [VECTOR]
    r : distance vector in m [VECTOR]
    p : pressure vector in Pa [VECTOR]
    r0 : distance where the pressure will be evaluated [SCALER]
    t: temperature in °C [SCALAR]
    rh: relative humidity in % [SCALAR]
    pa: atmospheric pressure in Pa [SCALAR]
    a0 : attenuation coefficient of the habitat in Neper/kHz/m [SCALAR]

  OUTPUT:
    p0 : estimated pressure at distance r0 [SCALAR]
  """
  # test if f is a scalar
  if not hasattr(f, "__len__") : f = np.array([f])
  if not hasattr(r, "__len__") : r = np.array([r])
  if not hasattr(p, "__len__") : p = np.array([p])

  Ageo_factor = geometrical_att_factor(r,r0)
  Aatm_factor = atmospheric_att_factor(f,r,r0,t,rh,pa)
  Ahab_factor = habitat_att_factor(f,r,r0,a0)
  
  if Ageo_factor.size == Aatm_factor.size:
      p0 = p * Ageo_factor**(-1) * Aatm_factor**(-1) * Ahab_factor**(-1)
  else:    
      p0 = p * (Ageo_factor**(-1))[np.newaxis,...] * Aatm_factor**(-1) * Ahab_factor**(-1)
  
  # test if p0 is a scalar
  if Ageo_factor.size == 1 : p0 = p0[0][0]
  
  return p0  

#************************** Get the pressure at r0 from simulation
def L0_at_r0 (f, r, p, r0=1, t=20, rh=60, pa=101325, a0=0.002, pRef=10e-6) :
  """ 
    get the pressure at distance r0 from experimental values of pressure p measured at distance r.
    This function takes into account the geometric, atmospheric and habitat attenuations

  INPUTS:
    f : frequency vector in kHz [VECTOR]
    r : distance vector in m [VECTOR]
    p : pressure vector in Pa [VECTOR]
    r0 : distance where the pressure will be evaluated [SCALER]
    t: temperature in °C [SCALAR]
    rh: relative humidity in % [SCALAR]
    pa: atmospheric pressure in Pa [SCALAR]
    a0 : attenuation coefficient of the habitat in Neper/kHz/m [SCALAR]

  OUTPUT:
    p0 : estimated pressure at distance r0 [SCALAR]
  """
  # test if f is a scalar
  if not hasattr(f, "__len__") : f = np.array([f])
  if not hasattr(r, "__len__") : r = np.array([r])
  if not hasattr(p, "__len__") : p = np.array([p])
  
  # Get the initial pressure (Pa)
  p0 = pressure_at_r0(f, r, p, r0, t, rh, pa, a0)
  
  # Transform the pressure into dB SPL
  L0 = pressure2dBSPL(p0, pRef)
  
  return L0  

## #************* apply attenuation
#propa.apply.att <- function(p0, fs, r, r0= 1, t=20, rh=60, pa=101325, a0=0.002)
#{
#  " 
#  Apply attenuation of a temporal signal p0 after propagation between the reference distance r0 and the final distance r 
#  taken into account the geometric, atmospheric and habitat attenuation contributions
#  
#  INPUTS:
#  p0 : temporal signal (time domain) [VECTOR]
#  fs: sampling frequency Hz [SCALAR]
#  r : propagation distances in m [SCALAR or VECTOR]
#  r0 : reference distance in m [SCALAR]
#  t: temperature in °C [SCALAR]
#  rh: relative humidity in % [SCALAR]
#  pa: atmospheric pressure in Pa [SCALAR]
#  a0 : attenuation coefficient of the habitat in Neper/kHz/m [SCALAR]
#  
#  OUTPUT:
#  p : temporal signal (time domain) after attenuation [VECTOR]
#  "
#  
#  # Fourier domain
#  P0 = fft(p0)/length(p0)
#  f = seq(0,length(P0)-1) / length(P0) * fs /2
#  # apply attenuation
#  P = P0  * propa.get.Atotal(f/1000, r, r0, t, rh, pa, a0)
#  # Go back to the time domain
#  p = fft(P, inverse=TRUE)
#  # keep the real part
#  p= Re(p)
#  
#  return (p)
#}


