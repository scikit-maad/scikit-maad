#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Collection of functions to estimation sound attenuation due to spreading loss,
atmospheric absorption and habitat attenuation (diffraction, reflexion, 
diffusion... due to folliage, ground, trunks).These functions are required to
estimate active distance (i.e. detection range) of audio recorder (such as SM4)
 as well as initial sound pressure level (dB SPL).
"""   
#
# Authors:  Juan Sebastian ULLOA <lisofomia@gmail.com>
#           Sylvain HAUPERT <sylvain.haupert@mnhn.fr>        
#
# License: New BSD License

#%%
#***************************************************************************
# -------------------       Load modules         ---------------------------
#***************************************************************************
# Import external modules
import numpy as np 
from numpy import sum, log10, sqrt, exp
import pandas as pd

# min value
import sys
_MIN_ = sys.float_info.min

# import internal modules
from maad.spl import dBSPL2pressure, pressure2dBSPL

#%%
# =============================================================================
# Private functions
# =============================================================================

def _geometric_att_factor (r,r0) :
  """
  Get the attenuation factor due to spreading loss (also known as geometric
  attenuation). Usually the source is considered to be ponctual which creates
  a spherical spreading loss.
  
  Parameters
  ----------
  r : scalar or array-like
      propagation distances in m [SCALAR or VECTOR]
  r0 : scalar
      reference distance in m [SCALAR]
  
  Returns
  -------
  Ageo_factor : scalar or array-like
      Geometric attenuation factor 
      => Multiply this factor with the effective reference acoustic pressure p0 
      measured at r0 to estimate the pressure after attenuation

  """ 
  # make sure it's array
  r = np.asarray(r)  
  
  Ageo_factor = r0/r
  
  return Ageo_factor

#%%
def _geometric_att_dB (r,r0) :
  """
  Get the attenuation in dB due to spreading loss (also known as geometric 
  attenuation).Usually the source is considered to be ponctual  which creates 
  a spherical spreading loss.
  
  Parameters
  ----------
  r : scalar or array-like
      propagation distances in m 
  r0 : scalar
      reference distance in m
  
  Returns
  -------
  Ageo_dB : scalar or array-like
      Geometric attenuation 
      => Subtract this value from the reference acoustic pressure L0 in dB 
      (or sound pressure level (SPL)) measured at r0 to estimate the
      acoustic pressure L in dB after attenuation
      
  """ 
  # make sure it's array
  r = np.asarray(r)    
  
  Ageo_dB = -20*log10(_geometric_att_factor(r,r0))
  
  return Ageo_dB

#%%
def _atmospheric_att_coef_dB (f, t=20, rh=60, pa=101325):
   
  """ 
  Get the atmospheric attenuation coefficient in dB/m  
  
  Parameters
  ----------
  f: scalar or array-like
      frequency in Hz
  t: scalar, optional, default is 20
      temperature in °C
  rh: scalar, optional, default is 60
      relative humidity in % 
  pa: scalar, optional, default is 101325
      atmospheric pressure in Pa 
  
  Returns
  -------
  Aatm_coef_dB : scalar or array-like
      atmospheric attenuation coefficient in dB/m  
  
  References
  ----------
  Partially from http://www.sengpielaudio.com/AirdampingFormula.htm
  """
  # make sure it's array
  f = np.asarray(f)  
  # test if t, rh and pa are scalars
  if hasattr(t, "__len__") : raise TypeError ('t must be a scalar ')    
  elif hasattr(rh, "__len__") : raise TypeError ('rh must be a scalar ')    
  elif  hasattr(pa, "__len__") : raise TypeError('pa must be a scalar ')       
  
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

#%%
def _atmospheric_att_coef (f, t=20, rh=60, pa=101325):
  """ 
  Get the atmospheric attenuation coefficient in Neper/m  
  
  Parameters
  ----------
  f: scalar or array-like
      frequency in Hz
  t: scalar, optional, default is 20
      temperature in °C
  rh: scalar, optional, default is 60
      relative humidity in % 
  pa: scalar, optional, default is 101325
      atmospheric pressure in Pa 
  
  Returns
  -------
  Aatm_coef: scalar or array-like
      atmospheric attenuation coefficient in Neper/m  
  
  References
  ----------
  Partially from http://www.sengpielaudio.com/AirdampingFormula.htm
  """  
  # make sure it's array
  f = np.asarray(f)  
  # test if t, rh and pa are scalars
  if hasattr(t, "__len__") : raise TypeError ('t must be a scalar ')    
  elif hasattr(rh, "__len__") : raise TypeError ('rh must be a scalar ')    
  elif  hasattr(pa, "__len__") : raise TypeError('pa must be a scalar ')  
    
  Aatm_coef = _atmospheric_att_coef_dB (f, t, rh, pa)/(20*log10(exp(1)))
  
  return Aatm_coef 
    
def _atmospheric_att_factor (f, r, r0, t=20, rh=60, pa=101325):

  """ 
  Get the atmospheric attenuation factor 
  
  Parameters
  ----------
  f: scalar or array-like
      frequency in Hz
  r : scalar or array-like
      propagation distances in m 
  r0 : scalar
      reference distance in m
  t: scalar, optional, default is 20
      temperature in °C
  rh: scalar, optional, default is 60
      relative humidity in % 
  pa: scalar, optional, default is 101325
      atmospheric pressure in Pa 
  
  Returns
  -------
  Aatm_factor : scalar or array-like (vector (1D) or matrix (2D))
      atmospheric attenuation factor 
      => Multiply this factor with the effective reference acoustic pressure p0 
      measured at r0 to estimate the pressure after attenuation
      rows : frequencies
      columns :distances
  """
  # make sure it's array
  r = np.asarray(r)
  f = np.asarray(f) 
  
  # test if r is a single value
  if r.ndim == 0 : Nr= 1
  else: Nr = len(r.flatten())
  
  # test if f is a single value
  if f.ndim == 0 : Nf= 1
  else: Nf = len(f.flatten())
  
  # test if t, rh and pa are scalars
  if hasattr(t, "__len__") : raise TypeError ('t must be a scalar ')    
  elif hasattr(rh, "__len__") : raise TypeError ('rh must be a scalar ')    
  elif  hasattr(pa, "__len__") : raise TypeError('pa must be a scalar ')   

  Aatm_coef = _atmospheric_att_coef(f, t, rh, pa)
  Aatm_factor = exp(-Aatm_coef.reshape(Nf,1) @ (r.reshape(1,Nr)-r0))
  
  # reshape the array when dimensions are 1
  if Aatm_factor.shape[0] == 1:
      Aatm_factor = Aatm_factor[0][:]
  elif Aatm_factor.shape[1] == 1:
      Aatm_factor = Aatm_factor[:,0]
  
  return Aatm_factor

def _atmospheric_att_dB (f, r, r0, t=20, rh=60, pa=101325):

  """ 
  Parameters
  ----------
  f: scalar or array-like
      frequency in Hz
  r : scalar or array-like
      propagation distances in m 
  r0 : scalar
      reference distance in m
  t: scalar, optional, default is 20
      temperature in °C
  rh: scalar, optional, default is 60
      relative humidity in % 
  pa: scalar, optional, default is 101325
      atmospheric pressure in Pa 
  
  Returns
  -------
  Aatm_dB : scalar or array-like (vector (1D) or matrix (2D))
      atmospheric attenuation in dB depending on the frequency, the temperature,
      the atmospheric pressure, the relative humidity and the distance 
      => subtract Aatm_dB from the reference acoustic pressure L0 in dB 
      (or sound pressure level (SPL)) measuread at distance r0 for each 
      frequency and distance
      rows : frequencies
      columns :distances
  """
  # make sure it's array
  r = np.asarray(r)
  f = np.asarray(f) 
  
  # test if r is a single value
  if r.ndim == 0 : Nr= 1
  else: Nr = len(r.flatten())
  
  # test if f is a single value
  if f.ndim == 0 : Nf= 1
  else: Nf = len(f.flatten())
  
  # test if r0, t, rh and pa are scalars
  if hasattr(r0, "__len__") : raise TypeError ('r0 must be a scalar ')   
  elif hasattr(t, "__len__") : raise TypeError ('t must be a scalar ')    
  elif hasattr(rh, "__len__") : raise TypeError ('rh must be a scalar ')    
  elif  hasattr(pa, "__len__") : raise TypeError('pa must be a scalar ') 
  
  Aatm_coef_dB = _atmospheric_att_coef_dB(f, t, rh, pa)
  Aatm_dB = Aatm_coef_dB.reshape(Nf,1) @ (r.reshape(1,Nr)-r0)
  
  # reshape the array when dimensions are 1
  if Aatm_dB.shape[0] == 1:
      Aatm_dB = Aatm_dB[0][:]
  elif Aatm_dB.shape[1] == 1:
      Aatm_dB = Aatm_dB[:,0]

  return Aatm_dB

#%%
def _habitat_att_factor (f, r, r0, a0=0.02):
  """ 
  Get the habitat attenuation factor
  
  Parameters
  ----------
  f: scalar or array-like
      frequency in Hz
  r : scalar or array-like
      propagation distances in m 
  r0 : scalar
      reference distance in m
  a0 : scalar
      attenuation coefficient of the habitat in dB/kHz/m 
  
  Returns
  -------
  Ahab : scalar or array-like (vector (1D) or matrix (2D))
      habitat attenuation depending on the frequency and the distance [MATRIX] 
      => Multiply this value with the effective reference acoustic pressure p0 
      measured at r0 to estimate the pressure after attenuation
  """
  # make sure it's array
  r = np.asarray(r)
  f = np.asarray(f) 

  # test if r is a single value
  if r.ndim == 0 : Nr= 1
  else: Nr = len(r.flatten())
  
  # test if f is a single value
  if f.ndim == 0 : Nf= 1
  else: Nf = len(f.flatten())
  
  # test if r0 and a0 are scalars
  if hasattr(r0, "__len__") : raise TypeError ('r0 must be a scalar ')   
  elif hasattr(a0, "__len__") : raise TypeError ('a0 must be a scalar ')    

  Ahab_coef = _habitat_att_coeff(f, a0)
  Ahab_factor = exp(-Ahab_coef.reshape(Nf,1) @ (r.reshape(1,Nr)-r0))
  
  # reshape the array when dimensions are 1
  if Ahab_factor.shape[0] == 1:
      Ahab_factor = Ahab_factor[0][:]
  elif Ahab_factor.shape[1] == 1:
      Ahab_factor = Ahab_factor[:,0]
  
  return Ahab_factor

#%%
def _habitat_att_dB(f, r, r0, a0=0.02) :
  """ 
  Get the habitat attenuation in dB
  
  Parameters
  ----------
  f: scalar or array-like
      frequency in Hz
  r : scalar or array-like
      propagation distances in m 
  r0 : scalar
      reference distance in m
  a0 : scalar, optional, default is 0.02
      attenuation coefficient of the habitat in dB/kHz/m 
    
  Returns
  -------
  Ahab_dB : scalar or array-like (vector (1D) or matrix (2D))
      habitat attenuation in dB depending on the frequency, the distance and 
      the habitat attenuation coefficient
      => subtract Aatm_dB from the reference acoustic pressure L0 in dB 
      (or sound pressure level (SPL)) measuread at distance r0 for each 
      frequency and distance
      rows : frequencies
      columns :distances
  """  
  #Ahab_dB = -20*log10(_habitat_att_factor(f,r,r0,a0))
  
  # make sure it's array
  r = np.asarray(r)
  f = np.asarray(f) 

  # test if r is a single value
  if r.ndim == 0 : Nr= 1
  else: Nr = len(r.flatten())
  
  # test if f is a single value
  if f.ndim == 0 : Nf= 1
  else: Nf = len(f.flatten())
  
  # test if r0 and a0 are scalars
  if hasattr(r0, "__len__") : raise TypeError ('r0 must be a scalar ')   
  elif hasattr(a0, "__len__") : raise TypeError ('a0 must be a scalar ')   
  
  Ahab_coef_dB = _habitat_att_coeff_dB(f,a0)
  Ahab_dB = Ahab_coef_dB.reshape(Nf,1) @ (r.reshape(1,Nr)-r0)
  
  # reshape the array when dimensions are 1
  if Ahab_dB.shape[0] == 1:
      Ahab_dB = Ahab_dB[0][:]
  elif Ahab_dB.shape[1] == 1:
      Ahab_dB = Ahab_dB[:,0]
    
  return (Ahab_dB)

#%%
def _habitat_att_coeff_dB (f, a0=0.02):
  """ 
  get the habitat attenuation coefficient in dB/m for the frequency f knowing 
  the habitat attenuation parameter a0 (in dB/kHz/m)
  
  Parameters
  ----------
  f: scalar or array-like
      frequency in Hz
  a0 : scalar, optional, default is 0.02
      attenuation coefficient of the habitat in dB/kHz/m 
      
  Returns
  -------
  coeff.Ahab.dB  : scalar or array-like (vector (1D)) 
      habitat attenuation coefficient in dB/m for the frequency f
      
  """
  # make sure it's array
  f = np.asarray(f)  
  
  Ahab_coeff_dB = a0*f/1000 
  
  return Ahab_coeff_dB

#%%
def _habitat_att_coeff (f, a0=0.02):
  """ 
  get the habitat attenuation factor for the frequency f knowing the habitat 
  attenuation parameter a0 (in dB/kHz/m)
  
  Parameters
  ----------
  f: scalar or array-like
      frequency in Hz
  a0 : scalar, optional, default is 0.02
      attenuation coefficient of the habitat in dB/kHz/m 
      
  Returns
  -------
  coeff.Ahab : scalar or array-like (vector (1D)) 
      habitat attenuation factor for the frequency f
  """
  # make sure it's array
  f = np.asarray(f)  
  
  Ahab_coeff = _habitat_att_coeff_dB(f,a0) /20*log10(exp(1))
  return Ahab_coeff

#%%
def _attenuation_factor (f, r, r0, t=20, rh=60, pa=101325, a0=0.02) :
  """ 
  get attenuation factor taking into account the geometric, atmospheric 
  and habitat attenuation contributions
  
  Parameters
  ----------
  f: scalar or array-like
      frequency in Hz
  r : scalar or array-like
      propagation distances in m 
  r0 : scalar
      reference distance in m
  t: scalar, optional, default is 20
      temperature in °C
  rh: scalar, optional, default is 60
      relative humidity in % 
  pa: scalar, optional, default is 101325
      atmospheric pressure in Pa 
  a0 : scalar, optional, default is 0.02
      attenuation coefficient of the habitat in dB/kHz/m 
  
  Returns
  -------
  A_factor : scalar or array-like (vector (1D) or matrix (2D))
      attenuation depending on the frequency and the distance [MATRIX] 
      => Multiply this value with the effective reference acoustic pressure p0 
      measured at r0 to estimate the pressure after attenuation taking into 
      account 3 types of attenuation
      rows : frequencies
      columns :distances
  """
  # make sure it's array
  f = np.asarray(f)  
  r = np.asarray(r)
  
  Ageo_factor = _geometric_att_factor(r,r0)
  Aatm_factor = _atmospheric_att_factor(f,r,r0,t,rh,pa)
  Ahab_factor = _habitat_att_factor(f,r,r0,a0)
  
  # make sure it's array
  Ageo_factor = np.asarray(Ageo_factor)  
  Aatm_factor = np.asarray(Aatm_factor)
  Ahab_factor = np.asarray(Ahab_factor)

  A_factor = Ageo_factor[np.newaxis, ...] * Aatm_factor * Ahab_factor
  
  return A_factor

#%%
# =============================================================================
# Public functions
# =============================================================================
def attenuation_dB (f, r, r0, t=20, rh=60, pa=101325, a0=0.02):
  """ 
  Compute the attenuation in decibels taking into account the geometric, atmospheric 
  and habitat attenuation contributions.
  
  Parameters
  ----------
  f: scalar or array-like
      frequency in Hz
  r : scalar or array-like
      propagation distances in m 
  r0 : scalar
      reference distance in m
  t: scalar, optional, default is 20
      temperature in °C
  rh: scalar, optional, default is 60
      relative humidity in % 
  pa: scalar, optional, default is 101325
      atmospheric pressure in Pa 
  a0 : scalar, optional, default is 0.02
      attenuation coefficient of the habitat in dB/kHz/m 
  
  Returns
  -------
  Asum_dB : scalar or array-like (vector (1D) or matrix (2D))
      Global attenuation in decibel taking into account the geometric, atmospheric 
      and habitat attenuation.
      => subtract Asum_dB from the reference acoustic pressure L0 in dB 
      (or sound pressure level (SPL)) measuread at distance r0 for each 
      frequency and distance to estimate the pressure after attenuation taking 
      into account 3 types of attenuation
      rows : frequencies
      columns :distances
      
  Examples
  --------
  >>> import numpy as np
  >>> fn = np.arange(500,10500,500)
  >>> r = np.arange(10,110)
  >>> r0 = 1
  >>> Asum_dB, df_att = maad.spl.attenuation_dB(fn,r,r0)
  
  For 2kHz @50m, the attenuation is mainly drive by geometric attenuation
  
  >>> df_att[(df_att.f==2000) & (df_att.r==50)]
              f   r  Ageo_dB   Aatm_dB   Ahab_dB    Asum_dB
      310  2000  50  33.9794  0.454776      1.96  36.394176
  
  For 10Hz @100m, the attenuation due to the atmosphere and the habitat is no
  negligible
      
  >>> df_att[(df_att.f==10000) & (df_att.r==100)]
                f   r  Ageo_dB      Aatm_dB   Ahab_dB    Asum_dB
      1990  10000  100    40.0    13.335002      19.8  73.135002
  """
  # make sure it's array
  f = np.asarray(f)  
  r = np.asarray(r)
  
  Ageo_dB = _geometric_att_dB(r,r0)
  Aatm_dB = _atmospheric_att_dB(f,r,r0,t,rh,pa)
  Ahab_dB = _habitat_att_dB(f,r,r0,a0)
  
  # make sure it's array
  Ageo_dB = np.asarray(Ageo_dB)  
  Aatm_dB = np.asarray(Aatm_dB)
  Ahab_dB = np.asarray(Ahab_dB)
    
  Asum_dB = Ageo_dB[np.newaxis,...] + Aatm_dB  + Ahab_dB
  
  #### create a dataframe with all values
  # test if f is a single value
  if f.ndim == 0 : Nf= 1
  else: Nf = len(f)
  # test if r is a single value
  if r.ndim == 0 : Nr= 1
  else: Nr = len(r)
  
  # for Ageo, repeat vector along frequency axis
  Ageo_dB = np.tile(Ageo_dB,(Nf,1))
  # for f, repeat vector along distance axis
  f = np.tile(f,(Nr,1)).T
  # for r, repeat vector along frequency axis  
  r = np.tile(r,(Nf,1))
  
  # prepare data (transform matrices into long vectors)
  data = {'f':f.flatten(), 
          'r':r.flatten(), 
          'Ageo_dB':Ageo_dB.flatten(), 
          'Aatm_dB':Aatm_dB.flatten(), 
          'Ahab_dB':Ahab_dB.flatten(), 
          'Asum_dB':Asum_dB.flatten()}
  df_att = pd.DataFrame(data, columns = ['f', 
                                         'r', 
                                         'Ageo_dB', 
                                         'Aatm_dB', 
                                         'Ahab_dB',
                                         'Asum_dB'])
  
  return Asum_dB, df_att

#%%
def dBSPL_per_bin (L, f) :
  """
  Function to spread the sound pressure level (Energy in dB) along a frequency 
  vector (bins).
  
  Parameters
  ----------
  L : scalar
      Sound Pressure Level in dB
  f: array-like (vector (1D))
      frequency vector in Hz
  
  Returns
  -------
  L_per_bin :  array-like (vector (1D))
      sound pressure level in dB with values corresponding to the frequency's 
      number of bins 
  
  Examples
  --------
  Spread 80dB SPL from 2kHz to 5kHz with 1kHz step
  
  >>> maad.spl.dBSPL_per_bin(80,[2000,3000,4000,5000])
  array([73.97940009, 73.97940009, 73.97940009, 73.97940009])
  
  Spread 80dB SPL from 2kHz to 10kHz with 1kHz step
  
  >>> maad.spl.dBSPL_per_bin(80,[2000,3000,4000,5000,6000,7000,8000,9000,10000])
  array([70.45757491, 70.45757491, 70.45757491, 70.45757491, 70.45757491,
       70.45757491, 70.45757491, 70.45757491, 70.45757491])
  
  Spread 80dB SPL from 2kHz to 5kHz with 0.5kHz step
  
  >>> maad.spl.dBSPL_per_bin(80,[2000,2500,3000,3500,4000,4500,5000])
  array([71.5490196, 71.5490196, 71.5490196, 71.5490196, 71.5490196,
       71.5490196, 71.5490196])  
  
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

  return L_per_bin

#%%
def detection_distance (L_bkg, L0, f, r0= 1, delta_r=1, t=20, rh=60, pa=101325, 
                     a0=0.02, rmax=10000):
  """ 
  Compute the detection distance also known as detection range or detection radius
  or active space.
  
  Parameters
  ----------
  L_bkg : scalar or array-like
      sound pressure level of the background/ambient "noise" in dB SPL
  L0 : scalar or array-like
      Initial sound pressure level measured at distance r0
  f: scalar or array-like
      frequency in Hz
  r0 : scalar
      distance at which L0 was measured (generally @1m)
  delta_r : scalar
      distance resolution in m 
  t: scalar, optional, default is 20
      temperature in °C
  rh: scalar, optional, default is 60
      relative humidity in % 
  pa: scalar, optional, default is 101325
      atmospheric pressure in Pa 
  a0 : scalar, optional, default is 0.02
      attenuation coefficient of the habitat in dB/kHz/m 
  rmax : scalar, optional, default is 10000
      define the maximal distance that is taken into account for active distance
      calculation. Default value is 10000m (i.e. 10km) which is OK for most of
      the purpose. If you increase the value, the calculation will be longer      
        
  Returns
  -------
  distance_max : scalar or array-like
      maximum distance of propagation before the sound pressure level is below 
      the ambient sound level
  
  Notes
  -----
  The maximum detection range is limited by the background noise or ambient sound 
  also called noise. The signal is attenuated during the propagation until
  the level reaches the ambient sound level, meaning that the signal could not 
  be disantangle from the background sound anymore.
  
  Examples
  --------
  Estimation of the active distance for an initial sound of 90dB SPL @2kHz 
  with a background noise of 50dB SPL
    
  >>> f, r = maad.spl.active_distance (L_bkg=50, L0=90, f=2000) 
  >>> print('Max active distance is %2.1fm' %r)
  Max active distance is 70.0m
  
  Estimation of the active distance for an initial sound of 90dB SPL @10kHz 
  with a background noise of 50dB SPL
    
  >>> f, r = maad.spl.active_distance (L_bkg=50, L0=90, f=10000) 
  >>> print('Max active distance is %2.1fm' %r)
  Max active distance is 32.0m
  
  Estimation of the active distance for an initial sound of 90dB SPL @2kHz 
  with a background noise of 30dB SPL
    
  >>> f, r = maad.spl.active_distance (L_bkg=30, L0=90, f=2000) 
  >>> print('Max active distance is %2.1fm' %r)
  Max active distance is 263.0m
  
  Estimation of the active distance for an initial sound of 90dB SPL @2kHz 
  with a background noise of 30dB SPL, in tropical rain forest atmosphere 
  (t=30, rh=99)
    
  >>> f, r = maad.spl.active_distance(L_bkg=30,L0=90,f=2000,t=30,rh=99) 
  >>> print('Max active distance is %2.1fm' %r)
  Max active distance is 247.0m
  
  Estimation of the active distance for an initial sound of 90dB SPL @2kHz 
  with a background noise of 30dB SPL, in cold dry forest atmosphere 
  (t=5, rh=40)
    
  >>> f, r = maad.spl.active_distance(L_bkg=30,L0=90,f=2000,t=5,rh=40) 
  >>> print('Max active distance is %2.1fm' %r)
  Max active distance is 225.0m
  
  """
    
  # test if f is a scalar
  if not hasattr(f, "__len__") : f = np.array([f])
  if not hasattr(L_bkg, "__len__") : L_bkg = np.array([L_bkg])
  if not hasattr(L0, "__len__") : L0 = np.array([L0])

  # test if f, L_bkg and L0 have the same length
  if not (len(f)==len(L_bkg) & len(f)==len(L_bkg) & len(L0)==len(L_bkg)) : 
      raise TypeError ('L_bkg, L0 and f must have the same length')  
      
  # test if r0 and a0 are scalars
  if hasattr(r0, "__len__") : raise TypeError ('r0 must be a scalar ') 
  if hasattr(rmax, "__len__") : raise TypeError ('rmax must be a scalar ')   

  # set the distance vector
  r = np.arange(1,rmax,delta_r) 

  # number of frequencies
  Nf = len(f)     
  
  # set the distance max vector to store the result   
  distance_max = np.zeros(Nf)

  # test for each frequency if the signal sound level at distance r is below
  # the background sound level    
  for ii in np.arange(Nf) :  
      # estimate the full attenuation of the signal
      Asum_dB, _ = attenuation_dB(f[ii], r, r0, t, rh, pa, a0)
      
      # Get the sound level after subtracting the attenuation
      L = L0[ii] - Asum_dB  
      
      # distance max that corresponds to a 
      if sum((L - L_bkg[ii])>0) >1 :
          distance_max[ii] = r[np.argmin((L - L_bkg[ii])[(L - L_bkg[ii])>0])] 
      else :
          distance_max[ii] = 0
      
  # test if f and distance_max are scalars
  if len(f) == 1 : f = f[0]
  if len(distance_max) == 1 : distance_max = distance_max[0]
  
  # return the frequency vector associated with the distance max
  return f,distance_max

#%%
def pressure_at_r0 (f, r, p, r0=1, t=20, rh=60, pa=101325, a0=0.02) :
  """ 
    Estimate the pressure p0 at distance r0 from pressure p measured at 
    distance r. This function takes into account the geometric, atmospheric 
    and habitat attenuations.

  Parameters
  ----------
  f: scalar or array-like
      frequency in Hz
  r: scalar or array-like
      distance in m at which p in Pa was measured 
  p : scalar or array-like
      pressure in Pa
  r0 : scalar
      distance at which p0 needs to be estimated (generally @1m)
  t: scalar, optional, default is 20
      temperature in °C
  rh: scalar, optional, default is 60
      relative humidity in % 
  pa: scalar, optional, default is 101325
      atmospheric pressure in Pa 
  a0 : scalar, optional, default is 0.02
      attenuation coefficient of the habitat in dB/kHz/m 
        
  Returns
  -------
  p0 : scalar or array-like
      estimated pressure at distance r0.
      This is a scalar if p is a scalar or a vector if p and r are vectors or 
      a matrix if p and r are matrix 

  Examples
  --------
  Estimation of L0, knowing L=50dB SPL @60m @100Hz
  
  >>> p = maad.spl.dBSPL2pressure(50)
  >>> p0 = maad.spl.pressure_at_r0(100,60,p)
  >>> L0 = maad.spl.pressure2dBSPL(p0)
  >>> L0
  85.68036510289447
  
  Estimation of L0, knowing L=50dB SPL @60m @10000Hz
  
  >>> p0 = maad.spl.pressure_at_r0(10000,100,p)
  >>> L0 = maad.spl.pressure2dBSPL(p0)
  >>> L0
  120.53306313162624
  
  Estimation of L0, knowing L=50dB SPL @10m @100Hz
  
  >>> p = maad.spl.dBSPL2pressure(50)
  >>> p0 = maad.spl.pressure_at_r0(100,10,p)
  >>> L0 = maad.spl.pressure2dBSPL(p0)
  >>> L0
  70.01789933655922
      
  """
  # if vectors, test if r and L have same length 
  if (hasattr(r, "__len__") and hasattr(p, "__len__")) :
      if not (len(r) == len(p)) :  raise TypeError ('if vectors, r and p must have the same length') 
  
  # if scalar set f, r L as ndarray 
  if not hasattr(f, "__len__") : f = np.array([f])
  if not hasattr(r, "__len__") : r = np.array([r])
  if not hasattr(p, "__len__") : p = np.array([p])
  
  # force to be an ndarray
  f = np.asarray(f)
  r = np.asarray(r)
  p = np.asarray(p)
    
  # f, r and L must be scalars or vectors
  if f.ndim>1 : raise TypeError ('f must be a scalar or a vector') 
  if r.ndim>1 : raise TypeError ('r must be a scalar or a vector')  
  if p.ndim>1 : raise TypeError ('p must be a scalar or a vector')  
  
  Ageo_factor = _geometric_att_factor(r,r0)
  Aatm_factor = _atmospheric_att_factor(f,r,r0,t,rh,pa)
  Ahab_factor = _habitat_att_factor(f,r,r0,a0)
  
  if Ageo_factor.size == Aatm_factor.size:
      p0 = p * Ageo_factor**(-1) * Aatm_factor**(-1) * Ahab_factor**(-1)
  else:    
      p0 = p * (Ageo_factor**(-1))[np.newaxis,...] * Aatm_factor**(-1) * Ahab_factor**(-1)
  
  if hasattr(p0, "__len__") :
      # reshape the array when dimensions are 1
      if p0.ndim ==1:
          if p0.shape[0] == 1:
              p0 = p0[0]  
  return p0  

#%%
def dBSPL_at_r0 (f, r, L, r0=1, t=20, rh=60, pa=101325, a0=0.02, pRef=10e-6) :
  """ 
    Estimate the sound pressure level L0 (dB SPL) at distance r0 from sound pressure
    level L measured at distance r.
    
    This function takes into account the geometric, atmospheric and habitat attenuations.

  Parameters
  ----------
  f: scalar or array-like
      frequency in Hz
  r: scalar or array-like
      distance in m at which p in Pa was measured 
      if r is a vector, r must have the same length as L
  L : scalar or array-like
      Sound pressure level L in dB SPL
      if L is a vector, L must have the same length as r
  r0 : scalar
      distance at which p0 needs to be estimated (generally @1m)
  t: scalar, optional, default is 20
      temperature in °C
  rh: scalar, optional, default is 60
      relative humidity in % 
  pa: scalar, optional, default is 101325
      atmospheric pressure in Pa 
  a0 : scalar, optional, default is 0.02
      attenuation coefficient of the habitat in dB/kHz/m 
        
  Returns
  -------
  L0 : scalar or array-like
      estimated sound presssure level L0 at distance r0 [SCALAR]
  """ 
  
  # Transform the pressure into dB SPL
  p = dBSPL2pressure(L)  
  
  # Get the initial pressure (Pa)
  p0 = pressure_at_r0(f, r, p, r0, t, rh, pa, a0)
  
  # Transform the pressure into dB SPL
  L0 = pressure2dBSPL(p0, pRef)
  
  return L0  

#%%
def apply_attenuation (p0, fs, r, r0= 1, t=20, rh=60, pa=101325, a0=0.02):
  """ 
  Apply attenuation of a temporal signal p0 after propagation between the 
  reference distance r0 and the final distance r taken into account the 
  geometric, atmospheric and habitat attenuation contributions.
  
  Parameters
  ----------
  p0 : array-like (vector 1d)
      temporal signal p0 in Pa (time domain)
  fs: scalar 
      sampling frequency Hz
  r: scalar
      distance of propagation (in m) of the signal p0 
  r0 : scalar
      distance at which the temporal signal p0 was measured
  t: scalar, optional, default is 20
      temperature in °C
  rh: scalar, optional, default is 60
      relative humidity in % 
  pa: scalar, optional, default is 101325
      atmospheric pressure in Pa 
  a0 : scalar, optional, default is 0.02
      attenuation coefficient of the habitat in dB/kHz/m 
  
  Returns
  -------
  p :  array-like (vector 1d)
      temporal signal (time domain) after attenuation
  
  Examples
  --------
  Prepare the spinetail sound (Sound level @1m = 85dB SPL).
  
  >>> w, fs = maad.sound.load('../data/spinetail.wav') 
  >>> p0 = maad.spl.wav2pressure(wave=w, gain=42)
  >>> p0_sig = p0[int(5.68*fs):int(7.48*fs)] 
  >>> p0_noise = p0[int(8.32*fs):int(10.12*fs)] 
  >>> Sxx_power, tn, fn, ext = maad.sound.spectrogram(p0_sig ,fs)
  >>> Sxx_power_noise, tn, fn, ext = maad.sound.spectrogram(p0_noise ,fs)
  >>> Sxx_dB = maad.util.power2dB(Sxx_power, db_range=96) + 96
  >>> Sxx_dB_noise = maad.util.power2dB(Sxx_power_noise, db_range=96) + 96
  
  Get the sound level of the spinetail song (sound between 4900-7500 Hz).
  
  >>> p0_sig_4900_7500 = maad.sound.select_bandwidth(p0_sig,fs,fcut=[4900,7300],forder=10, ftype='bandpass')
  >>> L = maad.spl.pressure2leq(p0_sig_4900_7500, fs) 
  >>> print ('Sound Level measured : %2.2fdB SPL' %L)
  
  Estimate maximum distance from the source.
  
  >>> r = maad.spl.active_distance(L, 85, f=(7500+4900)/2) 
  
  plot original spectrogram
  
  >>> import matplotlib.pyplot as plt
  >>> fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, sharex=True, figsize=(15,3))
  >>> maad.util.plot2d(Sxx_dB, ax=ax1, extent=ext, vmin=0, vmax=70, figsize=[3,3])

  Compute the audio attenuation at 10m.
  
  >>> p_att = maad.spl.apply_attenuation(p0_sig, fs, r0=5, r =10)
  >>> Sxx_power_att, tn, fn, ext = maad.sound.spectrogram(p_att,fs)
  >>> Sxx_dB_att_10m = maad.util.power2dB(Sxx_power_att,db_range=96) + 96 
  
  >>> p_att = maad.spl.apply_attenuation(p0_sig, fs, r0=5, r =20)
  >>> Sxx_power_att, tn, fn, ext = maad.sound.spectrogram(p_att,fs)
  >>> Sxx_dB_att_20m = maad.util.power2dB(Sxx_power_att,db_range=96) + 96 
  
  >>> p_att = maad.spl.apply_attenuation(p0_sig, fs, r0=5, r =40)
  >>> Sxx_power_att, tn, fn, ext = maad.sound.spectrogram(p_att,fs)
  >>> Sxx_dB_att_40m = maad.util.power2dB(Sxx_power_att,db_range=96) + 96 
  
  >>> p_att = maad.spl.apply_attenuation(p0_sig, fs, r0=5, r =80)
  >>> Sxx_power_att, tn, fn, ext = maad.sound.spectrogram(p_att,fs)
  >>> Sxx_dB_att_80m = maad.util.power2dB(Sxx_power_att,db_range=96) + 96 
  
  Add noise to the signal.
  
  >>> Sxx_dB_att_10m = maad.util.add_dB(Sxx_dB_att_10m,Sxx_dB_noise) - 3 
  >>> Sxx_dB_att_20m = maad.util.add_dB(Sxx_dB_att_20m,Sxx_dB_noise) - 3 
  >>> Sxx_dB_att_40m = maad.util.add_dB(Sxx_dB_att_40m,Sxx_dB_noise) - 3 
  >>> Sxx_dB_att_80m = maad.util.add_dB(Sxx_dB_att_80m,Sxx_dB_noise) - 3 
    
  Plot attenuated spectrogram.
  
  >>> maad.util.plot2d(Sxx_dB_att_10m, ax=ax2, extent=ext, vmin=0, vmax=70, figsize=[3,3])
  >>> maad.util.plot2d(Sxx_dB_att_20m, ax=ax3, extent=ext, vmin=0, vmax=70, figsize=[3,3])
  >>> maad.util.plot2d(Sxx_dB_att_40m, ax=ax4, extent=ext, vmin=0, vmax=70, figsize=[3,3])
  >>> maad.util.plot2d(Sxx_dB_att_80m, ax=ax5, extent=ext, vmin=0, vmax=70, figsize=[3,3])     
  
  """
  
  # Fourier domain
  p0_f = np.fft.fft(p0)
  f = np.arange(len(p0_f)) / len(p0_f) * fs /2
  # apply attenuation
  p_f = p0_f * _attenuation_factor(f, r, r0, t, rh, pa, a0)
  # Go back to the time domain
  p = np.fft.ifft(p_f)
  # keep the real part
  p = np.real(p)
  
  return (p)



