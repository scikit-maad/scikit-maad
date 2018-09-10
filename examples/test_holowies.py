# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 12:19:46 2018

@author: haupert
"""

import numpy as np
import holoviews as hv
hv.extension('matplotlib')

frequencies = [0.5, 0.75, 1.0, 1.25]

def sine_curve(phase, freq):
    xvals = [0.1* i for i in range(100)]
    return hv.Curve((xvals, [np.sin(phase+freq*x) for x in xvals]))

curve_dict = {f:sine_curve(0,f) for f in frequencies}

hmap = hv.HoloMap(curve_dict, kdims='frequency')
hmap