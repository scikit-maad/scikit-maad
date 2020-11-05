# -*- coding: utf-8 -*-
""" feature functions for scikit-maad
Methods to compute multiple descriptors to characterize sounds

"""

from .features_2d import (filter_multires,
                          filter_bank_2d_nodc,
                          opt_shape_presets,
                          shape_features,
                          shape_features_raw,
                          plot_shape,
                          centroid_features,
                          overlay_centroid,
                          rois_features,
                          compute_all_features)

from .features_1d import psd, rms

from .alpha_indices import (intoBins, # should be in sounds
                            entropy,
                            skewness,
                            kurtosis,
                            spectral_entropy,
                            acoustic_activity,
                            acoustic_events,
                            acousticComplexityIndex,
                            soundscapeIndex,
                            bioacousticsIndex,
                            acousticDiversityIndex,
                            acousticEvenessIndex,
                            roughness,
                            surfaceRoughness,
                            tfsd,
                            acousticGradientIndex,
                            raoQ)

__all__ = [
           # features_2d
           'filter_multires', 
           'filter_bank_2d_nodc',
           'opt_shape_presets',
           'shape_features',
           'shape_features_raw',
           'plot_shape',
           'centroid_features',
           'overlay_centroid',
           'rois_features',
           'compute_all_features',
           # features_1d
           'psd',
           'rms',
           # alpha_indices
           "intoBins",
           "entropy",
           "skewness",
           "kurtosis",
           "spectral_entropy",
           "acoustic_activity",
           "acoustic_events",
           "acousticComplexityIndex",
           "soundscapeIndex",
           'bioacousticsIndex',
           "acousticDiversityIndex",
           "acousticEvenessIndex",
           "roughness",
           "surfaceRoughness",
           "tfsd",
           "acousticGradientIndex",
           "raoQ"]