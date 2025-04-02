# -*- coding: utf-8 -*-
""" 
Acoustic features
=================

The module ``features`` is an ensemble of functions to characterize audio signals using temporal and spectral features, and ecoacoustic indices.


Spectro-temporal features
-------------------------
.. autosummary::
    :toctree: generated/
    
    shape_features
    shape_features_raw
    opt_shape_presets
    filter_multires
    filter_bank_2d_nodc
    centroid_features
    all_shape_features

Alpha acoustic indices
----------------------
.. autosummary::
    :toctree: generated/
        
    temporal_median
    temporal_entropy
    acoustic_richness_index
    temporal_activity
    temporal_events
    acoustic_complexity_index
    frequency_entropy
    number_of_peaks
    spectral_entropy
    spectral_activity
    spectral_events
    spectral_cover
    soundscape_index
    bioacoustics_index
    acoustic_diversity_index
    acoustic_eveness_index
    roughness
    temporal_leq
    spectral_leq
    surface_roughness
    tfsd
    more_entropy
    acoustic_gradient_index
    frequency_raoq
    region_of_interest_index_deprecated
    region_of_interest_index
    all_temporal_alpha_indices
    all_spectral_alpha_indices

Temporal features
-----------------
.. autosummary::
    :toctree: generated/
    
    temporal_moments
    zero_crossing_rate
    temporal_duration
    temporal_quantile
    all_temporal_features
    
Spectral features
-----------------
.. autosummary::
    :toctree: generated/
        
    spectral_moments
    peak_frequency
    spectral_quantile
    spectral_bandwidth
    all_spectral_features

Composite acoustic features
---------------------------
.. autosummary::
    :toctree: generated/

    graphical_soundscape
    plot_graph    

"""

from .shape import (filter_multires,
                    filter_bank_2d_nodc,
                    opt_shape_presets,
                    shape_features,
                    shape_features_raw,
                    centroid_features,
                    all_shape_features)

from .spectral import (spectral_moments,
                    peak_frequency,
                    spectral_quantile,
                    spectral_bandwidth,
                    all_spectral_features)

from .temporal import (temporal_moments,
                    zero_crossing_rate,
                    temporal_duration,
                    temporal_quantile,
                    all_temporal_features)

from .alpha_indices import (temporal_median,
                            temporal_entropy,
                            acoustic_richness_index,
                            temporal_activity,
                            temporal_events,
                            acoustic_complexity_index,
                            frequency_entropy,
                            number_of_peaks,
                            spectral_entropy,
                            spectral_activity,
                            spectral_events,
                            spectral_cover,
                            soundscape_index,
                            bioacoustics_index,
                            acoustic_diversity_index,
                            acoustic_eveness_index,
                            roughness,
                            temporal_leq,
                            spectral_leq,
                            surface_roughness,
                            tfsd,
                            more_entropy,
                            acoustic_gradient_index,
                            frequency_raoq,
                            region_of_interest_index_deprecated,
                            region_of_interest_index,
                            all_temporal_alpha_indices,
                            all_spectral_alpha_indices)

from .composite_soundscape_descriptors import (graphical_soundscape,
                                            plot_graph)

__all__ = [
        # shape
        'filter_multires', 
        'filter_bank_2d_nodc',
        'opt_shape_presets',
        'shape_features',
        'shape_features_raw',
        'centroid_features',
        'all_shape_features',
        
        # spectral
        'spectral_moments',
        'peak_frequency',
        'spectral_quantile',
        'temporal_quantile',
        'spectral_bandwidth',
        'all_spectral_features',
        
        # temporal
        'temporal_moments',
        'zero_crossing_rate',
        'temporal_duration',
        'all_temporal_features',
        
        # alpha_indices
        'temporal_moments',
        'temporal_median',
        'temporal_entropy',
        'acoustic_richness_index',
        "temporal_activity",
        "temporal_events",
        "acoustic_complexity_index",
        "frequency_entropy",
        "number_of_peaks",
        "spectral_entropy",
        "spectral_moments",
        "spectral_activity",
        "spectral_events",
        'spectral_cover',
        "soundscape_index",
        'bioacoustics_index',
        "acoustic_diversity_index",
        "acoustic_eveness_index",
        "roughness",
        'temporal_leq',
        'spectral_leq',
        "surface_roughness",
        "tfsd",
        "more_entropy",
        "acoustic_gradient_index",
        "frequency_raoq",
        "region_of_interest_index_deprecated",
        "region_of_interest_index",
        'all_temporal_alpha_indices',
        'all_spectral_alpha_indices',
        
        # composite features
        'graphical_soundscape',
        'plot_graph',
        ]