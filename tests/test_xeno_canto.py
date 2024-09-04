#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for xeno-canto functions

"""

# %%
import numpy as np
import os
import pandas as pd
from maad import util
import pytest

EXPECTED_FIELDS = [
            'id',
            'gen',
            'sp',
            'ssp',
            'group',
            'en',
            'rec',
            'cnt',
            'loc',
            'lat',
            'lng',
            'alt',
            'type',
            'sex',
            'stage',
            'method',
            'url',
            'file',
            'file-name',
            'sono',
            'osci',
            'lic',
            'q',
            'length',
            'time',
            'date',
            'uploaded',
            'also',
            'rmk',
            'bird-seen',
            'animal-seen',
            'playback-used',
            'temp',
            'regnr',
            'auto',
            'dvc',
            'mic',
            'smp'
 ]

# %%

# test if the fields in the metadata are still the same
def test_metadata_fields():

    output = util.xc_query(['gen:Picoides', 'ssp:tridactylus'])

    # list of fields
    found_fields = list(output)
    
    # Check if lists are different
    conflicted_fields = list(set(EXPECTED_FIELDS) - set(found_fields)) # type: ignore
    if conflicted_fields:
        assert 'The list of fields is not identical', \
        'The CONFLICTED fields are : {}'.format(list(set(EXPECTED_FIELDS) - set(found_fields))) # type: ignore

# %%

# test if the usual query are

@pytest.mark.parametrize(
    "searchTerms, expected",
    [
        (['gen:Picoides', 'ssp:tridactylus'],        'birds'),
        (['Agelaius','phoeniceus'],                  'birds'),
        (['Eurasian','Three-toed'],                  'birds'),
        (['Eurasian','Three-toed','type:drumming'],  'birds'),
        (['Agelaius','phoeniceus','type:song'],      'birds'),
        #(['Agelaius','phoeniceus','lat:">40.5"'],    'birds'), # don't work anymore 04/09/2024 => new version of xeno-canto API ?
        #(['Agelaius','phoeniceus','len:">1"'],       'birds')  # don't work anymore 04/09/2024 => new version of xeno-canto API ?
    ]
)

def test_xc_query(searchTerms, expected):

    output = util.xc_query(searchTerms)

    if output['group'][0] != expected :
        assert 'Query with search terms {} FAILED'.format(searchTerms)
    
# %%
