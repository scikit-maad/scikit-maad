#!/usr/bin/env python
""" Multiresolution Analysis of Acoustic Diversity
    Cluster funtions 
"""
#
# Authors:  Juan Sebastian ULLOA <lisofomia@gmail.com>
#           Sylvain HAUPERT <sylvain.haupert@mnhn.fr>
#
# License: New BSD License

# Load required modules
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA


def do_PCA (features, headers_select=None, col_min=8, col_max=None, **kwargs):
    """
    
    if you want to know the keys in the table => features.keys()
    
    """
   
    n_components=kwargs.pop('n_components',2)
    s=kwargs.pop('s',40)
    
    #------------------------------------------
    # X : observations matrix (2D array)
    #------------------------------------------
    X = [] 
    # select the data corresponding to the header key
    if headers_select is not None:
        X = features[headers_select].values
    # or select the data corresponding to the columns
    else:
        if (col_min <0 ): 
            print('Warning: col_min has to be >0')
        else:
            if col_max is None: col_max=len(features.keys())
            X = features[list(features.columns[col_min:col_max])].values  
            
    #------------------------------------------
    # YlabelID : label for each observations (1D array)
    #------------------------------------------
    YlabelID = []
    # Create a vector Y with colors corresponding to the label
    unique_labelName = np.unique(np.array(features.labelName))
    for label in features.labelName:
        for ii, name in enumerate(unique_labelName):   
            if label in name :
                YlabelID.append(int(ii))
    
    # Calcul the PCA and display th results
    plt.figure()
    pca = PCA(n_components=n_components, **kwargs)
    Xp = pca.fit_transform(X)
    plt.scatter(Xp[:, 0], Xp[:, 1], c=YlabelID, s=s, **kwargs)
    
    return pca, Xp, YlabelID

