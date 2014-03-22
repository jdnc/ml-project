"""
Python code to replicate the Naive Bayes classifier from the Large Scale Image
segmentation paper

Uses Naive Bayes for multi-class OvO classification given 24 labels
Uses 10-fold cross validation
Uses a uniform prior for all terms
"""

from __future__ import print_function

import os

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsOneClassifier
from sklearn import cross_validation


import preprocess as pp
import experiment as ex

def get_X_y():
    coordinate_dict = ex.filter_studies_active_voxels()
    target_dict = ex.filter_studies_terms(set_unique_label=True)
    coordinate_dict, target_dict = ex.get_intersecting_dicts(coordinate_dict,
                                                          target_dict)
    X, y = pp.get_features_targets(coordinate_dict, target_dict)
    return X, y


def main():
    pass

if __name__ == "__main__":
    main()
