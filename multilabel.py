from __future__ import print_function

import os
import sys
import argparse

import numpy as np
import json

import preprocess as pp
import experiment as ex

def main():
    feature_dict, col_names = pp.set_targets(feature_file, threshold=-1)
    # consider only the terms of interest
    with open('data/terms.json', 'rb') as f:
	terms = json.load(f)
    for key in list(feature_dict):
	feature_dict[key] = feature_dict[key].loc[terms]
   # filter coordinates based on voxels
   coord_dict = ex.filter_studies_active_voxels('data/docdict.txt', 'data/MNI152_T1_2mm_brain.nii.gz', threshold=500, radius=6)
   # find intersecting dicts
   coord_dict, feature_dict = ex.get_intersecting_dicts(coord_dict, feature_dict)
   # get the respective vectors
   X, y = pp.get_features_targets(coord_dict, feature_dict, is_voxels=True)
 


