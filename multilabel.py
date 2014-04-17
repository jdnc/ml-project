from __future__ import print_function

"""
Multi-label classifier for neurosynth.
Initially begins with the 25 terms from the paper
Uses Logistic regression for multi-label classification and l1 regularization.
"""

import os
import sys
import argparse

import numpy as np
import json
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

import preprocess as pp
import experiment as ex
from codesamples import utils

THRESH = 0.001
def main():
    feature_dict, col_names = pp.set_targets('data/features.txt', threshold=-1)
    # consider only the terms of interest
    with open('data/terms.json', 'rb') as f:
	terms = json.load(f)
    for key in list(feature_dict):
	feature_dict[key] = [x for x in terms if feature_dict[key][x] > THRESH]
	if not feature_dict[key]:
	    del(feature_dict[key])
    # filter coordinates based on voxels
    coord_dict = ex.filter_studies_active_voxels('data/docdict.txt', 'data/MNI152_T1_2mm_brain.nii.gz', 
						  threshold=500, radius=6)
    # find intersecting dicts
    coord_dict, feature_dict = ex.get_intersecting_dicts(coord_dict, feature_dict)
    # get the respective vectors
    X, y = pp.get_features_targets(coord_dict, feature_dict, is_voxels=True)
    # fit a label binarizer
    lb = preprocessing.LabelBinarizer()
    y_new = lb.fit_transform(y)
    # perform the 10 fold cross_validation
    score_per_class = []
    score_per_label = []
    clf =  OneVsRestClassifier(LogisticRegression(penalty='l1')) 
    kf = cross_validation.KFold(len(y_new), n_folds=10)
    for train, test in kf:
	 predicted  = clf.fit(x[train],y_new[train]).predict(x[test])
	 predict_prob = LogisticRegression(penalty='l1').fit(x[train],y_new[train]).predict_proba(x[test])
	 cls_scores = utils.score_results(y_new[test], predicted, predicted_prob)
	 label_scores = utils.label_scores(y_new[test], predicted, predicted_prob)
	 score_per_class.append(cls_scores)
	 score_per_label.append(label_scores)
     with open('class_scores.json', 'wb') as f:
	 json.dump(scores_per_class, f) 
     with open('label_scores.json', 'wb') as f:
	 json.dump(scores_per_label, f) 
     return 


if __name__ == '__main__':
    main()

   


