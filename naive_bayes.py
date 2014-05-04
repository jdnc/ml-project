from __future__ import print_function
#!/usr/bin/env python
"""
Python code to replicate the Naive Bayes classifier from the Large Scale Image
segmentation paper

Uses Naive Bayes for multi-class OvO classification given 24 labels
Uses 10-fold cross validation
Uses a uniform prior for all terms
"""
import argparse
import sys
import json
import os

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

import preprocess as pp
import experiment as ex

def get_X_y(coordinate_file, feature_file, filter=True):
    if filter:
        coordinate_dict = ex.filter_studies_active_voxels(coordinate_file, 'data/MNI152_T1_2mm_brain.nii.gz')
    
    	with open(coordinate_file) as f:
    		coordinate_dict = json.load(f)
    target_dict = ex.filter_studies_terms(feature_file, set_unique_label=True)
    coordinate_dict, target_dict = ex.get_intersecting_dicts(coordinate_dict,
                                                          target_dict)
    X, y = pp.get_features_targets(coordinate_dict, target_dict)
    return X, y


def main():
    parser = argparse.ArgumentParser(prog=sys.argv[0])
    parser.add_argument("-s",required=True, help="name of folder to save the confusion matrix numpy array")
    parser.add_argument("-c",required=True, help="name of file having the jsonized coordinate dict")
    parser.add_argument("-f",required=True, help="name of file with the raw features")
    args = parser.parse_args()
    x, y = get_X_y(args.c, args.f)
    # Since y has string labels encode them to numerical values
    le = preprocessing.LabelEncoder()
    le.fit(y)
    # now encode y so that it has numerical classes rather than string
    y_enc = le.transform(y)
    # since study assumes uniform prior for each term, set fit_prior to false
    clf = MultinomialNB()
    kf = cross_validation.KFold(len(y_enc), n_folds=10)
    conf_mat = np.zeros((len(le.classes_),len(le.classes_)))
    std_dev_fold = []
    for train, test in kf:
        predicted = OneVsRestClassifier(clf).fit(x[train],y_enc[train]).predict(x[test])
        conf_mat += confusion_matrix(y_enc[test], predicted, labels=np.arange(22))
	std_dev_fold.append(np.std(predicted))
    print(conf_mat) # just for debugging
    print("Saving...") 
    print (std_dev_fold) # debugging
    np.save(os.path.join(args.s, "conf_mat.npy"), conf_mat)
    np.save(os.path.join(args.s, "std_dev.npy"), np.array(std_dev_fold))
    

if __name__ == "__main__":
    main()
