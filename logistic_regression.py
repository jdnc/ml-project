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
from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

import preprocess as pp
import experiment as ex

def main():
    parser = argparse.ArgumentParser(prog=sys.argv[0])
    parser.add_argument("-x",required=True, help="file containing brain features")
    parser.add_argument("-y",required=True, help="file containing target values")
    args = parser.parse_args()
    x = np.loadtxt(args.x)
    y = np.loadtxt(args.y)
    # Since y has string labels encode them to numerical values
    le = preprocessing.LabelEncoder()
    le.fit(y)
    # now encode y so that it has numerical classes rather than string
    y_enc = le.transform(y)
    # since study assumes uniform prior for each term, set fit_prior to false
    kf = cross_validation.KFold(len(y_enc), n_folds=10)
    conf_mat = np.zeros((len(le.classes_),len(le.classes_)))
    std_dev_fold = []
    for train, test in kf:
        logreg = linear_model.LogisticRegression(C=1e5)
        predicted = logreg.fit(x[train],y_enc[train]).predict(x[test])
        conf_mat += confusion_matrix(y_enc[test], predicted, labels=np.arange(22))
	std_dev_fold.append(np.std(predicted))
    print(conf_mat) # just for debugging
    print("Saving...") 
    print (std_dev_fold) # debugging
    np.save(os.path.join(args.s, "conf_mat.npy"), conf_mat)
    np.save(os.path.join(args.s, "std_dev.npy"), np.array(std_dev_fold))
    np.save(args.)

if __name__ == "__main__":
    main()
