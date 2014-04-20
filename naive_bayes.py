from __future__ import print_function
#!/usr/bin/env python
"""
Python code to replicate the Naive Bayes classifier from the Large Scale Image
segmentation paper

Uses Naive Bayes for multi-class OvO classification given 24 labels
Uses 10-fold cross validation
Uses a uniform prior for all terms
"""

import json
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

import preprocess as pp
import experiment as ex



def classify(x, y):
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y_new = le.transform(y)
    clf = MultinomialNB()
    kf = cross_validation.KFold(len(y_new), n_folds=10)
    accuracy = []
    for train, test in kf:
        predicted = clf.fit(x[train],y_new[train]).predict(x[test])
        conf_mat =  confusion_matrix(y_new[test], predicted, labels=[0,1])
        accuracy.append(conf_mat)
    return accuracy

def main():
    with open('data/docdict.txt') as f:
    	coordinates = json.load(f)
    study_dict = ex.filter_studies_active_voxels(coordinates, 'data/MNI152_T1_2mm_brain.nii.gz', radius=6, threshold=500)
    # ensure that study dict has int as keys
    for key in list(study_dict):
        study_dict[int(key)] = study_dict[key]
        del(study_dict[key])
    feature_dict = ex.filter_studies_terms('data/features.txt', terms=['emotion', 'reward', 'pain'], set_unique_label=True)
    e_r = {}
    e_p = {}
    p_r = {}
    for key in feature_dict:
        if feature_dict[key] in ['emotion', 'reward']:
            e_r[key] = feature_dict[key]
    for key in feature_dict:
        if feature_dict[key] in ['emotion', 'pain']:
            e_p[key] = feature_dict[key]
    for key in feature_dict:
        if feature_dict[key] in ['pain', 'reward']:
            p_r[key] = feature_dict[key]
    studyER, ER = ex.get_intersecting_dicts(study_dict, e_r)
    studyEP, EP = ex.get_intersecting_dicts(study_dict, e_p)
    studyRP, RP = ex.get_intersecting_dicts(study_dict, p_r)
    xER, yER = pp.get_features_targets(studyER, ER, mask='data/MNI152_T1_2mm_brain.nii.gz')
    xEP, yEP = pp.get_features_targets(studyEP, EP, mask='data/MNI152_T1_2mm_brain.nii.gz')
    xRP, yRP = pp.get_features_targets(studyRP, RP, mask='data/MNI152_T1_2mm_brain.nii.gz')
    cfEP = classify(xEP, yEP)
    cfER = classify(xER, yER)
    cfRP = classify(xRP, yRP)
    with open('e_vs_p.json', f):
	    json.dump(cfEP, f)
    with open('e_vs_r.json', f):
	    json.dump(cfER, f)
    with open('p_vs_r.json', f):
	    json.dump(cfRP, f)

if __name__ == "__main__":
    main()
