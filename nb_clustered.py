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
from sklearn.cluster import WardAgglomeration
from sklearn.feature_extraction import image

import preprocess as pp
import experiment as ex

N_CLUSTERS = 100000

def classify(x, y):
    # get the connectivity for Ward
    mask = np.load('data/2mm_brain_mask.npy')
    shape = mask.shape
    connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1], n_z=shape[2], mask=mask)
    ward = WardAgglomeration(n_cluster=N_CLUSTERS, connectivity=connectivity)
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y_new = le.transform(y)
    clf = MultinomialNB()
    kf = cross_validation.KFold(len(y_new), n_folds=10)
    accuracy = []
    for train, test in kf:
        ward.fit(x[train])
        train_reduced = ward.transform(x[train])
        test_reduced = ward.transform(x[test])
        predicted = clf.fit(train_reduced, y_new[train]).predict(test_reduced)
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
    with open('data/terms.json', 'rb') as f:
        terms = json.load(f)
    feature_dict = ex.filter_studies_terms('data/features.txt', terms=terms, set_unique_label=True)
    for i in range(len(terms)):
        for j in range(i+1, len(terms)):
            sub_dict = {}
            for key in feature_dict:
                if feature_dict[key] in [terms[i], terms[j]]:
                    sub_dict[key] = feature_dict[key]
            study, feat  = ex.get_intersecting_dicts(study_dict, sub_dict)
            x, y = pp.get_features_targets(study, feat, mask='data/MNI152_T1_2mm_brain.nii.gz')
            cf = classify(x, y)
            save_name = terms[i] + '_vs_' + terms[j] + '.npy'
            with open(save_name, 'wb') as f:
                np.save(f, cf)


if __name__ == "__main__":
    main()


