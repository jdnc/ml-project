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
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

import preprocess as pp
import experiment as ex



def classify(x, y):
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y_new = le.transform(y)
    tuned_parameters = tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                                           'C': [1, 10, 100, 1000]},
                                           {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    svr = SVC()
    clf = GridSearchCV(svr, tuned_parameters)
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


