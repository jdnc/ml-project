from __future__ import print_function

"""
Multi-label classifier for neurosynth.
Initially begins with the 25 terms from the paper
Uses Logistic regression for multi-label classification and l1 regularization.
"""


import numpy as np
import nibabel as nb
import json
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import WardAgglomeration
from sklearn.feature_extraction import image

import pickle
import preprocess as pp
import experiment as ex
import get_openfmri_data as openfmri
import utils

THRESH = 0.001
N_CLUSTERS = 5000

def classify(x, y):
    # fit a label binarizer
    lb = preprocessing.LabelBinarizer()
    y_new = lb.fit_transform(y)

    # specify connectivity for clustering
    mask = nb.load('data/MNI152_T1_2mm_brain.nii.gz').get_data().astype('bool')
    shape = mask.shape
    connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1], n_z=shape[2], mask=mask)

    ward = WardAgglomeration(n_clusters=N_CLUSTERS, connectivity=connectivity)
    #ward = WardAgglomeration(n_clusters=N_CLUSTERS)
    clf =  OneVsRestClassifier(LogisticRegression(penalty='l2'))
    kf = cross_validation.KFold(len(y_new), n_folds=10)
    score_per_class = []
    score_per_label = []
    cntr=1
    for train, test in kf:
        print(cntr)
	cntr+=1
        ward.fit(x[train])
        train_reduced = ward.transform(x[train])
        test_reduced = ward.transform(x[test])
        model = clf.fit(train_reduced, y_new[train])
        predicted  = model.predict(test_reduced)
        predict_prob = model.predict_proba(test_reduced)
        cls_scores = utils.score_results(y_new[test], predicted, predict_prob)
        label_scores = utils.label_scores(y_new[test], predicted, predict_prob)
        score_per_class.append(cls_scores)
        score_per_label.append(label_scores)
        #if(cntr>1):
        #  break
    #with open('log_clust_class_scores.json', 'wb') as f:
    #    json.dump(score_per_class, f)
    #pickle.dump(score_per_label,open('log_cl_label_scores.p','wb'))
    return (score_per_class,score_per_label)


def main():
    # consider only the terms of interest
    with open('data/terms.json', 'rb') as f:
       	terms = json.load(f)
    # extract X (image features) and y (label values)
    X = openfmri.get_X('data/zstat_run1.nii.gz')
    y = openfmri.get_Y('data/features.txt','data/data_key_run1.txt',terms)
    # perform clustering, cross validation and classify
    (score_per_class,score_per_label) = classify(X,y)
    with open('clustered_class_scores.json', 'wb') as f:
        json.dump(score_per_class, f)
    pickle.dump(score_per_label,open('clustered_label_scores.p','wb'))
    return


if __name__ == '__main__':
    main()


