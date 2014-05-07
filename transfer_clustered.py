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
    coord_dict = ex.filter_studies_active_voxels('data/docdict.txt', 'data/goodvoxmask.nii.gz',
                                                threshold=500, radius=6)
    # ensure that the keys are ints
    for key in list(coord_dict):
        if not isinstance(key, int):
            coord_dict[int(key)] = coord_dict[key]
            del(coord_dict[key])
    # find intersecting dicts
    coord_dict, feature_dict = ex.get_intersecting_dicts(coord_dict, feature_dict)
    # try for a sample
    sub_coord_dict = {k: coord_dict[k] for k in coord_dict.keys()[0:2500]}
    sub_coord_dict, sub_feature_dict = ex.get_intersecting_dicts(sub_coord_dict, feature_dict)
    X, y = pp.get_features_targets(sub_coord_dict, sub_feature_dict, labels=terms, mask='data/goodvoxmask.nii.gz')
    # get the respective vectors
    #X, y = pp.get_features_targets(coord_dict, feature_dict, labels=terms, mask='data/MNI152_T1_2mm_brain.nii.gz')
    # perform clustering, cross validation and classify
    # fit a label binarizer
    lb = preprocessing.LabelBinarizer()
    y_new = lb.fit_transform(y)

    # specify connectivity for clustering
    mask = nb.load('data/goodvoxmask.nii.gz').get_data().astype('bool')
    shape = mask.shape
    connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1], n_z=shape[2], mask=mask)

    ward = WardAgglomeration(n_clusters=N_CLUSTERS, connectivity=connectivity)
    ward.fit(X)
    # load test data from openfmri
    # extract X (image features) and y (label values)
    test_X = openfmri.get_X('zstat_run1.npy')
    test_y = openfmri.get_Y('cognitive_concepts/cognitive_concepts.txt','data_key_run1.txt','data/new_terms.json')
    test_y_new = lb.transform(test_y)

    clf =  OneVsRestClassifier(LogisticRegression(penalty='l2'))
    score_per_class = []
    score_per_label = []
    train_reduced = ward.transform(X)
    test_reduced = ward.transform(test_X)
    model = clf.fit(train_reduced, y_new)
    predicted  = model.predict(test_reduced)
    predict_prob = model.predict_proba(test_reduced)
    cls_scores = utils.score_results(test_y_new, predicted, predict_prob)
    label_scores = utils.label_scores(test_y_new, predicted, predict_prob)
    score_per_class.append(cls_scores)
    score_per_label.append(label_scores)
    with open('transfer_clustered_class_scores.json', 'wb') as f:
        json.dump(score_per_class, f)
    pickle.dump(score_per_label,open('transfer_clustered_label_scores.p','wb'))
    return


if __name__ == '__main__':
    main()


