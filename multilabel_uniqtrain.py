from __future__ import print_function

"""
Multi-label classifier for neurosynth.
Initially begins with the 25 terms from the paper
Uses Logistic regression for multi-label classification and l1 regularization.
Train on studies with unique label but test on multi.
"""


import numpy as np
import random
import json
import pickle
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

import preprocess as pp
import experiment as ex
import utils

THRESH = 0.001

def predict_multilabel(X_test, y_new_test, clf_model, labels):
  """Use probabilities to do multilabel classification"""
  y_predicted = np.zeros((len(y_new_test),len(labels)), dtype=int)
  test_predict_proba = clf.predict_proba(X_test)
  test_predicted[np.where(test_predict_proba > 0.5)] = 1


def main():
    feature_dict, col_names = pp.set_targets('data/features.txt', threshold=-1)
    # consider only the terms of interest
    with open('data/terms.json', 'rb') as f:
       	terms = json.load(f)
    for key in list(feature_dict):
	feature_dict[key] = [x for x in terms if feature_dict[key][x] > THRESH]
	if not feature_dict[key]:
	    del(feature_dict[key])
    # get studies that have unique labels
    unique_feature_dict = ex.filter_studies_terms('data/features.txt', terms=terms, set_unique_label=True)
    # filter coordinates based on voxels
    coord_dict = ex.filter_studies_active_voxels('data/docdict.txt', 'data/MNI152_T1_2mm_brain.nii.gz', threshold=500, radius=6)
    # ensure that the keys are ints
    for key in list(coord_dict):
        if not isinstance(key, int):
            coord_dict[int(key)] = coord_dict[key]
            del(coord_dict[key])
    # train on studies with unique labels
    uniqtrain_coord_dict, uniqtrain_feature_dict = ex.get_intersecting_dicts(coord_dict, unique_feature_dict)
    X_train, y_train = pp.get_features_targets(uniqtrain_coord_dict, uniqtrain_feature_dict, mask='data/MNI152_T1_2mm_brain.nii.gz')
    # fit a label binarizer
    labels = sorted(terms)
    le = preprocessing.LabelEncoder()
    y_new_train = le.fit_transform(y_train)
    # build 1vsAll and use probabilities to do multiclass classification
    clf =  OneVsRestClassifier(LogisticRegression(penalty='l2'))
    model = clf.fit(X_train, y_new_train)
    # find the studies to test on
    test_keys = set(feature_dict.keys()) - set(uniqtrain_feature_dict.keys())
    test_feature_dict = {key: feature_dict[key] for key in test_keys}
    # find intersecting dicts
    test_coord_dict, test_feature_dict = ex.get_intersecting_dicts(coord_dict, test_feature_dict)
    X_test, y_test = pp.get_features_targets(test_coord_dict, test_feature_dict, labels=terms, mask='data/MNI152_T1_2mm_brain.nii.gz')
    lb = preprocessing.LabelBinarizer()
    y_new_test = lb.fit_transform(y_test)
    score_per_class = []
    score_per_label = []
    # convert to multilabel predictions
    predicted = np.zeros((len(y_new_test),len(labels)), dtype=int)
    predict_prob = clf.predict_proba(X_test)
    np.save('uniqtr_predicted.npy',predict_prob)
    predicted[np.where(predict_prob > 0.1)] = 1
    #convert top 2 values  to 1s
    for ind in range(predict_prob.shape[0]):
      sort_ind = np.argsort(predict_prob[ind])[-2:][::-1]
      np.put(predicted, sort_ind,1)
    cls_scores = utils.score_results(y_new_test, predicted, predict_prob)
    label_scores = utils.label_scores(y_new_test, predicted, predict_prob)
    score_per_class.append(cls_scores)
    score_per_label.append(label_scores)
    with open('uniqtr_class_scores.json', 'wb') as f:
        json.dump(score_per_class, f)
    pickle.dump(score_per_label, open('uniqtr_label_scores.p', 'wb'))
    return


if __name__ == '__main__':
    main()

