from __future__ import print_function

"""
Multi-label classifier for neurosynth.
Initially begins with the 25 terms from the paper
Do dimensionality reduction using PCA first.
Uses Logistic regression for multi-label classification and l1 regularization.
"""


import sys,ctypes
_old_rtld = sys.getdlopenflags()
sys.setdlopenflags(_old_rtld|ctypes.RTLD_GLOBAL)
from sklearn import decomposition
sys.setdlopenflags(_old_rtld)
import numpy as np
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
N_CLUSTERS = 5000
BATCH_SIZE = 2000
pca_components=2000

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
    # ensure that the keys are ints
    for key in list(coord_dict):
        if not isinstance(key, int):
            coord_dict[int(key)] = coord_dict[key]
            del(coord_dict[key])
    # find intersecting dicts
    coord_dict, feature_dict = ex.get_intersecting_dicts(coord_dict, feature_dict)
    # do PCA on a random sample
    coord_keys=coord_dict.keys()
    coord_sub_dict={ key: coord_dict[key] for key in coord_keys[0:BATCH_SIZE]}
    # get the respective vectors
    X_train, y_train = pp.get_features_targets(coord_sub_dict, feature_dict, labels=terms, mask='data/MNI152_T1_2mm_brain.nii.gz')
    # perform PCA
    pca = decomposition.PCA(n_components=pca_components)
    pca.fit(X_train)
    X_reduced=pca.transform(X_train)
    # do dim reduction
    y_full=y_train
    for chunk_start in range(BATCH_SIZE,len(coord_keys),BATCH_SIZE):
      coord_sub_dict={ key: coord_dict[key] for key in coord_keys[chunk_start:chunk_start+BATCH_SIZE]}
      # get the respective vectors
      X, y = pp.get_features_targets(coord_sub_dict, feature_dict, labels=terms, mask='data/MNI152_T1_2mm_brain.nii.gz')
      X_pca=pca.transform(X)
      if X_reduced is None:
        X_reduced=X_pca
      else:
        X_reduced=np.concatenate((X_reduced, X_pca))
      y_full.extend(y)
    # fit a label binarizer
    lb = preprocessing.LabelBinarizer()
    y_new = lb.fit_transform(y_full)

    clf =  OneVsRestClassifier(LogisticRegression(penalty='l1'))
    kf = cross_validation.KFold(len(y_new), n_folds=10)
    score_per_class = []
    score_per_label = []
    for train, test in kf:
        model = clf.fit(X_reduced[train], y_new[train])
        predicted  = model.predict(X_reduced[test])
        predict_prob = model.predict_proba(X_reduced[test])
        cls_scores = utils.score_results(y_new[test], predicted, predict_prob)
        label_scores = utils.label_scores(y_new[test], predicted, predict_prob)
        score_per_class.append(cls_scores)
        score_per_label.append(label_scores)
    with open('pca_class_scores.json', 'wb') as f:
        json.dump(score_per_class, f)
    pickle.dump(score_per_label, open('pca_label_scores.p', 'wb'))
    return


if __name__ == '__main__':
    main()


