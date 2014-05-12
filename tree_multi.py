from __future__ import print_function

"""
Multi-label classifier for neurosynth. Uses decision trees with
multi-label output from sklearn.
Initially begins with the 22 terms from the paper

launch -s run_tree -j Analysis_Lonestar -n dec_tree -p 240 -e 1way -m mparikh@cs.utexas.edu
"""


import numpy as np
import nibabel as nb
import json
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.cluster import WardAgglomeration
from sklearn.feature_extraction import image
from sklearn import tree

import pickle
import preprocess as pp
import experiment as ex
import utils

THRESH = 0.001
N_CLUSTERS = 5000

def classify(x, y):
    # fit a label binarizer
    lb = preprocessing.LabelBinarizer()
    y_new = lb.fit_transform(y)
    #specify connectivity for clustering
    mask = nb.load('data/MNI152_T1_2mm_brain.nii.gz').get_data().astype('bool')
    shape = mask.shape
    connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1], n_z=shape[2], mask=mask)
    ward = WardAgglomeration(n_clusters=N_CLUSTERS, connectivity=connectivity)
    clf = tree.DecisionTreeClassifier()
    kf = cross_validation.KFold(len(y_new), n_folds=5)
    score_per_class = []
    score_per_label = []
    for train, test in kf:
        train_feat = np.ascontiguousarray(x[train])
        train_labels = np.ascontiguousarray(y_new[train])
        test_feat = np.ascontiguousarray(x[test])
        test_labels = np.ascontiguousarray(y_new[test])
        ward.fit(train_feat)
        train_reduced = ward.transform(train_feat)
        test_reduced = ward.transform(test_feat)
        model = clf.fit(train_reduced, train_labels)
        predicted  = model.predict(test_reduced)
        predict_prob = model.predict_proba(test_reduced)
        cls_scores = utils.score_results(test_labels, predicted, predict_prob)
        label_scores = utils.label_scores(test_labels, predicted, predict_prob)
        score_per_class.append(cls_scores)
        score_per_label.append(label_scores)
    return (score_per_class,score_per_label)


def main():
    feature_dict, col_names = pp.set_targets('data/features.txt', threshold=-1)
    # consider only the terms of interest
    with open('data/reduced_terms.json', 'rb') as f:
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
    # get the respective vectors
    X, y = pp.get_features_targets(coord_dict, feature_dict, labels=terms, mask='data/MNI152_T1_2mm_brain.nii.gz')
    # perform clustering, cross validation and classify
    (score_per_class,score_per_label) = classify(X,y)
    pickle.dump(score_per_label,open('tree_label_scores.pkl','wb'))
    pickle.dump(score_per_class,open('tree_class_scores.pkl','wb'))
    return


if __name__ == '__main__':
    main()



