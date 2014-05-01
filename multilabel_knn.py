from __future__ import print_function

"""
Multi-label classifier for neurosynth.
Initially begins with the 25 terms from the paper
Use k-nearest neighbours to classify.
"""


import numpy as np
import json
import pickle
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors

import preprocess as pp
import experiment as ex
import utils

THRESH = 0.001
k = 3 # number of neighbors

def get_predictions(knn_neighbors, y_new):
  """Compute probability of each label based on frequency of neighbor labels
     Input: Neighbours, y- the labels (not binarized)
  """
  predictions=[]
  #predicted_prob_3 = [y_new[neighbors[0]]+y_new[neighbors[1]]+y_new[neighbors[2]] for neighbors in knn_neighbors]
  predicted_proba=[]
  for neighbors in knn_neighbors:
    predicted_labs=np.zeros(len(y_new[0]), dtype=int)
    for i in range(0,k):
      predicted_labs+=y_new[neighbors[i]]
    predicted_proba.append(predicted_labs/(k*1.0))
    label_ones = predicted_labs >= 1
    predicted_labs[label_ones]=1
    predictions.append(predicted_labs)
  return (predictions, predicted_proba)

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
    #try for a sample
    sub_coord_dict = {k: coord_dict[k] for k in coord_dict.keys()[0:4000]}
    sub_coord_dict, sub_feature_dict = ex.get_intersecting_dicts(sub_coord_dict, feature_dict)
    X, y = pp.get_features_targets(sub_coord_dict, sub_feature_dict, labels=terms, mask='data/MNI152_T1_2mm_brain.nii.gz')
    # get the respective vectors
    #X, y = pp.get_features_targets(coord_dict, feature_dict, labels=terms, mask='data/MNI152_T1_2mm_brain.nii.gz')
    # fit a label binarizer
    lb = preprocessing.LabelBinarizer()
    y_new = lb.fit_transform(y)
    # perform the 10 fold cross_validation
    score_per_class = []
    score_per_label = []
    predicted_labels=[]
    test_vals=[]
    knnModel = NearestNeighbors(n_neighbors=k)
    kf = cross_validation.KFold(len(y_new), n_folds=10)
    cvrun=1
    for train, test in kf:
        model = knnModel.fit(X[train])
        ng_dist, ng_indices  = knnModel.kneighbors(X[test])
	(predicted, predict_proba) = get_predictions(ng_indices, y_new[train])
        cls_scores = utils.score_results(y_new[test], predicted, predict_prob)
        label_scores = utils.label_scores(y_new[test], predicted, predict_prob)
        score_per_class.append(cls_scores)
        score_per_label.append(label_scores)
	predicted_labels.append(predicted)
        test_vals.append(y_new[test])
        cvrun+=1
    with open('knn_class_scores.json', 'wb') as f:
        json.dump(score_per_class, f)
    pickle.dump(score_per_label, open('knn_label_scores.p', 'wb'))
    return


if __name__ == '__main__':
    main()

