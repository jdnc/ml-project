from __future__ import print_function

"""
Consolidates all the multi-label approaches tried so far
Classifiers : Naive Bayes, Logistic Regression, unique train
Clustering(Ward) : yes/no
Uses One vs Rest models for multilabel
Also added : Decision tree, classifier chains
"""


import numpy as np
import json
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import WardAgglomeration
from sklearn.feature_extraction import image


import preprocess as pp
import experiment as ex
import utils

THRESH = 0.001
N_CLUSTERS = 10000


def classify(x, y, classifier='naive_bayes', clustering=True, n_folds=10):
    """
    Given the predictors and labels, performs multi-label 
    classification with the given classifier using n-fold
    c.v. Constructs a OvR classifier for multilabel prediction.
    
    Parameters
    -----------
    x : `numpy.ndarray`
        (n_samples x n_features) array of features
    y : `numpy.ndarray`
        (n_samples x n_labels) array of labels
    classifier : str, optional
        which classifier model to use. Must be one of 'naive_bayes'| 'knn' | 'logistic_regression' | 'unique_train'.
        Defaults to the original naive_bayes.
    clustering : bool, optional
        whether to do Ward clustering or not. Uses n_clusters = 10,000. Change global N_CLUSTERS for different
        value. Defaults to True.
    n_folds : int
        the number of fold of cv
        
    Returns
    -------
    score_per_label, score_per_class : tuple
        The results are stored as a tuple of two dicts, with the keywords specifying the metrics.
    """
    clf = None
    ward = None
    lb = preprocessing.LabelBinarizer()
    y_new = lb.fit_transform(y)
    
    score_per_class = []
    score_per_label = []
    
    # choose and assign appropriate classifier
    classifier_dict = { 'naive_bayes' : MultinomialNB(),
                        'logistic_regression' : LogisticRegression(penalty='l2'),
                        'unique_train' : 
                        
                       }
    if classifier == 'ensemble':
      clf_nb = classifier_dict['naive_bayes']
      clf_svm = classifier_dict['svm']
      clf_lr = classifier_dict['logistic_regression']
    else:
        clf = classifier_dict[classifier]
        
    # perform ward clustering if specified    
    if clustering:
        mask = np.load('data/2mm_brain_mask.npy')
        shape = mask.shape
        connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1], n_z=shape[2], mask=mask)
        ward = WardAgglomeration(n_clusters=N_CLUSTERS, connectivity=connectivity)
    
    # actual cross validation    
    kf = cross_validation.KFold(len(y_new), n_folds=n_folds)
    accuracy = []
    for train, test in kf:
        x_train = x[train]
        y_train  = y_new[train]
        x_test = x[test]
        y_test = y_new[test] 
        if clustering:
            ward.fit(x_train)
            x_train = ward.transform(x_train)
            x_test = ward.transform(x_test)
        if classifier != 'ensemble':        
            predicted = clf.fit(x_train, y_train).predict(x_test)
        else:
            predicted_nb = clf_nb.fit(x_train, y_train).predict(x_test)
            predicted_lr = clf_lr.fit(x_train, y_train).predict(x_test)
            predicted_svm = clf_svm.fit(x_train, y_train).predict(x_test)
            predicted = predicted_nb + predicted_lr + predicted_svm
            predicted = np.array(predicted >= 2, dtype=int)
        conf_mat =  confusion_matrix(y_test, predicted, labels=[0,1])
        accuracy.append(conf_mat)
    return accuracy


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
    # get the respective vectors
    X, y = pp.get_features_targets(coord_dict, feature_dict, labels=terms, mask='data/MNI152_T1_2mm_brain.nii.gz')
    # fit a label binarizer
    lb = preprocessing.LabelBinarizer()
    y_new = lb.fit_transform(y)
    # perform the 10 fold cross_validation
    score_per_class = []
    score_per_label = []
    clf =  OneVsRestClassifier(LogisticRegression(penalty='l1'))
    kf = cross_validation.KFold(len(y_new), n_folds=10)
    for train, test in kf:
        model = clf.fit(X[train], y_new[train])
        predicted  = model.predict(X[test])
        predict_prob = model.predict_proba(X[test])
        cls_scores = utils.score_results(y_new[test], predicted, predict_prob)
        label_scores = utils.label_scores(y_new[test], predicted, predict_prob)
        score_per_class.append(cls_scores)
        score_per_label.append(label_scores)
    with open('class_scores.json', 'wb') as f:
        json.dump(score_per_class, f)
    with open('label_scores.json', 'wb') as f:
        json.dump(score_per_label, f)
    return


if __name__ == '__main__':
    main()




