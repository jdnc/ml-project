from __future__ import print_function
#!/usr/bin/env python

"""
Python code that performs single-label classification using different classifiers
Classifiers = MultinomialNB, LogisticRegression, LinearSVM, Ensemble of these three.
Configurable = Parameters for each classifier
Clustering(Ward) = True/False
"""

import json
import numpy as np

from sklearn.naive_bayes import MultinomialNB  # ---------------------- Naive Bayes
from sklearn.svm import LinearSVC # ----------------------------------- Linear SVM
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression # ----------------- Logistic Regression


from sklearn.cluster import WardAgglomeration
from sklearn.feature_extraction import image
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

import preprocess as pp
import experiment as ex

# ------------------- define basic checks ----------

def validate_classifier(func):
    """
    Helper function decorator to check if user passes the right classifier
    """
    def wrapper(*args, **kwargs):
        classifier = kwargs.get('classifier')
        if not classifier in ['naive_bayes', 'svm', 'logistic_regression', 'ensemble']:
            throw ValueError("Supported classifiers are 'naive_bayes', 'svm', 'logistic_regression', 'ensemble'")
        else:
            return func(*args, **kwargs)
    return wrapper  

# --------------------------------------------------

@validate_classifier
def classify(x, y, classifier='naive_bayes', clustering=True, n_folds=10):
    """
    Given the predictors and labels, performs single-class
    classification with the given classifier using n-fold
    c.v. Constructs a OvO classifier for every pair of terms.
    
    Parameters
    -----------
    x : `numpy.ndarray`
        (n_samples x n_features) array of features
    y : `numpy.ndarray`
        (1 x n_samples) array of labels
    classifier : str, optional
        which classifier model to use. Must be one of 'naive_bayes'| 'svm' | 'logistic_regression' | 'ensemble'.
        Defaults to the original naive_bayes.
    clustering : bool, optional
        whether to do Ward clustering or not. Uses n_clusters = 10,000. Change global N_CLUSTERS for different
        value. Defaults to True.
    n_folds : int
        the number of fold of cv
        
    Returns
    -------
    accuracy : `numpy.ndarray`
        The results are stored as a list of confusion matrices for each fold and saved
        as a numpy array of arrays, for further analysis.
    """
    clf = None
    ward = None
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y_new = le.transform(y)
    
    # choose and assign appropriate classifier
    classifier_dict = { 'naive_bayes' : MultinomialNB(),
                        'logistic_regression' : LogisticRegression(penalty='l2')
                        'svm' : GridSearchCV(LinearSVC(), [{'C': [1, 10, 100, 1000]}])  
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
    with open('data/docdict.txt') as f:
    	coordinates = json.load(f)
    study_dict = ex.filter_studies_active_voxels(coordinates, 'data/MNI152_T1_2mm_brain.nii.gz', radius=6, threshold=500)
    # ensure that study dict has int as keys
    for key in list(study_dict):
        study_dict[int(key)] = study_dict[key]
        if not isinstance(key, int):
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

