from __future__ import print_function

"""
Tal's baseline classifier.
Initially begins with the 25 terms from the paper
"""

import os
import sys
import argparse

import numpy as np
import json
from sklearn import preprocessing
from neurosynth.base.dataset import Dataset
from neurosynth.analysis import meta
from scipy.stats import ss

import preprocess as pp
import experiment as ex
import get_openfmri_data as openfmri
import utils

def pearson(x, y):
    """ Correlates row vector x with each row vector in 2D array y. """
    data = np.vstack((x, y))
    ms = data.mean(axis=1)[(slice(None, None, None), None)]
    datam = data - ms
    datass = np.sqrt(ss(datam, axis=1))
    temp = np.dot(datam[1:], datam[0].T)
    rs = temp / (datass[1:] * datass[0])
    return rs

THRESH = 0.001
def main():
    dataset = Dataset('data/database.txt')
    dataset.add_features('data/features.txt')
    # consider only the terms of interest
    with open('data/new_terms.json', 'rb') as f:
	terms = json.load(f)
    # generate z-map to analyze features
    label_zmaps = meta.analyze_features(dataset, terms)
    label_zmaps = label_zmaps.transpose()

    test_X = openfmri.get_X('zstat_run1.npy')
    test_y = openfmri.get_Y('cognitive_concepts/cognitive_concepts.txt','data_key_run1.txt','data/new_terms.json')
    lb = preprocessing.LabelBinarizer()
    test_y_new = lb.fit_transform(test_y)
    predicted = np.zeros(test_y_new.shape, dtype=int)
    # find correlation
    for ind in range(test_X.shape[0]):
      test_img=test_X[ind]
      corr = pearson(test_img, label_zmaps)
      abs_corr = np.absolute(corr)
      #label_ordered = sorted(range(len(abs_corr)),key=abs_corr.__getitem__)
      lab_indices = np.argsort(abs_corr)[-2:][::-1]
      np.put(predicted[ind],lab_indices, 1)
    predict_prob = predicted
    score_per_class = []
    score_per_label = []
    cls_scores = utils.score_results(test_y_new, predicted, predict_prob)
    label_scores = utils.label_scores(test_y_new, predicted, predict_prob)
    score_per_class.append(cls_scores)
    score_per_label.append(label_scores)
    with open('tal_class_scores.json', 'wb') as f:
        json.dump(score_per_class, f)
    with open('tal_label_scores.json', 'wb') as f:
        json.dump(score_per_label, f)
    return


if __name__ == '__main__':
    main()




