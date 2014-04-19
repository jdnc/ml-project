from __future__ import print_function
"""
A script that takes in a dict or jsonized dict of terms corresponding to any study
and plots a  bar chart plotting frequency vs term
"""

import json
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter


def plot_histogram(feature_dict):
    """
    A function that takes a dictionary with the keys being the studies and the
    values being all the labels for that study. Plots a bar chart showing the
    overall counts of each label

    Parameters
    ----------
    feature_dict : dict/str
        A dict that has the studies as its keys and the labels as its values.
        Can also be a string which is a filepath to the jsonized dict

    Returns
    -------
    None
        simply plots the bar chart if possible.
    """
    if isinstance(feature_dict, basestring):
        with open(feature_dict, 'rb') as f:
            feature_dict = json.load(f)
    # now generate the counts for each term
    term_counts = Counter()
    for key in feature_dict:
        for word in feature_dict[key]:
            term_counts[word] += 1
    # now load the terms from the actual term file
    with open('data/terms.json') as f:
        terms = json.dump(f)

    # set the parameters for the actual plot
    # in vein with the example at matplotlib
    # see url <http://matplotlib.org/1.3.1/examples/api/barchart_demo.html>
    n = len(terms)
    ind = np.arange(n)
    width = 0.35
    fig, ax = plt.subplots()
    ax.set_ylabel('count')
    ax.set_xticks(ind+width)
    ax.set_xticklabels(terms)
    counts = [term_counts[x] for x in terms]
    rects = ax.bar(ind+width, counts, width, color='b')
    plt.show()
