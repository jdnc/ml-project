from __future__ import print_function
"""
Given a  dict with keys as studies and vals as list of labels corresponding to each study,
plots the histogram of the label counts.

Assumes the existence of mapping.json in the current directory, if the labels are ints rather than str.
mapping.json is dict from str --> int
"""

import json
import pandas
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_histogram(y):
    """
    Plot frequency bar plot of items from  a given numpy array/ a list of lists.

    Parameters
    -----------
    y : numpy array/ list of lists/ string
        either a list of list type construct or the filename of the json
        serialized object

    Returns
    -------
    None
    """
    # check whether json file or actual object
    if isinstance(y, basestring):
        y = np.load(y)

    # count the occurence of each label
    count_dict = defaultdict(int)
    for labels in y:
        for label in labels:
            count_dict[label] += 1
    # tranlate the int labels back to their string form using mappings.json
    inv_map = {}
    map = json.load(open('mappings.json', 'rb'))
    for key in map:
        inv_map[map[key]] = key

    # similarly map counts back to original strings
    term_counts =  {}
    for key in count_dict:
        term_counts[inv_map[key]] = count_dict[key]

    # plot easily by converting the count_dict to a pandas Series object
    count_series = pandas.Series(term_counts)
    count_series.sort(axis=1)
    count_series.plot(kind='bar', figsize=(10,10))
    plt.ylabel('number of studies')
    plt.savefig('histogram.png')
