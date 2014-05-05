from __future__ import print_function
"""
functions to read in and construct numpy arrays from the fmri data
"""

# all paths mentioned in the foll code will be relative to this dir:
DIR = "/corral-repl/utexas/poldracklab/data/sanmi/workdir/data/openfMRI/raw/may_openfMRI"

import  os

import json
import numpy as np
import pandas
from collections import OrderedDict as odict

def get_X(file_name):
    """
    Function that simply reads in the X array from the numpy matrix already stored.
    'zstat_run1.npy' | 'zstat_run2.npy'.

    Parameters
    ----------
    file_name : str
        the file name that stores the numpy array, and will automatically be prefixed
        by DIR.

    Returns
    -------
    `numpy.ndarray` : (n_samples x n_features) array from brain images.
    """
    data_file = os.path.join(DIR, file_name)
    x = np.load(data_file)
    return x.transpose()

def get_Y(file_name, mapping, terms, get_dataframe=False):
    """
    Similarly form the labels corresponding to each study, returns a pandas dataframe
    that has terms as column names and frequencies as values. The file name will be
    prefixed by the DIR.

    Parameters
    ----------
    file_name :  str
        file name that has the frequency corresponding to each term for the 26 studies.
        e.g. 'cognitive_concepts/cognitive_concepts.txt' saved as numpy txt format.
    mapping : str
        file name containing study id for each of the 26 studies. e.g. 'data_key_run1.txt'
    terms : str
        file name that has the terms which are to be used  (json format)
    get_dataframe : bool, optional
        defines format in which to return the y values. When true returns the
        data frame rather than actual label list. Defaults to False.

    Returns
    -------
    y : `pandas.DataFrame`/ list of lists
        list of list of labels associated with each study, `pandas.dataframe` that has term vs frequency.
    """
    freq_mat = np.loadtxt(os.path.join(DIR, file_name))
    lines = open(os.path.join(DIR, mapping), 'rb').readlines()

    # process and sort lines to get the right mapping
    lines = [line.strip() for line in lines]
    lines = [x.split() for x in lines]
    lines = [[int(x) for x in l] for l in lines]
    lines = sorted(lines, key=lambda x: x[0])

    # construct dict from the terms used
    terms = json.load(open('data/terms_openfmri.json'))
    row_dict = {}
    for i in range(1, 27):
        col_dict = odict()
        for j in range(len(terms)):
            col_dict[terms[j]] = freq_mat[i-1][j]
        row_dict[i] = col_dict

    # reconstruct dict extending to 479 actual terms
    row_num = 0
    new_row_dict = odict()
    for line in lines:
        new_key = 'r' + str(row_num)
        row_num += 1
        new_row_dict[new_key] = row_dict[line[0]]

    # construct a dataframe
    df = pandas.DataFrame.from_dict(new_row_dict, orient='index')
    if get_dataframe:
        return df
    else:
        label_list = []
        for idx, row in df.iterrows():
            labels = [x for x in terms if row[x] > 0]
            label_list.append(labels)
        return label_list


