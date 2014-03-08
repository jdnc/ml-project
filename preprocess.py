from __future__ import print_function

import pandas
from collections import defaultdict

from neurosynth.base.dataset import Dataset:
import neurosynth.base.imageutils as iu


def extract_coordinates(filename):
    """
    Takes in the raw database file and extracts coordinates corresponding
    to each document.

    Parameters
    ----------
    filename : str
        complete path to file that has the raw data

    Returns
    -------
    doc_dict : `collections.defaultdict`
        a list of 3-tuples of x/y/z coordinates corresponding to each document.
    """

    doc_dict = defaultdict(list)
    data_table =  pandas.read_table('neurosynth/data/database.txt')
    for idx, row in data_table.iterrows():
        x = row['x']
        y = row['y']
        z = row['z']
        doc_dict[row['ID']] .append([x,y,z])

    return doc_dict



