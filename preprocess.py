from __future__ import print_function

import pandas
from collections import defaultdict

from neurosynth.base.dataset import Dataset:
import neurosynth.base.imageutils as nbi
import neurosynth.base.mask as nbm

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


def validate_coordinates(coordinates, mask):
    """
    Validates that the given x/y/z tuple is valid
    for the given mask

    Parameters
    ----------
    coordinates : list/tuple
        x, y, z coordinates to be validated
    mask : mask object in nifti format

    Returns
    -------
    bool : whether it is a valid coordinate or not
    """
    pass


def peaks_to_vector(coordinates, mask, radius=6):
    """
    Takes in a list of valid peak coordinates and 
    returns a vector of the corresponding image
    Parameters
    ----------
    coordinates : list of lists
        list of x/y/z coordinates
    mask : mask object in nifti fomat
        used to vectorize the image
    radius : int, optional
        the radius of sphere to expand around the peaks in mm.
        defaults to 6mm.

    Returns
    -------
    img_vector : 1D `numpy.array` object
        vectorized image to be used as a feature
    """
    # first get the denser image, expanding via spheres
    dense_img  = nbi.map_peaks_to_image(coordinates, r=radius)

    # now vectorize it to yield the 1D numpy array
    mask_obj = nbm.Mask(mask)
    img_vector = mask_obj.mask(dense_img)
    return mask_obj


    
def set_targets(filename, threshold=0):
    """
    Given the feature file, return the target vector showing 
    the presence/absence of terms for a document
    
    Parameters
    ----------
    filename : str
        complete path to file that has the raw data
    threshold : real, optional
        mark term only if its frequency > threshold. Defaults
        to 0

    Returns
    -------
    target_dict : `collections.defaultdict`
        the key is the study and the value is the target vector
    """
    target_dict = defaultdict(list)
    feature_table =  pandas.read_table('neurosynth/data/features.txt')
    for idx, row in feature_table.iterrows():
        target_dict[row['Doi']] = [int(x > threshold) for x in row[1:]]
    return target_dict



