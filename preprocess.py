from __future__ import print_function

import pandas
import numpy as np
from collections import defaultdict

from neurosynth.base.dataset import Dataset
import neurosynth.base.imageutils as nbi
import neurosynth.base.mask as nbm
import neurosynth.base.transformations as nbt


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
        if is_valid([x,y,z]):
            doc_dict[row['ID']].append([x,y,z])

    return doc_dict


def is_valid(coordinates, mask='2mm_brain_mask.npy'):
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
    try:
        mat = np.load(mask)
        idx = nbt.xyz_to_mat(np.asarray([coordinates]))
        idx_t = tuple(idx.reshape(1, -1)[0])
        return mat[idx_t] == 1
    except IndexError as e:
        return False


def peaks_to_vector(coordinates, mask=
                    'neurosynth/neurosynth/resources/MNI152_T1_2mm_brain.nii.gz', 
                    radius=6):
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
    return img_vector


    
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
    target_names : list
        the name of the target labels used
    """
    target_dict = defaultdict(list)
    feature_table =  pandas.read_table('neurosynth/data/features.txt')
    target_names = feature_table.columns[1:]
    for idx, row in feature_table.iterrows():
        target_dict[row['Doi']] = [int(x > threshold) for x in row[1:]]
    return (target_dict, target_names)


def get_features_targets():
    pass
