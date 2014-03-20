from __future__ import print_function

import pandas
import numpy as np
from nilearn import input_data
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
        for key in list(doc_dict.keys()):
            if not doc_dict[key]:
                del(doc_dict[key])
                

    return doc_dict


def is_valid(coordinates, mask='data/2mm_brain_mask.npy'):
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
                    radius=10):
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
        defaults to 10mm.

    Returns
    -------
    img_vector : 1D `numpy.array` object
        vectorized image to be used as a feature
    """
    # transform the coordinates to matrix space
    #print(coordinates)
    new_coordinates = nbt.xyz_to_mat(np.array(coordinates))
    # now  get the denser image, expanding via spheres
    dense_img  = nbi.map_peaks_to_image(new_coordinates, r=radius)
    # now vectorize the image
    img_vector = dense_img.get_data().ravel()
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
        to 0. If -1 then simply returns the raw count.

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
    if threshold == -1:
        target_dict = {}
        for idx, row in feature_table.iterrows():
            target_dict[row['Doi']] = row[1:]
    else:
        for idx, row in feature_table.iterrows():
            target_dict[row['Doi']] = [int(x > threshold) for x in row[1:]]
    return (target_dict, target_names)


def get_features_targets(coordinate_dict, target_dict,
                         mask='neurosynth/neurosynth/resources/MNI152_T1_2mm_brain.nii.gz'):
    """
    Given the dicts that have the list of coordinates and the list of targets
    corresponding to each study, returns the numpy arrays as expected by
    scikit-learn classifier functions.

    Parameters
    ----------
    coordinate_dict : dict
        a dict having the studies as the keys and the list of coordinates for
        that study as the values. The coordinates are the raw coordinates from
        the text without any transformation to the matrix space
    target_dict : dict
        the dict that has the study as the key and the presence absence of
        terms as values
    mask: mask in Nifti format, optional
        fits the X array to be within the bounds of the mask.

    Returns
    -------
    (X, y) : X is the n_samples x n_features array for the data input to
        scikit-learn. y is the n_samples x n_classes array of targets.
    """

    n_samples = len(coordinate_dict)
    n_features = 902629 # 91 * 109 * 91
    n_classes = len(target_dict.values()[0])
    X = np.zeros((n_samples, n_features))
    y = np.zeros((n_samples, n_classes))
    for idx, key in enumerate(coordinate_dict):
        X[idx] = peaks_to_vector(coordinate_dict[key])
        y[idx] = target_dict[key]
    nifti_masker = input_data.NiftiMasker(mask=mask, memory_level=1,
                                          standardize=False)
    X = nifti_masker.fit_transform(np.array(X))
    return X, y

def features_targets_from_file(db_file, feature_file, threshold=0):
    """
    Takes the absolute filenames of the data and feature files and returns the
    numpy arrays expected by scikit learn algorithms

    Parameters
    ----------
    db_file : str
        absoulte path to the file that has the raw coordinates data
    feature_file : str
        absolute path to the file that has the features and term frequencies
        corresponding to each study
    threshold : real, optional
        term present only if frequency > threshold, defaults to 0
    Returns
    -------
    (X, y) : X is the n_samples x n_features array for the data input to
        scikit-learn. y is the n_samples x n_classes array of targets.
    """

    coordinate_dict = extract_coordinates(db_file)
    target_dict, target_names = set_targets(feature_file, threshold=threshold)
    X, y = get_features_targets(coordinate_dict, target_dict)
    return (X, y)



