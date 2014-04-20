from __future__ import print_function
import os
import re

import json
import pandas
import numpy as np
from collections import defaultdict

from neurosynth.base.dataset import Dataset
import neurosynth.base.imageutils as nbi
import neurosynth.base.mask as nbm
import neurosynth.base.transformations as nbt


def extract_coordinates(filename, mask):
    """
    Takes in the raw database file and extracts coordinates corresponding
    to each document.

    Parameters
    ----------
    filename : str
        complete path to file that has the raw data
    mask : mask object in nifti format

    Returns
    -------
    doc_dict : `collections.defaultdict`
        a list of 3-tuples of x/y/z coordinates corresponding to each document.
    """

    doc_dict = defaultdict(list)
    data_table =  pandas.read_table(filename)
    for idx, row in data_table.iterrows():
        x = row['x']
        y = row['y']
        z = row['z']
        if is_valid([x,y,z], mask):
            doc_dict[int(row['id'])].append([x,y,z])
    for key in list(doc_dict.keys()):
        if not doc_dict[key]:
            del(doc_dict[key])
    return doc_dict


def is_valid(coordinates, mask):
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
        defaults to 10mm.

    Returns
    -------
    dense_img : nifti image
        1D Numpy array of in-mask voxels
    """
    # transform the coordinates to matrix space
    #print(coordinates)
    new_coordinates = nbt.xyz_to_mat(np.array(coordinates))
    # now  get the denser image, expanding via spheres
    dense_img  = nbi.map_peaks_to_image(new_coordinates, r=radius)
    # Create a mask object for the image
    niftiMask = nbm.Mask(mask)
    # mask the image formed
    return niftiMask.mask(dense_img)
    #img_vector = dense_img.get_data().ravel()
    #return img_vector


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
    feature_table =  pandas.read_table(filename)
    target_names = feature_table.columns[1:]
    if threshold == -1:
        target_dict = {}
        for idx, row in feature_table.iterrows():
            target_dict[int(row['pmid'])] = row[1:]
    else:
        for idx, row in feature_table.iterrows():
            target_dict[int(row['pmid'])] = [int(x > threshold) for x in row[1:]]
        for key in list(target_dict):
            if not target_dict[key]:
                del(target_dict[key])
    return (target_dict, target_names)


def get_features_targets(coordinate_dict, target_dict, labels=None, mask=None, is_voxels=False):
    """
    Given the dicts that have the list of coordinates and the list of targets
    corresponding to each study, returns the numpy arrays as expected by
    scikit-learn classifier functions.

    Parameters
    ----------
    coordinate_dict : dict
        a dict having the studies as the keys and the list of coordinates for
        that study as the values. The coordinates are the raw coordinates from
        the text without any transformation to the matrix space. Alternately
        this may be the list of voxels corresponding to the study, rather than
        the raw coordinates, 'is_voxels' must be set to true.
    target_dict : dict
        the dict that has the study as the key and the presence absence of
        terms as values
    labels : iterable, optional
	a list of all the unique labels for the learner. Default is None.
	If specified, it will generate a mapping from the labels to numeric values.
	Not specifying may lead to erros later on. The mapping is stored in
	current working dict as 'mapping.json', is a python dict.
    mask: mask in Nifti format
        fits the X array to be within the bounds of the mask.
    is_voxels : bool, optional
        defines the format of the coordinate_dict. When true, the values
        in coordinate_dict are the voxels, rather than the raw coordinates.
        Defaults to false.

    Returns
    -------
    (X, y) : X is the n_samples x n_features array for the data input to
        scikit-learn. y is the n_samples x n_classes array of targets.
    """
    n_samples = len(coordinate_dict)
    #n_classes = len(target_dict.values()[0]) don't know what this means - has no use
    n_features = 228453 # 91 * 109 * 91 now reduced!
    X = np.zeros((n_samples, n_features), dtype=int)
    #y = np.empty(n_samples, dtype=object)
    if labels:
	mapping = {}
	y = []
	for i in range(len(labels)):
	    mapping[labels[i]] = i
	with open('mappings.json', 'wb') as f:
	    json.dump(mapping, f)
    else:
        y = np.empty(n_samples, dtype=object)
    for idx, key in enumerate(coordinate_dict):
        if labels:
            y.append([mapping[x] for x in target_dict[key]])
        else:
            y[idx] = target_dict[key]
        X[idx] = coordinate_dict[key] if is_voxels else peaks_to_vector(coordinate_dict[key], mask)
    return X, y

def features_targets_from_file(db_file, feature_file, mask, threshold=0):
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
    mask : mask in nifti format, optional
        may not be specified if giving the coordinate dict values as voxels.
    threshold : real, optional
        term present only if frequency > threshold, defaults to 0
    Returns
    -------
    (X, y) : X is the n_samples x n_features array for the data input to
        scikit-learn. y is the n_samples x n_classes array of targets.
    """

    coordinate_dict = extract_coordinates(db_file, mask)
    target_dict, target_names = set_targets(feature_file, threshold=threshold)
    X, y = get_features_targets(coordinate_dict, target_dict, 'data/MNI152_T1_2mm_brain.nii.gz')
    return (X, y)



