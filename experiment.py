from __future__ import print_function

import numpy as np
import json

import preprocess as pp

def filter_studies_active_voxels(study_dict, mask, threshold=5000, radius=10):
    """
    Takes the studies with coordinates, and considers only those studies that
    have number of activated voxels >= threshold. Returns the dict containing
    only those studies.

    Parameters
    ----------
    study_dict  :  dict/str
	      filename  to json stored dict or the dict of all the studies with coordinates.
    mask : mask in nifti format
    threshold : int, optional
        the number of activated voxels in the study, for it to be included.
        Defaults to 5000, as was used in the original study.
    radius : int, optional
	the radius of the sphere in mm to expand around the actual coordinates. Defaults
	to 10 mm.

    Returns
    -------
    study_dict : dict
        dict that includes only studies matching the criteria.
    """
    if isinstance(study_dict, basestring):
        with open(study_dict, 'rb') as f:
            study_dict = json.load(f)
    for key in list(study_dict.keys()):
        if len(study_dict[key]) < 0: # study has fewer than 4 reported foci
            del(study_dict[key])
        else:  # study has less than 5000 activated foci
            voxels = pp.peaks_to_vector(study_dict[key], mask, radius=radius)
            num_activated = (voxels > 0).sum()
            if num_activated < threshold:
                del(study_dict[key])
            else:
                study_dict[len(key)] = voxels
    return study_dict

def filter_studies_terms(feature_file, terms=None, threshold=0.001,
                         set_unique_label=False):
    """
    Given the frequency of terms corresponding to each study, as well as the
    tems to consider, eliminates all studies that have more than one term
    occuring at frequency >= threshold

    Parameters
    ----------
    feature_file : str
        the file with the raw features.
    terms : list of str, optional
        the terms that are being considered as labels. If not specified,
        uses the 25 terms from the original study.
    threshold : real, optional
        the frequency of the term for it to be considered, as significant with
        respect to the study. If not specified, uses 0.001 as by the original
        paper.
    set_unique_label : bool, optional
        defaults to false, when true returns only a single label corresponding
        to each study

    Returns
    -------
    feature_dict : dict
        the dict such that all studies with conflicting labels are eliminated.
    """
    if terms is None:
        terms = ['Semantic',
                'Encoding',
                'Executive',
                'Language',
                'Verbal',
                'Phonological',
                'Visual',
                'Inference',
                'Working Memory',
                'Conflict',
                'Spatial',
                'Attention',
                'Imagery',
                'Action',
                'Sensory',
                'Perception',
                'Auditory',
                'Pain',
                'Reward',
                'Arousal',
                'Emotion',
                'Social',
                'Episodic',
                'Retrieval',
                'Recognition'
                ]
    feature_dict, target_names = pp.set_targets(feature_file,
                                                    threshold=-1)
    # validate that the terms are actual features and convert to lower case
    new_terms = [x.lower() for x in terms if x.lower() in target_names]
    for key in list(feature_dict.keys()):
            # remove all studies that have more than one major term
            if len([x for x in new_terms if feature_dict[key][x] >
            threshold])>1:
                del(feature_dict[key])
    if set_unique_label:
        for key in list(feature_dict):
            vmax = 0
            label = None
            for x in new_terms:
                if feature_dict[key][x] > vmax:
                    vmax = feature_dict[key][x]
                    label = x
            if label is not None:
                feature_dict[key] = label
            else:
                del(feature_dict[key])
    return feature_dict

def get_intersecting_dicts(coordinate_dict, target_dict):
    """
    Given dicts with studies, filtered on diffierent criteria, convenience
    function that returns dicts having only the studies that pass all the
    criteria

    Parameters
    ----------
    coordinate_dict :  dict
        dict with studies - activation points pairs
    target_dict : dict
        dict with studies - terms pairs

    Returns
    -------
    new_coordinate_dict, new_target_dict :  tuple
        dicts with only the studies that appear in both the input dicts
    """
    new_coordinate_dict = {}
    new_target_dict = {}
    intersection = [key for key in target_dict if int(key) in coordinate_dict]
    for key in intersection:
        new_coordinate_dict[key] = coordinate_dict[key]
        new_target_dict[key] = target_dict[key]
        # much simpler to directly use dict comprehensions
        # not using to remain python 2.x compatible.
    return new_coordinate_dict, new_target_dict

