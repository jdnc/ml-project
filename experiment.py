from __future__ import print_function

import numpy as np
import json

import preprocess as pp

def filter_studies_active_voxels(study_dict=None, threshold=5000):
    """
    Takes the studies with coordinates, and considers only those studies that
    have number of activated voxels >= threshold. Returns the dict containing
    only those studies.

    Parameters
    ----------
    study_dict  :  dict, optional
        the dict of all the studies with coordinates. Optionally may directly
        load the precomputed dict from the data folder.

    threshold : int, optional
        the number of activated voxels in the study, for it to be included.
        Defaults to 5000, as was used in the original study.

    Returns
    -------
    study_dict : dict
        dict that includes only studies matching the criteria.
    """
    if study_dict is None:
        with open('data/docdict.txt', 'rb') as f:
            study_dict = json.load(f)
    for key in list(study_dict.keys()):
        if len(study_dict[key]) < 4: # study has fewer than 4 reported foci
            del(study_dict[key])
        else:  # study has less than 5000 activated foci 
            voxels = pp.peaks_to_vector(study_dict[key], 4)
            num_activated = (voxels > 0).sum()
            if num_activated < 5000:
                del(study_dict[key])
    return study_dict

def filter_studies_terms(feature_dict=None, terms=None, threshold=0.001):
    """
    Given the frequency of terms corresponding to each study, as well as the
    tems to consider, eliminates all studies that have more than one term
    occuring at frequency >= threshold

    Parameters
    ----------
    study_dict : dict, optional
        the dictionary with studies as keys and frequency corresponding to each
        term as values. If not specified, loads the precomputed dictionary from
        the data/ folder.
    terms : list of str, optional 
        the terms that are being considered as labels. If not specified,
        uses the 25 terms from the original study.
    threshold : real, optional
        the frequency of the term for it to be considered, as significant with
        respect to the study. If not specified, uses 0.001 as by the original
        paper.

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
    if feature_dict is None:
        feature_dict, target_names = pp.set_targets('data/features.txt', threshold=-1,
                                       terms=terms)
    for key in list(feature_dict.keys()):
        if len(feature_dict[key][feature_dict[key] > threshold]) > 1:
            del(feature_dict[key])
    return feature_dict
    
 
