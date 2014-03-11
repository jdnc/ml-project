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


