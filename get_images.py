from __future__ import print_function
"""
A simple module that fetches images from the neurovault database and stores
them in the specified directory
"""
import os
import urllib
import json

def get_images(response_file, save_dir):
    """
    Expects a list of dicts from the HTTP get requests. Fetches images from
    the specified URL and stores it in the  specified directory.

    Parameters
    ----------
    response_file : str
        the file that has the GET response from the Neurovault API
    save_dir : str
        the complete path of the directory in which the img files must
        be saved.

    Returns
    -------
    None
    """
    with open(response_file, 'rb') as f:
        dict_list = json.dump(f)
    for d in dict_list:
        file_url = d['file']
        file_name = file_url.split('/')[-1]
        save_name = os.path.join(save_dir, file_name)
        urllib.urlretrieve(file_url, save_name)


