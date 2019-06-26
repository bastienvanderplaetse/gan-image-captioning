"""Some functions to manage files and directories
This script contains some functions to help the user to manage files and
directories.
"""

import csv
import json
import os
import tabulate

tabulate.LATEX_ESCAPE_RULES={}

def create_directory(dir_name):
    """Creates a directory if it does not exist
    Parameters
    ----------
    dir_name : str
        The name of the directory to create
    """
    if not os.path.exists(dir_name) or not os.path.isdir(dir_name):
            os.mkdir(dir_name)

def load_json(filename):
    """Loads a JSON file
    Parameters
    ----------
    filename : str
        The name of the JSON file to load
    Returns
    -------
    dict
        the content of the JSON file
    """
    with open(filename, 'r') as f:
        return json.load(f)

def read_file(filename):
    """Returns all lines of a text file
    Parameters
    ----------
    filename : str
        The name of the file to read
    Returns
    -------
    list
        a list containing each line of the file
    """
    f = open(filename)
    lines = f.readlines()
    lines = [line.replace('\n', '') for line in lines]
    f.close()
    
    return lines

def write_json(data, filename):
    """Saves data into a JSON file
    Parameters
    ----------
    data :
        The data to save in a JSON file
    filename : str
        The name of the JSON file in which the data must be saved
    """
    with open(filename, 'w') as fp:
        json.dump(data, fp)

def write_text(data, filename):
    """Saves data into a text file
    Parameters
    ----------
    data :
        The data to save in a text file
    filename : str
        The name of the text file in which the data must be saved
    """
    with open(filename, 'w') as fp:
        fp.write(data)