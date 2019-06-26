import argparse
import random
import numpy as np
import sys
import torch

__all__ = ['explorer_helper', 'vocab']

def check_args(argv):
    """Checks and parses the arguments of the command typed by the user
    Parameters
    ----------
    argv :
        The arguments of the command typed by the user
    Returns
    -------
    ArgumentParser
        the values of the arguments of the commande typed by the user
    """
    parser = argparse.ArgumentParser(description="Train model with a specific configuration.")
    parser.add_argument('CONFIG', type=str, help="the name of the configuration file (JSON file)")

    args = parser.parse_args()

    return args

def fix_seed(seed=None):
    """Fixes a seed for reproducibility
    Parameters
    ----------
    seed :
        The seed to use
    Returns
    -------
    int
        the seed that will be used
    """
    if seed is None:
        seed = time.time()
    
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return seed