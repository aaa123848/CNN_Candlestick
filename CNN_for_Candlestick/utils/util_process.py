import pandas as pd
import numpy as np
import pickle

def load_pkl(pkl_name):
    '''
    Args:
        pkl_name (string): path of pickle name

    Returns:
        data (dict): with the following keys
            "train_data" (numpy): (train_N, 32, 4)
            "train_gaf" (numpy): (train_N, 32, 32, 4)
            "train_label" (numpy): (train_N, 3)
            "train_label_arr" (numpy): (train_N, 9)
            "val_data" (numpy): (val_N, 32, 4)
            "val_gaf" (numpy): (val_N, 32, 32, 4)
            "val_label" (numpy): (val_N, 3)
            "val_label_arr" (numpy): (val_N, 9)
            "test_data" (numpy): (test_N, 32, 4)
            "test_gaf" (numpy): (test_N, 32, 32, 4)
            "test_label" (numpy): (test_N, 3)
            "test_label_arr" (numpy): (test_N, 9)
    '''
    with open(pkl_name, "rb") as f:
        data = pickle.load(f)
    return data


