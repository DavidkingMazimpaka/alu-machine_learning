#!/usr/bin/env python3
''' A function that creates a
pd.DataFrame from np.ndarray'''


import pandas as pd


def from_numpy(array):
    '''
    Creates pd.DataFrame from np.ndarray

    Args:
    array -> numpy array

    Returns:
    DataFrame
    '''
    num_columns = array.shape[1]
    columns = [chr(i) for i in range(65, 65 + num_columns)]

    df = pd.DataFrame(array, columns=columns)

    return df