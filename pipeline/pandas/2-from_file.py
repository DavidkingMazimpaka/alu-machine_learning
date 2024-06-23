#!/usr/bin/env python3
''' Loads data from a file as a pd.DataFrame '''


import pandas as pd


def from_file(filename, delimiter):
    '''
    Loads data from a file as pd

    '''
    return pd.read_csv(filename, delimiter=delimiter)
