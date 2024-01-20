#!/usr/bin/env python3
'''
This function computes two arrays
of the same lengths and returns the sum
of the arr[i]s in a new array
'''


def add_arrays(arr1, arr2):
    '''
    computes two arrays
    of the same lengths
    '''
    if len(arr1) != len(arr2):
        return None
    result = []
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])
    return result
