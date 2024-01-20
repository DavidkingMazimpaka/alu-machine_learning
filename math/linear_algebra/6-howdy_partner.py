#!/usr/bin/env python3
'''
This function computes two arrays
of the same lengths and returns the sum
of the arr[i]s in a new array
'''
def cat_arrays(arr1, arr2):
    '''
    computing two arrays
    '''
    concatenated_array = arr1 + arr2
    return concatenated_array


if __name__ == "__main__":
    arr1 = [1, 2, 3, 4, 5]
    arr2 = [6, 7, 8]
    result = cat_arrays(arr1, arr2)
    print(result)
    print(arr1)
    print(arr2)
