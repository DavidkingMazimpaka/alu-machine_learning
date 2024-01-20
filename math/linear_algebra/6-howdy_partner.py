#!/usr/bin/env python3
def cat_arrays(arr1, arr2):
    concatenated_array = arr1 + arr2
    return concatenated_array


if __name__ == "__main__":
    arr1 = [1, 2, 3, 4, 5]
    arr2 = [6, 7, 8]
    result = cat_arrays(arr1, arr2)
    print(result)
    print(arr1)
    print(arr2)