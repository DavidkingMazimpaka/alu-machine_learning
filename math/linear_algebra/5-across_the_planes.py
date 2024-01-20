#!/usr/bin/env python3
def add_matrices2D(mat1, mat2):
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    result = []

    for i in range(len(mat1)):
        row_result = []
        for j in range(len(mat1[0])):
            row_result.append(mat1[i][j] + mat2[i][j])
        result.append(row_result)

    return result


if __name__ == "__main__":
    mat1 = [[1, 2], [3, 4]]
    mat2 = [[5, 6], [7, 8]]
    print(add_matrices2D(mat1, mat2))
    print(mat1)
    print(mat2)
    print(add_matrices2D(mat1, [[1, 2, 3], [4, 5, 6]]))
