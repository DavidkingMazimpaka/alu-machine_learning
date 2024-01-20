#!/usr/bin/env python3

def matrix_shape(matrix):
    """
    Get the shape of a nested list representing a matrix.

    Parameters:
    - matrix (list): A nested list representing a matrix.

    Returns:
    - list: A list representing the shape of the matrix, where each element
            corresponds to the size of a dimension.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape

# Example usage:
if __name__ == "__main__":
    # Example matrix: a 3x2x4 matrix
    example_matrix = [
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ],
        [
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ],
        [
            [17, 18, 19, 20],
            [21, 22, 23, 24]
        ]
    ]

    # Get the shape of the matrix
    matrix_shape_result = matrix_shape(example_matrix)

    # Display the result
    print(f"The shape of the matrix is: {matrix_shape_result}")
