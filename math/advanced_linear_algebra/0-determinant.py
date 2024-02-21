#!/usr/bin/env python3
"""
Function to calculate the determinant of a square matrix.
"""

def determinant(matrix):
    """
    Calculates the determinant of a square matrix.

    Args:
        matrix (list of lists): The matrix whose determinant should be calculated.

    Raises:
        TypeError: If matrix is not a list of lists.
        ValueError: If matrix is not square.

    Returns:
        float: The determinant of the matrix.
    """
    # Check if matrix is a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    
    # Check if matrix is square
    n_rows = len(matrix)
    n_cols = len(matrix[0])
    if n_rows != n_cols:
        raise ValueError("matrix must be a square matrix")
    
    # Base case: 0x0 matrix has determinant 1
    if n_rows == 0:
        return 1
    
    # Base case: 1x1 matrix
    if n_rows == 1:
        return matrix[0][0]
    
    # Base case: 2x2 matrix
    if n_rows == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    # Recursive case: Use Laplace expansion along the first row
    det = 0
    for j in range(n_cols):
        minor = [row[:j] + row[j+1:] for row in matrix[1:]]
        det += (-1) ** j * matrix[0][j] * determinant(minor)
    return det

# Test the function
if __name__ == "__main__":
    matrix = [[-2, -4, 2], [-2, 1, 2], [4, 2, 5]]
    print(determinant(matrix))  # Output should be -79
