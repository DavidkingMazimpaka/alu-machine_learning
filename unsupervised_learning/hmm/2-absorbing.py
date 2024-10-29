#!/usr/bin/env python3
'''
    function def absorbing(P): that
    determines if a markov chain is absorbing
'''

import numpy as np


def absorbing(P):
    '''
    Determines if a markov chain is absorbing.
    A Markov chain is absorbing if:
    1. It has at least one absorbing state
    2. It's possible to go from each non-absorbing state to at least one absorbing state
    
    Args:
        P: numpy.ndarray transition matrix
    Returns:
        bool: True if chain is absorbing, False if not, None if invalid input
    '''
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    n1, n2 = P.shape
    if n1 != n2:
        return None
    # Check if row sums are 1 (valid probability matrix)
    if not np.allclose(np.sum(P, axis=1), 1):
        return None
    # Find absorbing states (states where probability of staying is 1)
    absorbing_states = np.where(np.diag(P) == 1)[0]
    # Must have at least one absorbing state
    if len(absorbing_states) == 0:
        return False
    # For non-absorbing states, check if they can reach an absorbing state
    non_absorbing = np.where(np.diag(P) != 1)[0]
    if len(non_absorbing) == 0:
        return True
    # Create matrix of non-absorbing states transitions
    R = P[non_absorbing][:, absorbing_states]  # transitions to absorbing states
    Q = P[non_absorbing][:, non_absorbing]     # transitions between non-absorbing states
    # Check if each non-absorbing state can reach an absorbing state
    I = np.eye(len(non_absorbing))
    try:
        N = np.linalg.inv(I - Q)  # fundamental matrix
        # If any entry in NR is 0, it means that state cannot reach any absorbing state
        reachability = np.dot(N, R)
        return np.all(reachability.sum(axis=1) > 0)
    except np.linalg.LinAlgError:
        # If I-Q is not invertible, some states cannot reach absorbing states
        return False
