#!/usr/bin/env python3
"""
function that calculates  most likely sequence of HS for
the HMM
"""


import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    # Ensure inputs are valid (as you have done)
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None
    T = Observation.shape[0]
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    N, M = Emission.shape
    if not isinstance(Transition, np.ndarray) or Transition.shape != (N, N):
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.shape != (N, 1):
        return None, None
    # Initialize the Viterbi matrix and path tracking
    V = np.zeros((N, T))
    path = np.zeros(T, dtype=int)
    # Initialization step
    V[:, 0] = Initial.flatten() * Emission[:, Observation[0]]
    # Recursion step
    for t in range(1, T):
        for j in range(N):
            trans_prob = V[:, t-1] * Transition[:, j]
            V[j, t] = np.max(trans_prob) * Emission[j, Observation[t]]
            path[t] = np.argmax(trans_prob)
    # Termination step
    P = np.max(V[:, T-1])
    best_last_state = np.argmax(V[:, T-1])
    # Backtrack to find the best path
    best_path = np.zeros(T, dtype=int)
    best_path[T-1] = best_last_state
    for t in range(T-2, -1, -1):
        best_path[t] = path[t+1]
    return best_path.tolist(), P
