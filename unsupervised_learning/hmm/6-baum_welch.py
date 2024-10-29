#!/usr/bin/env python3
"""Baum-Welch algorithm implementation"""


import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Forward algorithm implementation
    """
    N = Transition.shape[0]  # Number of hidden states
    T = Observation.shape[0]  # Length of observation sequence
    F = np.zeros((N, T))
    # Initialize first column of forward matrix
    F[:, 0] = Initial.flatten() * Emission[:, Observation[0]]
    # Forward algorithm iterations
    for t in range(1, T):
        for n in range(N):
            F[n, t] = np.sum(F[:, t-1] * Transition[:, n]) * Emission[n, Observation[t]]
    return F
def backward(Observation, Emission, Transition, Initial):
    """Backward algorithm implementation
    """
    N = Transition.shape[0]  
    T = Observation.shape[0]  
    beta = np.zeros((N, T))
    beta[:, T-1] = 1
    for t in range(T-2, -1, -1):
        for n in range(N):
            beta[n, t] = np.sum(
                Transition[n, :] * Emission[:, Observation[t+1]] * beta[:, t+1]
            )
    return beta
def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Performs the Baum-Welch algorithm
    """
    if not isinstance(Observations, np.ndarray) or len(Observations.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.shape[0] != Transition.shape[1]:
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.shape[0] != Transition.shape[0]:
        return None, None
    N, M = Emission.shape
    T = Observations.shape[0]
    for _ in range(iterations):
        # Forward-Backward algorithm
        alpha = forward(Observations, Emission, Transition, Initial)
        beta = backward(Observations, Emission, Transition, Initial)
        # Computing xi and gamma
        xi = np.zeros((N, N, T-1))
        for t in range(T-1):
            denominator = np.sum(alpha[:, t].reshape(-1, 1) * Transition *
                               Emission[:, Observations[t+1]].reshape(1, -1) *
                               beta[:, t+1].reshape(1, -1))
            for i in range(N):
                numerator = alpha[i, t] * Transition[i, :] * \
                           Emission[:, Observations[t+1]] * beta[:, t+1]
                xi[i, :, t] = numerator / denominator
        # Update gamma
        gamma = np.sum(xi, axis=1)
        gamma = np.hstack((gamma, 
                          np.sum(xi[:, :, T-2], axis=0).reshape(-1, 1)))
        # Update parameters
        # Transition matrix update
        Transition = np.sum(xi, axis=2) / np.sum(gamma, axis=1).reshape(-1, 1)
        # Emission matrix update
        denominator = np.sum(gamma, axis=1)
        for s in range(M):
            Emission[:, s] = np.sum(gamma[:, Observations == s], axis=1)
        Emission = Emission / denominator.reshape(-1, 1)
        # Ensure probabilities sum to 1
        Transition = Transition / np.sum(Transition, axis=1, keepdims=True)
        Emission = Emission / np.sum(Emission, axis=1, keepdims=True)
    return Transition, Emission
