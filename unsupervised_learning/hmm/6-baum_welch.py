#!/usr/bin/env python3
"""Baum-Welch Algorithm for Hidden Markov Models"""


import numpy as np


def forward(observations, emission, transition, initial):
    """Performs the forward algorithm for a Hidden Markov Model."""
    T = observations.shape[0]
    N = transition.shape[0]
    F = np.zeros((N, T))
    # Initialize the forward probabilities
    F[:, 0] = initial.T * emission[:, observations[0]]
    # Recursion step
    for t in range(1, T):
        for n in range(N):
            F[n, t] = np.sum(F[:, t-1] * transition[:, n] * emission[n, observations[t]])
    # Total probability of the observations
    P = np.sum(F[:, -1])
    return P, F

def backward(observations, emission, transition, initial):
    """Performs the backward algorithm for a Hidden Markov Model."""
    T = observations.shape[0]
    N = transition.shape[0]
    B = np.zeros((N, T))
    # Initialization step
    B[:, T - 1] = 1
    # Recursion step
    for t in range(T - 2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(transition[i, :] * emission[:, observations[t + 1]] * B[:, t + 1])
    # Total probability of the observations
    P = np.sum(initial[:, 0] * emission[:, observations[0]] * B[:, 0])
    return P, B

def baum_welch(observations, transition, emission, initial, iterations=1000):
    """Performs the Baum-Welch algorithm for a Hidden Markov Model."""
    N = transition.shape[0]
    M = emission.shape[1]
    T = observations.shape[0]
    for _ in range(iterations):
        # E-step: compute forward and backward probabilities
        P, F = forward(observations, emission, transition, initial)
        _, B = backward(observations, emission, transition, initial)
        xi = np.zeros((N, N, T - 1))
        # Calculate xi values
        for t in range(T - 1):
            denominator = np.dot(np.dot(F[:, t].T, transition) * emission[:, observations[t + 1]], B[:, t + 1])
            for i in range(N):
                numerator = F[i, t] * transition[i, :] * emission[:, observations[t + 1]] * B[:, t + 1]
                xi[i, :, t] = numerator / denominator
        # Calculate gamma values
        gamma = np.sum(xi, axis=1)
        # Update transition matrix
        transition = np.sum(xi, axis=2) / np.sum(gamma, axis=1, keepdims=True)
        # Update emission matrix
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0, keepdims=True)))
        denominator = np.sum(gamma, axis=1)
        for i in range(M):
            emission[:, i] = np.sum(gamma[:, observations == i], axis=1)
        emission /= denominator[:, np.newaxis]
    return transition, emission
