#!/usr/bin/env python3
'''
    Script that defines a function def bi_rnn(bi_cell, X, h_0, h_T):
    that performs forward propagation for a bidirectional RNN:
'''

import numpy as np

def bi_rnn(bi_cell, X, h_0, h_T):
    '''
        Function that performs forward propagation for a bidirectional RNN

        parameters:
            bi_cell: an instance of BidirectionalCell
            X: data
            h_0: initial hidden state
            h_T: terminal hidden state

        return:
            H: all hidden states
            Y: all outputs
    '''

    t, m, i = X.shape
    
    # Check the shape of h_0
    if len(h_0.shape) == 2:
        # If h_0 is 2D, we need to add an additional dimension
        h_0 = h_0[np.newaxis, :, :]  # Shape becomes (1, m, h)
    
    l, m, h = h_0.shape
    H = np.zeros((t + 1, 2, m, h))
    H[0, 0] = h_0[0]  # Use the first dimension of h_0
    H[0, 1] = h_T

    Y = None  # Initialize Y

    for step in range(t):
        h_prev, y = bi_cell.forward(H[step, 0], X[step])
        H[step + 1, 0] = h_prev
        h_next, y = bi_cell.forward(H[step, 1], y)
        H[step + 1, 1] = h_next
        
        if step == 0:
            Y = y
        else:
            Y = np.concatenate((Y, y))

    output_shape = Y.shape[-1]
    Y = Y.reshape(t, 2, m, output_shape)
    
    return (H, Y)
