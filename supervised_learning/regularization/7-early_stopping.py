#!/usr/bin/env python3
""" Early Stopping """


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if gradient descent should be stopped early.

    Parameters:
        cost (float): The current validation cost of the NN.
        opt_cost (float): The lowest recorded validation cost of the NN.
        threshold (float): The threshold used for ES.
        patience (int): The patience count used for ES.
        count (int): The count of how long the threshold has not been met.

    Returns:
    Tuple[bool, int]: A boolean indicating to stop gradient descent early,
                     followed by the updated count.
    """
    if cost - opt_cost <= threshold:
        count += 1
    else:
        count = 0

    if count < patience:
        return False, count
    else:
        return True, count
