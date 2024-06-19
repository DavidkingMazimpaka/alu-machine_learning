#!/usr/env/bin python3
""" Confusing Matrix""" 

import numpy as np


def create_confusion_matrix(labels, logits):
    """ Confusing Matrix"""
    true_classes = np.argmax(labels, axis=1)
    predicted_classes = np.argmax(logits, axis=1)
    classes = labels.shape[1]
    confusion_matrix = np.zeros((classes, classes), dtype=int)
    # populate confusion matrix
    for true, pred in zip(true_classes, predicted_classes):
        confusion_matrix[true][pred] += 1
    return confusion_matrix
