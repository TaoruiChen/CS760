# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 09:42:01 2023

@author: taoru
"""

import math
import numpy as np

# Define entropy and gain ratio functions
def entropy(labels):

    if not labels:  # Check if labels list is empty
        return 0

    pos_c = sum(labels)
    neg_c = len(labels) - pos_c
    proba = [float(pos_c) / len(labels), float(neg_c) / len(labels)]

    # Check for zero probabilities to avoid log errors
    if proba[0] == 0 or proba[1] == 0:
        E = 0
    else:
        E = -1 * np.dot(proba, np.log2(proba))

    return E


def gain_ratio(data, feature_index, threshold):
    total_entropy = entropy([point[2] for point in data])

    left = [point[2] for point in data if point[feature_index] >= threshold]
    right = [point[2] for point in data if point[feature_index] < threshold]

    left_entropy = entropy(left)
    right_entropy = entropy(right)

    gain = total_entropy - (len(left) / len(data)) * left_entropy - (len(right) / len(data)) * right_entropy

    split_entropy = entropy([len(left) / len(data), len(right) / len(data)])
    if split_entropy == 0:
        return gain

    return gain / split_entropy


# Load the data from the provided content
data = [
    [0.1, -2, 0],
    [0, -1, 1],
    [0, 0, 0],
    [0, 1, 0],
    [0, 2, 0],
    [0, 3, 0],
    [0, 4, 0],
    [0, 5, 0],
    [0, 6, 1],
    [0, 7, 0],
    [0, 8, 1]
]

#data = np.loadtxt("Druns.txt")



for feature_index in range(2):
    print(f"Feature x{feature_index + 1}:")
    thresholds = sorted(list(set([point[feature_index] for point in data])))
    for threshold in thresholds:
        ratio = gain_ratio(data, feature_index, threshold)
        print(f"Threshold: {threshold}, Gain Ratio: {ratio}")
    print("\n")
    



