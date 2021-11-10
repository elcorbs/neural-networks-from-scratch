#! python3
import numpy as np
from numpy import array

def calculate_layer(inputs, weights, biases):
    inputs = array(inputs)
    weights = array(weights)
    biases = array(biases)
    return np.matmul(inputs, weights.T) + biases


    