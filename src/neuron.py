#! python3
import numpy as np

def calculate_single_neuron_in_layer(inputs, weights, bias):
    return np.dot(inputs, weights) + bias


if __name__ == '__main__':
    # i
    inputs = [1, 2, 3, 2.5]

    # W
    weights = [
        [0.2, 0.8, -0.5, 1],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ]

    # b
    biases = [2, 3, 0.5]

    # W i^T + b
    outputs = np.matmul(weights, inputs) + biases

    print(outputs)
