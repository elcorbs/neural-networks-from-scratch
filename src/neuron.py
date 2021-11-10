#! python3

def calculate_single_neuron_in_layer(inputs, weights, bias):
    return sum([i * w for i,w in zip(inputs, weights)]) + bias


if __name__ == '__main__':
    inputs = [1, 2, 3, 2.5]

    weights = [
        [0.2, 0.8, -0.5, 1],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ]

    biases = [2, 3, 0.5]

    outputs = [calculate_single_neuron_in_layer(inputs, w, b) for w, b in zip(weights, biases)]

    print(outputs)
