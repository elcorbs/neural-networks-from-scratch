from numpy import array
from src.neural_network import calculate_layer 
from test.helpers import assertions 

class TestSingleLayerNetwork:

    def test_single_neuron_single_sample(self):
        inputs = [1.0, 2.0, 3.0, 2.5]
        weights = [0.2, 0.8, -0.5, 1.0]
        bias = 2.0
        expected_ouptut = [4.8]
        assert (calculate_layer(inputs, weights, [bias]) == expected_ouptut)

    def test_three_neurons_single_sample(self):
        inputs = [1.0, 2.0, 3.0, 2.5]
        weights = [[0.2, 0.8, -0.5, 1],
                [0.5, -0.91, 0.26, -0.5],
                [-0.26, -0.27, 0.17, 0.87]]
        biases = [2.0, 3.0, 0.5]
        expected_output = [4.8, 1.21, 2.385]
        assertions.listEquals(calculate_layer(inputs, weights, biases), expected_output)

    def test_three_neurons_three_samples(self):
        inputs = array([
            [1, 2, 3, 2.5],
            [2, 5, -1, 2],
            [-1.5, 2.7, 3.3, -0.8]
        ])
        weights = array([
            [0.2, 0.8, -0.5, 1],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]
        ])
        biases = array([2, 3, 0.5])
        expected_output = array([
            [ 4.8, 1.21, 2.385],
            [ 8.9,   -1.81,   0.2],
            [ 1.41,   1.051,  0.026]
        ])
        assertions.arrayEqual(calculate_layer(inputs, weights, biases), expected_output)




