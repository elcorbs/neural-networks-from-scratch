from numpy import array
from src.neural_network import calculate_layer 

class TestSingleLayerNetwork:
    def assertListEquals(self, actual, expected):
        assert len(actual) == len(expected)
        assert all([a == b for a, b in zip(actual, expected)])

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
        self.assertListEquals(calculate_layer(inputs, weights, biases), expected_output)





