import numpy as np

## this class is the setup for each of the Neuron inside the layers
class NeuronLayers():
    def __init__(self, number_of_neurons, number_of_inputs):
        self.weight = 2 * np.random((number_of_inputs, number_of_neurons)) - 1
        ## ?? do not understand about the double () of random

class NeuralNetwork():

    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, e_poch):
        for i in xrange(e_poch):
            
            output_from_layer1, output_from_layer2 = self.think(training_set_inputs)

            layer2_error = training_set_outputs - output_from_layer2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer2)

            layer1_error = layer2_delta.np.dot(self.layer2.weight.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer1)

            layer1_ajustment = training_set_inputs.T.dot(layer1_delta)
            layer2_ajustment = output_from_layer1.T.dot(layer2_delta)

            self.layer1.weight += layer1_ajustment
            self.layer2.weight += layer2_ajustment



    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(np.dot(inputs, self.layer1.weight))
        output_from_layer2 = self.__sigmoid(np.dot(output_from_layer1, self.layer2.weight))
        return output_from_layer1, output_from_layer2

    
