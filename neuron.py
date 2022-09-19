import math
import random
import numpy as np

class RandomWeightGenerator:
    def __init__(self, range_min, range_max, precision=None):
        self.min = range_min
        self.max = range_max
        self.precision = precision

    def __call__(self):
        if self.precision is None:
            return random.uniform(self.min, self.max)
        else:
            return round(random.uniform(self.min, self.max), self.precision)

class Neuron:
    def __init__(self, weight_count, activation_function, weight_generator):
        self.weights = [0] * weight_count
        self.bias = 0
        self.activation_function = activation_function
        self.weight_generator = weight_generator
        self.generate_weights()

    def generate_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] = self.weight_generator()
        self.bias = self.weight_generator()
    
    def activate(self, inputs):
        if len(inputs) == len(self.weights):
            output = self.bias
            output += np.dot(self.weights, inputs)
            return self.activation_function(output)
        else:
            raise Exception("Invalid input count")
    
    def __str__(self):
            return f"Weights: {self.weights} Bias: {self.bias}"

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def binary_step(x):
    return 1 if x >= 0 else 0

def round_output(neuron_output, threshold):
    if neuron_output >= threshold:
        return 1
    elif neuron_output < 1-threshold:
        return 0
    else:
        return None

def round_outputs(neuron_output, threshold):
    return list(map(round_output, neuron_output, [threshold] * len(neuron_output)))

