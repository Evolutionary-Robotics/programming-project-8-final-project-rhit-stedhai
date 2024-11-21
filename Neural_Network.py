import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, layers, weightsRange, biasRange):
        self.layers = layers
        self.numLayers = len(layers)
        self.weightsRange = weightsRange
        self.biasRange = biasRange

    def sigmoid(self, input):
         return 1/(1+np.exp(-input))
    
    def intialize_from_genetic_algorithm(self, genotype):
        self.weights = []
        start = 0
        for layer in range(self.numLayers-1):
            end = start + self.layers[layer]*self.layers[layer+1]
            genes = genotype[start:end]
            for gene in range(len(genes)):
                genes[gene] *= self.weightsRange
            self.weights.append(np.reshape(genes, (self.layers[layer],self.layers[layer+1])))
            start = end
        self.biases = []
        for layer in np.arange(self.numLayers-1):
            end = start + self.layers[layer+1]
            genes = genotype[start:end]
            for gene in range(len(genes)):
                genes[gene] *= self.biasRange
            self.biases.append(np.reshape(genes, (1,self.layers[layer+1])))
            start = end

    def forward(self, inputs):
        state = inputs
        for layer in range(self.numLayers-1):
            state = self.sigmoid(np.dot(state, self.weights[layer]) + self.biases[layer])
        return state