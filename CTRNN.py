import numpy as np

class CTRNN():
    def __init__(self, size):
        self.size = size                        
        self.states = np.zeros(size)            
        self.timeConstants = np.random.uniform(0.1,5.0,size=(self.size))    
        self.invTimeConstants = 1.0/self.timeConstants
        self.biases = np.random.uniform(-10,10,size=(self.size))        
        self.weights = np.random.uniform(-10,10,size=(self.size,self.size))
        self.inputs = np.zeros(size) 
        self.outputs = self.sigmoid(self.states+self.biases)     
        # print("weights: ", self.weights)  
        # print("biases: ", self.biases)   
        # print("time: ", self.timeConstants)  
        # print("indv time: ", self.invTimeConstants)
                   
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def step(self, dt, inputs = None):
        if inputs is not None:
            self.inputs = inputs
        currentInput = self.inputs + np.dot(self.weights.T, self.outputs)
        self.states += dt * (self.invTimeConstants*(-self.states+currentInput))
        self.outputs = self.sigmoid(self.states+self.biases)
        #self.outputs = np.tanh(self.states+self.biases)
        return self.outputs
    
    def intialize_from_genetic_algorithm(self, genotype):
        count = 0
        for weight in range(len(self.weights)):
            self.weights[weight] = genotype[count]
            count += 1
        for bias in range(len(self.biases)):
            self.biases[bias] = genotype[count]
            count += 1
        for timeConst in range(len(self.timeConstants)):
            self.timeConstants[timeConst] = genotype[count]
            count += 1
        self.invTimeConstants = 1.0/self.timeConstants

        
    
    def save(self, filename):
        np.savez(filename, size=self.size, weights=self.weights, biases=self.biases, timeconstants=self.timeConstants)

    def load(self, filename):
        params = np.load(filename)
        self.size = params['size']
        self.weights = params['weights']
        self.biases = params['biases']
        self.timeConstants = params['timeconstants']
        self.invTimeConstants = 1.0/self.timeConstants