# a simple neural network
# created by alex mounsey

# import modules
import numpy as np

# sigmoid activation function
def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

# sigmoid derivative
def sigDeriv(x):
    return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.w1 = np.random.rand(self.input.shape[1], 4) # first set of weights
        self.w2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    # feedforward function
    def feedForward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.w1))
        self.output = sigmoid(np.dot(self.layer1, self.w2))
    
    # backpropagation function
    def backprop(self):
        # the 'chain rule' which finds the derivation of the loss with respect to the weights
        d_w1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigDeriv(self.output), self.w2.T) * sigDeriv(self.layer1)))
        d_w2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigDeriv(self.output)))

        self.w1 += d_w1
        self.w2 += d_w2

if __name__ == "__main__":
    # training data
    x = np.array([
                [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1]])
    
    # training labels
    y = np.array([[0], [1], [1], [0]])

    neuralNet = NeuralNetwork(x, y)

    # train for 1500 epochs
    for i in range(1500):
        neuralNet.feedForward()
        neuralNet.backprop()
    
    print(neuralNet.output)
    