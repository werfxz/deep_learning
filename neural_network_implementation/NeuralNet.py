import numpy as np
class NeuralNet:

	def __init__(self, input_size, hidden_size, num_classes, reg, lr):
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_classes = num_classes
		self.reg = reg
		self.lr = lr

	def initializeWeights(self, mu=0):
		# Initializing random weights w0from sklearn.datasets import fetch_mldata
		#there are hidden_size*input_size number of weights in the first layer
		# Initializing random weights w1
		#there are hidden_size*num_classes number of weights in the first layer
        w0 = np.random.rand(hidden_size, input_size)
        w1 = np.random.rand(num_classes, hidden_size)
        return w0, w1
    
	def calculateL2(self, w0, w1):
		# L2 Regularization loss
        L2 = np.sum(np.square(wo)) + np.sum(np.square(w1))
        
	def relu(self, neurons):
	     # determines the activations of a ReLU when input is given
        activation_neurons = np.maximum(neurons, 0)
        
	def softmax(self, neurons, y):
	     # computes normalized probabilities and loss when labels(y) are given
        class_probs = np.exp(neurons) / np.sum(np.exp(neurons))
        loss = -1 * np.sum(np.log(np.multiply(neurons, y)))
        return class_probs, loss
        
	def forwardPass(self, X, w0, w1, b0, b1, y=None, training=True):
		#forward propagation and loss computation
        z1 = np.dot(X, w0) + b0
        a1 = relu(z1)
        z2 = np.dot(a1, w1) + b1
        class_probs, loss = self.softmax(z2, y) # y yi burda nasıl vercez? 
        
	def predict(self, X):
		# Given some data X, predict the class per each sample
    
	def backProp(self, li, labels):    
		#backpropagate the loss
		#compute weight updates
		return dw0, dw1, db0, db1

	def train(self, X_train, y_train, num_epoch, batch_size):
		for j in range(num_epoch):
			for i in range(num_batch):
				#forwardPass
             #backProp
             #weightUpdate
		return errorbyepoch
