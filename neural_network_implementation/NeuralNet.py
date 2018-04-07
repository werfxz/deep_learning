import numpy as np
class NeuralNet:

    def __init__(self, input_size, hidden_size, num_classes, reg, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.reg = reg
        self.learning_rate = learning_rate

    def initializeWeights(self, input_size, hidden_size, num_classes):
        # Initializing random weights w0from sklearn.datasets import fetch_mldata
        #there are hidden_size*input_size number of weights in the first layer
        # Initializing random weights w1
        #there are hidden_size*num_classes number of weights in the first layer
       
        w1 = np.random.rand(hidden_size, input_size) * 0.01
        b1 = np.zeros(shape=(hidden_size, 1))
        w2 = np.random.rand(num_classes, hidden_size) * 0.01
        b2 = np.zeros(shape=(num_classes, 1))
        parameters = {"w1":w1,
                      "b1":b1,
                      "w2":w2, 
                      "b2":b2}
        return parameters
    
    def calculateL2(self, w1, w2):
        # L2 Regularization loss
        L2 = np.sum(np.square(w1)) + np.sum(np.square(w2))
        return L2
        
    def relu(self, neurons):
        # determines the activations of a ReLU when input is given
        return np.maximum(neurons, 0)

    def relu_derivative(self, neurons):
        #calculates the derivate of relu activation
        return np.multiply(1*(neurons > 0),neurons)
        
    def softmax(self, neurons):
         # computes normalized probabilities 
        class_probs = np.exp(neurons) / np.sum(np.exp(neurons), axis = 0)
        
        return class_probs
        
    def forwardPass(self, X, parameters):
        #forward propagation and loss computation
        w1 = parameters['w1']
        b1 = parameters['b1']
        w2 = parameters['w2']
        b2 = parameters['b2']
        
        z1 = np.dot(w1, X) + b1
        a1 = self.relu(z1)
        z2 = np.dot(w2, a1) + b2
        class_probs = self.softmax(z2)
#        print(class_probs.shape, z2.shape)
        cache = {"z1":z1,
                 "a1":a1,
                 "z2":z2,
                 "a2":class_probs}
        
        return class_probs, cache
    
    def computeCost(self, class_probs, Y):
        m = Y.shape[1]  
        cost = -1 * np.sum(np.multiply(np.log(class_probs), Y)) / m
        
        return cost
    
    def predict(self, X, w1, w2, b1, b2, threshold):
        # Given some data X, predict the class per each sample
        class_probs, cache = self.forwardPass(X, w1, w2, b1, b2)
        #convert class probabilities to 0 or 1 by threshold
        class_probs = (class_probs > threshold)
       
        return class_probs
        
    def backProp(self, parameters, cache, X, Y):    
        #backpropagate the loss
        m = X.shape[1]
        a1 = cache['a1']
        a2 = cache['a2']
        w2 = parameters['w2']
        
        dz2 = a2 - Y
        dw2 = (1 / m) * np.dot(dz2, a1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.multiply(np.dot(w2.T, dz2), self.relu_derivative(a1))
        dw1 = (1 / m) * np.dot(dz1, X.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
        
        grads = {"dw1": dw1,
                 "db1": db1,
                 "dw2": dw2,
                 "db2": db2} 
        
        return grads
    
    def update_parameters(self, parameters, grads, learning_rate):
        # retrieve each parameters that should be optimized
        w1 = parameters['w1']
        b1 = parameters['b1']
        w2 = parameters['w2']
        b2 = parameters['b2']
        # retrieve each gradients to updatee the parameters
        dw1 = grads['dw1']
        db1 = grads['db1']
        dw2 = grads['dw2']
        db2 = grads['db2']
        #update each parameter
        w1 = w1 - learning_rate * dw1
        b1 = b1 - learning_rate * db1
        w2 = w2 - learning_rate * dw2
        b2 = b2 - learning_rate * db2
        #save updated parameters to dictionary
        parameters = {"w1": w1,
              "b1": b1,
              "w2": w2,
              "b2": b2}
        return parameters
    
    def train(self, X_train, Y_train, num_epoch, batch_size):
        parameters = self.initializeWeights(self.input_size, self.hidden_size, self.num_classes)
        m = X_train.shape[1]
        
        for j in range(num_epoch):
            for i in range(int(m/batch_size)):
                #forwardPass
                class_probs, cache = self.forwardPass(X_train[:,(i*batch_size)+1:(i+1)*batch_size], parameters)
                #computeCost
                cost = self.computeCost(class_probs, Y_train[:,(i*batch_size)+1:(i+1)*batch_size])
                #backProp
                grads = self.backProp(parameters, cache, X_train[:,(i*batch_size)+1:(i+1)*batch_size], Y_train[:,(i*batch_size)+1:(i+1)*batch_size])
                #weightUpdate
                parameters = self.update_parameters(parameters, grads, self.learning_rate)
               
                np.set_printoptions(suppress=True)
                
            errorbyepoch = cost
            print(errorbyepoch)
            
        return errorbyepoch
