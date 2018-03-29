from mnistdata import mnist
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing
#from NeuralNet import NeuralNet

def readData(mode="training"):
	# Reading Data
    	foldername = os.path.join(os.path.realpath(''), 'mnistdata/data')
    
    	mndata = mnist.MNIST(foldername)
    	if mode == "training":
    		images, labels = mndata.load_training()
    	elif mode == "test":
    		images, labels = mndata.load_testing()
    	else:
    		print (" Data not read.")
    		images, labels = None
    	print ("MNIST handwritten digit data is read.")
    	label_binarizer = sklearn.preprocessing.LabelBinarizer()
    	label_binarizer.fit(range(max(labels)+1))
    	labels = label_binarizer.transform(np.array(labels)).T
    	return np.array(images), np.array(labels)


X_train, y_train = readData(mode="training")
X_test, y_test = readData(mode="test")
#number of neurons in each layer
input_size = 28 * 28 * 1
hidden_size = 30
num_classes = 10
#reg:regularization strength 
#lr:learning rate 
reg=0.001
lr=1e-3
#batch_size is the number of samples in each batch
#num_epoch denotes how many times training goes over whole training set
num_epoch=10
batch_size=100
# Initialize the network
net = NeuralNet(input_size, hidden_size, num_classes, reg, lr)
# Train the network using batches of data.
# stats keeps the amount of error at each epoch
errorbyepoch = net.train(X_train, y_train, num_epoch, batch_size)
# Predict on the test
train_acc = (net.predict(X_train) == y_train).mean()
print ('Train accuracy: ', train_acc)
test_acc = (net.predict(X_test) == y_test).mean()
print ('Test accuracy: ', test_acc)

# Plot the points using matplotlib
x = np.arange(0, num_epoch, 1)
plt.plot(x, errorbyepoch)
 
#lri=10**np.random.uniform(-1,-5,10)
#regi=10**np.random.uniform(-1,-5,10)
#trials = list(range(5)) 
#for count in trials:
#	net = NeuralNet(input_size, hidden_size, num_classes, regi[count], lri[count])
#	errorbyepoch = net.train(X_train, y_train, num_epoch, batch_size)
#	test_acc = (net.predict(X_test) == y_test).mean()
#	print ('Test accuracy: ', count, ' ', test_acc)