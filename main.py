import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v2 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward, load_data


np.random.seed(1)

def initialize_parameters_deep(layer_dims):
	np.random.seed(3)

	parameters = {}
	L = len(layer_dims)

	for l in range(1, L):
		parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
		parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

		assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
		assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

	return parameters


def linear_forward(A, W, b):
	Z = np.dot(W, A) + b
	cache = (A, W, b)
	return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
	
	Z, linear_cache = linear_forward(A_prev, W, b)
	if activation == "sigmoid":
		A, activation_cache = sigmoid(Z)
	elif activation == "relu":
		A, activation_cache = relu(Z)
	cache = (linear_cache, activation_cache)

	return A, cache

def L_model_forward(X, parameters):
	caches = []			# caches will finally contain Z, A, W, b, A, Z ?
	A = X
	L = len(parameters) // 2		# No of layers
	print(L)
	for l in range(1, L):
		A_prev = A
		W = parameters['W' + str(l)]
		b = parameters['b' + str(l)]
		A, cache = linear_activation_forward(A_prev, W, b, activation = "relu")
		caches.append(cache)

	W = parameters['W' + str(L)]
	b = parameters['b' + str(L)]
	AL, cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
	caches.append(cache)

	assert(AL.shape == (1, X.shape[1]))
	return AL, caches

def compute_cost():
	pass




def linear_backward(dZ, cache):
	A_prev, W, b = cache
	m = A_prev.shape[1]


	dW = (1.0 / m) * np.dot(dZ, cache[0].T)
	db = (1.0 / m) * np.sum(dZ, axis = 1, keepdims = True)
	dA_prev = np.dot(cache[1].T, dZ)

	return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
	linear_cache, activation_cache = cache

	if activation == "relu":
		dZ = relu_backward(dA, activation_cache)
	elif activation == "sigmoid":
		dZ = sigmoid_backward(dA, activation_cache)

	dA_prev, dW, db = linear_backward(dZ, linear_cache)

	return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
	# Y has true label vector -> 0 for non-cat, 1 for cat
	grads = {}
	L = len(caches)		# number of layers
	m = AL.shape[1]
	Y = Y.reshape(AL.shape)

	dAL = -np.divide(Y, AL) + np.divide(1 - Y, 1 - AL)

	grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")

	for l in reversed(rangee(L-1)):
		current_cache = caches[l]

		dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+2)], current_cache, activation = "relu")
		grads["dA" + str(l+1)] = dA_prev_temp
		grads["dW" + str(l+1)] = dW_temp
		grads["db" + str(l+1)] = db_temp

	return grads

def update_parameters(parameters, grads, learning_rate):
	L = len(parameters) // 2
	for l in range(L):
		parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
		parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

	return parameters


def predict(X, y, parameters):
	m = X.shape[1]
	n = len(parameters) // 2
	p = np.zeros(1, m)
	probas, caches = L_model_forward(X, parameters)

	for i in range(0, probas.shape[1]):
		if(probas[0, i] > 0.5):
			p[0,i] = i
		else:
			p[0, i] = 0

	print("Accuracy: " + str(np.sum((p == y)/m)))
	return p

if __name__ == '__main__':

	train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
	layers_dims = [12288, 20, 7, 5, 1]

	m_train = train_x_orig.shape[0]
	num_px = train_x_orig.shape[1]
	m_test = test_x_orig.shape[0]

	num_iterations = 2500

	train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
	test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
	train_x = train_x_flatten/255.
	test_x = test_x_flatten/255.

	np.random.seed(1)
	parameters = initialize_parameters_deep(layers_dims)
	print(len(parameters))

	for i in range(0, num_iterations):
		AL, caches = L_model_forward(train_x, parameters)

		grads = L_model_backward(AL, train_y, caches)

		parameters =  update_parameters(parameters, grads, learning_rate = learning_rate)

	pred_train = predict(train_x, train_y, parameters)
	pred_test = predict(test_x, test_y, parameters)

	
