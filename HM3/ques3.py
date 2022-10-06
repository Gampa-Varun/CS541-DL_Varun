import numpy as np 
import math
import matplotlib.pyplot as plt
import time
import csv
import json
import numba as nb
import time
from concurrent.futures import  ThreadPoolExecutor
import copy


#computing the feedforward output yhat for a given set of parameters and examples
@nb.jit(nopython=True,cache=True,nogil=True,parallel=True,fastmath=True)
def softmax(x,W,b):


	Z = np.dot(x,W) + b

	den = (np.sum(np.exp(Z),axis=1))
	
	num = np.transpose(np.exp(Z))

	probability = num/den

	return probability


#Computing  the gradient of the cost function, 
#note that in the cost function bias is not regularised and the same suit is followed in the gradient as well
@nb.jit(nopython=True,cache=True,nogil=True,parallel=True,fastmath=True)
def grad_cost_function (X, y, W, b, alpha,gradW,gradb) :

	

	y_hat = softmax(X,W,b)

	n = np.int32(np.shape(y)[0])

	yt = np.transpose(y)

	Xt = np.transpose(X)

	diff = yt - y_hat

	for i in nb.prange(np.shape(W)[1]):
		gradW[:,i] = -np.sum(Xt*diff[i,:],axis=1)/n

	gradW = gradW + alpha*W

	#gradW = -np.einsum('ij,kj->ki',diff,X)/n + alpha*W // slower method, though more concise

	gradb = -np.sum((diff),axis=1)/n




	return gradW,gradb

#The weight vector and bias are updated based on learning rate and the gradient 
@nb.jit(nopython=True,cache=True,nogil=True,fastmath=True)
def parameter_update (X, y, W, b, epsilon, alpha) :

	gradW = np.zeros(np.shape(W),dtype='float32')
	gradb = np.zeros((np.shape(y)[1]),dtype='float32')

	gradientW, gradientb = grad_cost_function(X,y,W,b, alpha,gradW,gradb)

	W_new = (W - epsilon*gradientW)

	b_new = (b - epsilon*gradientb)

	return W_new,b_new




#MSE Cost function with regularization except on bias term 
@nb.jit(nopython=True,cache=True,nogil=True,parallel=True)
def cost_function (X, y, W, b, alpha) :

	n = (np.shape(y)[0])

	y_hat = softmax(X,W,b)	

	cost = -np.sum(y.T*np.log(y_hat))/n  + alpha*np.sum(np.square(W))


	return cost




def one_hotencoding(y):

	yencode = np.zeros((np.max(y)+1,np.shape(y)[0]))

	columns = np.arange(len(y))

	yencode[y,columns] = 1;

	return yencode

@nb.jit(nopython=True,nogil=True,fastmath=True,cache=True)
def train(epsilon,alpha,total_epochs,mini_batch_size,X_tr,ytr,X_val,y_val):

	np.random.seed(20)

	W = np.random.normal(loc = 0.0, scale = 1.0, size = (np.shape(X_tr)[1],np.shape(ytr)[1])).astype('float32')

	b = np.random.normal(loc = 0.0, scale = 1.0, size = (np.shape(ytr)[1])).astype('float32')
	num_examples = np.shape(X_tr)[0]

	for epochs in range(total_epochs) :

		#Set random seed
		np.random.seed(epochs)

		indices = np.random.permutation(np.arange(num_examples))
		X_tr_shuf = X_tr[indices,...]	
		ytr_shuf = ytr[indices,...]	



		for minibatch in range(math.floor(np.shape(X_tr)[0]/mini_batch_size)) :


			X_minibatch = X_tr_shuf[(minibatch)*mini_batch_size:(minibatch+1)*mini_batch_size,...]
			y_minibatch = ytr_shuf[(minibatch)*mini_batch_size:(minibatch+1)*mini_batch_size,...]

			W,b = parameter_update(X_minibatch,y_minibatch,W,b,epsilon, alpha)

			

		costfn = cost_function(X_tr,ytr,W,b, alpha)
		#print(epochs)

	valcostfn = cost_function(X_val,y_val,W,b, alpha)
	
	hyper_set = np.asarray([valcostfn ,alpha,epsilon,total_epochs,mini_batch_size],dtype='float32')

	return W,b,hyper_set


@nb.jit(nopython=True,cache=True,nogil=True,fastmath=True,parallel=True)
def learning(epsilon_set,alpha_set,epoch_lengths,mini_batch_sizes,X_tr,ytr,X_val,y_val):

	num_params = len(epsilon_set)*len(alpha_set)*len(epoch_lengths)*len(mini_batch_sizes)
	W_out = np.zeros((np.shape(X_tr)[1],np.shape(ytr)[1]), dtype='float32')
	b_out = np.zeros((np.shape(ytr)[1]), dtype='float32') 
	params = np.zeros((num_params,5),dtype='float32')
	weight = np.zeros((num_params,np.shape(X_tr)[1],np.shape(ytr)[1]), dtype='float32')
	bias = np.zeros((num_params,np.shape(ytr)[1]), dtype='float32')
	alpha_best = 0
	epsilon_best = 0
	epoch_best = 0
	batch_size_best = 0
	valcostfn_min = 10000

	for arg_epsilon in nb.prange(len(epsilon_set)):

		for arg_alpha in range(len(alpha_set)):

			for arg_total_epochs in range(len(epoch_lengths)):

				for arg_mini_batch_size in range(len(mini_batch_sizes)):

					
					index = arg_epsilon*8 + 4*arg_alpha + 2*arg_total_epochs + arg_mini_batch_size	

					mini_batch_size = mini_batch_sizes[math.floor(index%2)]
					total_epochs = epoch_lengths[math.floor((index%4)/2)]
					alpha = alpha_set[math.floor(((index%8)/4))]
					epsilon = epsilon_set[math.floor(index/8)]




					weight[index,...],bias[index,...],params[index,...] = train(epsilon,alpha,total_epochs,mini_batch_size,X_tr,ytr,X_val,y_val)


	for arg_epsilon in range(len(epsilon_set)):

		for arg_alpha in range(len(alpha_set)):

			for arg_total_epochs in range(len(epoch_lengths)):

				for arg_mini_batch_size in range(len(mini_batch_sizes)):		
				
					index = arg_epsilon*8 + 4*arg_alpha + 2*arg_total_epochs + arg_mini_batch_size			

					if params[index,0]< valcostfn_min:
						W_out = weight[index,...]
						b_out = bias[index,...]
						epsilon_best = params[index,1]
						alpha_best = params[index,2]
						epoch_best = params[index,3]
						batch_size_best = params[index,4]
						valcostfn_min = params[index,0]

	return W_out,b_out,alpha_best,epsilon_best,epoch_best,batch_size_best



def train_regressor ():

	# Load data

	X_tr = np.load("fashion_mnist_train_images.npy").astype('float32')

	X_tr = X_tr/255
	ytr = (np.load("fashion_mnist_train_labels.npy")).astype(int)

	#Perform one hot encoding
	ytr = one_hotencoding(ytr).T.astype('float32')

	#Seperating the validation set after permutating the examples

	#Set random seed
	np.random.seed(1)

	indices = np.random.permutation(range(np.shape(X_tr)[0]))
	X_tr = X_tr[indices,:]	
	ytr = ytr[indices,:]


	num_ex_val = math.floor(0.2*np.shape(X_tr)[0])

	X_val = X_tr[-num_ex_val:,...].astype('float32')
	X_tr = X_tr[0:-num_ex_val,...].astype('float32')

	y_val = ytr[-num_ex_val:,...]
	ytr = ytr[0:-num_ex_val,...]

	#print(np.shape(X_tr))

 	# Hyparameter sets

	epsilon_set = np.asarray([0.01,0.05], dtype='float32')

	alpha_set = np.asarray([0.001,0.01], dtype='float32')

	mini_batch_sizes = np.asarray([100,500], dtype='int') 

	epoch_lengths = np.asarray([50,100], dtype='int')



	start = time.time()

	W,b,alpha_best,epsilon_best,epoch_best,batch_size_best = learning(epsilon_set,alpha_set,epoch_lengths,mini_batch_sizes,X_tr,ytr,X_val,y_val)			

	end = time.time()


	print("time",start-end)
									
	return W,b,alpha_best,epsilon_best,epoch_best,batch_size_best



if __name__ == '__main__':

	W,b,alpha_best,epsilon_best,epoch_best,batch_size_best = train_regressor()

	print('Optimal params are -  epochs of', epoch_best," and mini_batch_size of ", batch_size_best, " with alpha of ",alpha_best, "and epsilon of", epsilon_best)

	X_te = np.load("fashion_mnist_test_images.npy").astype('float32')

	X_te = X_te/255
	yte = (np.load("fashion_mnist_test_labels.npy")).astype('int')

	#Perform one hot encoding
	yte = one_hotencoding(yte).T.astype('float32')


	#Find error on test set with optimal hyperparameter set and corresponding weights

	test_set_cost = cost_function(X_te,yte,W,b, 0)
	print("Test set cost: ",test_set_cost)


	#Find the accuracy

	estimate_full = softmax(X_te,W,b);
	yte = yte.T.astype(int)

	print(np.shape(yte))


	count = 0
	for i in range (np.shape(yte)[1]):
		if np.argmax(yte[:,i]) == np.argmax(estimate_full[:,i]):
			count = count + 1

	print("Accuracy : ", 100*count/np.shape(yte)[1])



