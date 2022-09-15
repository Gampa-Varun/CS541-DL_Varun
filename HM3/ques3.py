import numpy as np 
import math
import matplotlib.pyplot as plt
import time
import csv
import json


def dictionary_minimum (dictionary):

	minvalue = float('inf')
	k = []

	for entry in dictionary.keys():
		if dictionary[entry][0] < minvalue:
			key = entry
			minvalue = dictionary[entry][0]
			
	return key



def softmax(x,W,b):

	Z = x.T@W + b
	den = np.atleast_2d(np.sum(np.exp(Z),axis=1))
	
	num = np.exp(x.T@W+b.T).T

	probability = num/den

	return probability


#Computing  the gradient of the cost function, 
#note that in the cost function bias is not regularised and the same suit is followed in the gradient as well
def grad_cost_function (X, y, W, b, alpha) :

	y_hat = softmax(X,W,b)

	n = np.shape(y)[1]

	diff = y - y_hat

	gradW = np.zeros(np.shape(W))

	for i in range(np.shape(W)[1]):
		gradW[:,i] = -np.sum(X*diff[i,:],axis=1)/n

	gradW = gradW + alpha*W

	#gradW = -np.einsum('ij,kj->ki',diff,X)/n + alpha*W // slower method, though more concise

	gradb = -np.sum((y-y_hat),axis=1)/n



	return gradW,gradb

#The weight vector and bias are updated based on learning rate and the gradient 
def parameter_update (X, y, W, b, epsilon, alpha) :

	gradientW, gradientb = grad_cost_function(X,y,W,b, alpha)

	W_new = W - epsilon*gradientW

	b_new = b - epsilon*gradientb

	return W_new,b_new




#MSE Cost function with regularization except on bias term 
def cost_function (X, y, W, b, alpha) :

	n = np.shape(y)[1]

	y_hat = softmax(X,W,b)	

	cost = -np.sum(y*np.log(y_hat))/n  + alpha*np.sum(np.einsum('ij,ij->j',W,W))


	return cost




def one_hotencoding(y):

	yencode = np.zeros((np.max(y)+1,len(y)))

	columns = np.arange(len(y))

	yencode[y,columns] = 1;

	return yencode



def train_regressor ():

	

	# Load data

	X_tr = np.load("fashion_mnist_train_images.npy").T

	X_tr = X_tr/255
	ytr = (np.load("fashion_mnist_train_labels.npy"))

	#Perform one hot encoding
	ytr = one_hotencoding(ytr)

	#Seperating the validation set after permutating the examples

	#Set random seed
	np.random.seed(seed=1)

	indices = np.random.permutation(range(np.shape(X_tr)[1]))
	X_tr = X_tr[:,indices]	
	ytr = ytr[:,indices]


	num_ex_val = math.floor(0.2*np.shape(X_tr)[1])

	X_val = X_tr[:,-num_ex_val:]
	X_tr = X_tr[:,0:-num_ex_val]

	y_val = ytr[:,-num_ex_val:]
	ytr = ytr[:,0:-num_ex_val]

 	# Hyparameter sets

	epsilon_set = [0.1]

	alpha_set = [0.0001]

	mini_batch_sizes = [100] 

	epoch_lengths = [100]

	#####################
	##Set up the graphs##
	#####################
	plt.ion()
	figure, ax = plt.subplots(figsize=(10, 8))
	 
	plt.xlabel("Epochs")
	plt.ylabel("Cost")
	

	background = figure.canvas.copy_from_bbox(ax.bbox)

	plt.show(block = False)

	###################### 
	### Training starts###
	######################

	# Iterating over the hyperparameters

	Costs_wrt_hyp = {}

	for epsilon in epsilon_set:

		for alpha in alpha_set:

			for total_epochs in epoch_lengths:

				for mini_batch_size in mini_batch_sizes:

					#initialize weights
					#Set random seed

					np.random.seed(seed=20)

					W = np.random.normal(loc = 0.0, scale = 1.0, size = [np.shape(X_tr)[0],np.shape(ytr)[0]])

					b = np.random.normal(loc = 0.0, scale = 1.0, size = [np.shape(ytr)[0]])


					costfn = cost_function(X_tr,ytr,W,b,alpha)


					check = grad_cost_function(X_tr, ytr, W, b, alpha)

					costfn_set = [costfn]

					# Set x axis of graph and title
					x = [0]
					plt.title(str("Epochs : "+ str(total_epochs) + ", mini batch size : " + str(mini_batch_size) + ", alpha " + str(alpha) + ", epsilon : " + str(epsilon) ))
					figure.canvas.draw()

					###################### 
					### Training starts###
					######################


					for epochs in range(0,total_epochs) :

						#Set random seed
						np.random.seed(seed=epochs)

						indices = np.random.permutation(range(np.shape(X_tr)[1]))
						X_tr_shuf = X_tr[:,indices]	
						ytr_shuf = ytr[:,indices]	



						for minibatch in range(0,math.floor(np.shape(X_tr)[1]/mini_batch_size)) :


							X_minibatch = X_tr_shuf[:,(minibatch)*mini_batch_size:(minibatch+1)*mini_batch_size]
							y_minibatch = ytr_shuf[:,(minibatch)*mini_batch_size:(minibatch+1)*mini_batch_size]

							W_new,b_new = parameter_update(X_minibatch,y_minibatch,W,b,epsilon, alpha)
							W = W_new
							b = b_new
							

						costfn = cost_function(X_tr,ytr,W_new,b_new, alpha)

						costfn_set.append(costfn)
						x.append(epochs+1)

						figure.canvas.restore_region(background)
						ax.plot(x,costfn_set, color = 'black')
						figure.canvas.blit(ax.bbox)
						figure.canvas.flush_events()
							
					plt.cla()
					


					#####################
					### Training ends ###
					#####################
			
					valcostfn = cost_function(X_val,y_val,W,b, alpha)

					# Display costs of validation and training set over terminal post training, given a hyperparameter set	
					
					print('cost function for epochs of', total_epochs," and mini_batch_size of ", mini_batch_size, " with alpha of ", alpha, "and epsilon of", epsilon, 'is: ', valcostfn)

					print('cost with same hyperparameters on training set is: ', costfn)

					#Store relevant weights in a dictionary corresponding to a key which contains information on the training set

					key = str("Epochs : "+ str(total_epochs) + ", mini batch size : " + str(mini_batch_size) + ", alpha " + str(alpha) + ", epsilon : " + str(epsilon) )

					Costs_wrt_hyp[key] = [valcostfn, W,b]

				


	
	min_key = dictionary_minimum(Costs_wrt_hyp)

	

	return  Costs_wrt_hyp, min_key



if __name__ == '__main__':

	Costs_wrt_hyp, min_key = train_regressor()

	print("Optimal hyperparameters set :", min_key)

	X_te = np.load("fashion_mnist_test_images.npy").T

	X_te = X_te/255
	yte = (np.load("fashion_mnist_test_labels.npy"))

	#Perform one hot encoding
	yte = one_hotencoding(yte)


	#Find error on test set with optimal hyperparameter set and corresponding weights

	test_set_cost = cost_function(X_te,yte,Costs_wrt_hyp[min_key][1],Costs_wrt_hyp[min_key][2], 0)
	print("Test set cost: ",test_set_cost)


	#Find the accuracy

	estimate_full = softmax(X_te,Costs_wrt_hyp[min_key][1],Costs_wrt_hyp[min_key][2]);
	yte = yte.astype(int)


	count = 0
	for i in range (np.shape(yte)[1]):
		if np.argmax(yte[:,i]) == np.argmax(estimate_full[:,i]):
			count = count + 1

	print("Accuracy : ", 100*count/np.shape(yte)[1])
	

	#saving the data

	np.save('weights_FMNIST.npy',Costs_wrt_hyp[min_key][1])
	np.save('bias_FMNIST.npy',Costs_wrt_hyp[min_key][2])

	np.save('Costs_wrt_hyp.npy',  Costs_wrt_hyp)    

	#my_dict_back = np.load('Costs_wrt_hyp.npy',allow_pickle=True)

	#print(my_dict_back.item().get(min_key)[0])



