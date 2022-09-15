import numpy as np 
import math
import matplotlib.pyplot as plt
import time
import csv


def dictionary_minimum (dictionary):

	minvalue = float('inf')
	k = []

	for entry in dictionary.keys():
		if dictionary[entry][0] < minvalue:
			key = entry
			minvalue = dictionary[entry][0]
			

	return key


#Computing  the gradient of the cost function, 
#note that in the cost function bias is not regularised and the same suit is followed in the gradient as well
def grad_cost_function (X, y, w, b, alpha) :



	w_nonreg = np.vstack([w,b])

	w_reg = np.vstack([w,0])

	
	X = np.vstack([X, np.ones([1,np.shape(X)[1]])] )
	pred_min_true = (np.transpose(X)@w_nonreg - y)
	grad = X@pred_min_true/len(y) + alpha*w_reg/(np.shape(y)[0])


	return grad

#The weight vector and bias are updated based on learning rate and the gradient 
def parameter_update (X, y, w, b, epsilon, alpha) :

	gradient = grad_cost_function(X,y,w,b, alpha)

	#X = np.vstack([X, np.ones([1,np.shape(X)[1]])] )
	#Coefficient_matrix = X@np.transpose(X)
	#Dependent_vector =  X@y
	#w = np.linalg.solve(Coefficient_matrix,Dependent_vector)

	#w_new = w[0:-1]
	#b_new = w[-1]

	w_new = w - epsilon*gradient[0:-1]

	b_new = b - epsilon*gradient[-1]

	return w_new,b_new

#MSE Cost function with regularization except on bias term 
def cost_function (X, y, w, b, alpha) :

	cost = np.sum(np.square((np.transpose(X)@w + b*np.ones([np.shape(X)[1],1]) - y)),axis=0)*0.5/len(y) + np.ndarray.flatten(alpha*np.transpose(w)@w/(2*np.shape(y)[0]))

	return cost




def train_regressor ():

	

	# Load data
	X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
	ytr = np.load("age_regression_ytr.npy")
	

	
	# Rearrange data to have instances as column vectors
	X_tr = np.transpose(X_tr)

	# To make the result data in a column vector
	ytr = ytr[:,np.newaxis]

	#Seperating the validation set after permutating the examples


	#Set random seed
	np.random.seed(seed=1)

	indices = np.random.permutation(range(np.shape(X_tr)[1]))
	X_tr = X_tr[:,indices]	
	ytr = ytr[indices,:]


	num_ex_val = math.floor(0.2*np.shape(X_tr)[1])

	X_val = X_tr[:,-num_ex_val:]
	X_tr = X_tr[:,0:-num_ex_val]

	y_val = ytr[-num_ex_val:,:]
	ytr = ytr[0:-num_ex_val,:]

 	# Hyparameter sets

	epsilon_set = [0.0005, 0.001,0.002, 0.0025]

	alpha_set = [0.1,0.2,0.4,0.8]

	mini_batch_sizes = [10,50,100,200] 

	epoch_lengths = [100,200,400,500]




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
					np.random.seed(seed=1)

					w = np.random.normal(loc = 0.0, scale = 1.0, size = [np.shape(X_tr)[0],1])

					b = np.random.normal(loc = 0.0, scale = 1.0, size = 1)


					costfn = cost_function(X_tr,ytr,w,b,alpha)

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
						ytr_shuf = ytr[indices,:]	



						for minibatch in range(0,math.floor(np.shape(X_tr)[1]/mini_batch_size)) :


							X_minibatch = X_tr_shuf[:,(minibatch)*mini_batch_size:(minibatch+1)*mini_batch_size]
							y_minibatch = ytr_shuf[(minibatch)*mini_batch_size:(minibatch+1)*mini_batch_size,:]

							w_new,b_new = parameter_update(X_minibatch,y_minibatch,w,b,epsilon, alpha)
							w = w_new
							b = b_new
							

						costfn = cost_function(X_tr,ytr,w_new,b_new, alpha)

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
			
					valcostfn = cost_function(X_val,y_val,w,b, alpha)

					costfn = cost_function(X_tr,ytr,w,b, alpha)

					# Display costs of validation and training set over terminal post training, given a hyperparameter set	
					

					print('cost function for epochs of', total_epochs," and mini_batch_size of ", mini_batch_size, " with alpha of ", alpha, "and epsilon of", epsilon, 'is \n', valcostfn)

					print('cost with same hyperparameters on training set is \n', costfn)

					#Store relevant weights in a dictionary corresponding to a key which contains information on the training set

					key = str("Epochs : "+ str(total_epochs) + ", mini batch size : " + str(mini_batch_size) + ", alpha " + str(alpha) + ", epsilon : " + str(epsilon) )

					Costs_wrt_hyp[key] = [valcostfn, w,b]


	min_key = dictionary_minimum(Costs_wrt_hyp)

	


	return  Costs_wrt_hyp, min_key



if __name__ == '__main__':

	Costs_wrt_hyp, min_key = train_regressor()


	X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
	yte = np.load("age_regression_yte.npy")

	# Rearrange data to have instances as column vectors
	X_te = np.transpose(X_te)

	# To make the result data in a column vector
	yte = yte[:,np.newaxis]



	test_set_cost = cost_function(X_te,yte,Costs_wrt_hyp[min_key][1],Costs_wrt_hyp[min_key][2], 0)


	#saving the data

	np.save('weights.npy',Costs_wrt_hyp[min_key][1])
	np.save('bias.npy',Costs_wrt_hyp[min_key][2])

	# open file for writing, "w" is writing
	w = csv.writer(open("All weights.csv", "w"))

	# loop over dictionary keys and values
	for key, val in Costs_wrt_hyp.items():

		# write every key and value to file
		w.writerow([key, val])





	print("hyperparameters set :", min_key)
	print("Answer: \n",test_set_cost)


	#hyperparameters set : Epochs : 100, mini batch size : 200, alpha 0.8, epsilon : 0.002