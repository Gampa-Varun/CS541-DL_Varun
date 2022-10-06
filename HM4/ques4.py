import numpy as np 
import math
import matplotlib.pyplot as plt
import time
import numba as nb
from scipy.optimize import check_grad


def one_hotencoding(y):

	yencode = np.zeros((np.max(y)+1,np.shape(y)[0]))

	columns = np.arange(len(y))

	yencode[y,columns] = 1;

	return yencode

class Model_architeture():
	"""docstring for Model_train"""
	def __init__(self, Hypset,X_tr,y_tr,X_val,y_val):
		
		self.layers = Hypset[0]
		self.units = [np.shape(X_tr)[0]] + Hypset[1]
		self.alpha = Hypset[2]
		self.epsilon = Hypset[3]
		self.paramsW = {}
		self.paramsb = {}
		self.caches = {}
		self.X_tr = X_tr
		self.y_tr = y_tr
		self.X_val = X_val
		self.y_val = y_val
		
		self.prediction = np.zeros((Hypset[1][-1],np.shape(X_tr)[1]))

		L = self.layers
		for l in range(1,L+1):
			np.random.seed(l)
			self.paramsW['W'+str(l)] = np.random.normal(loc = 0.0, scale = 1/np.sqrt(self.units[l-1]), \
													   size = (self.units[l],self.units[l-1])).astype('float32')
			self.paramsb['b'+str(l)] = np.random.normal(loc = 0.0, scale = 1/np.sqrt(self.units[l]), \
														size = (self.units[l],1)).astype('float32')


	def linear_forward(self, A_prev,W,b):
		H = W@A_prev + b
		
		linear_cache = [A_prev,W,b]

		return H, linear_cache

	def relu(self,H):

		A = np.where(H<0, 0, H)
		activation_cache = H
		return A,activation_cache

	def  softmax(self,H):
		
		den = (np.sum(np.exp(H),axis=0))

		num = (np.exp(H))

		A = num/den

		activation_cache = H

		return A, activation_cache

	def linear_forward_activation(self,A_prev,W,b,type):

		H, linear_cache = self.linear_forward(A_prev,W,b)

		if type == 'relu':
			A, activation_cache = self.relu(H)

		elif type == 'softmax':
			A, activation_cache = self.softmax(H)

		#print(linear_cache)

		cache = [activation_cache,linear_cache]

		return A,cache

	def forward_propagation(self):
		
		A_L_1 = self.X_tr
		
		for l in range(1,self.layers):

			AL, self.caches['Layer' + str(l)] = self.linear_forward_activation(A_L_1,self.paramsW['W'+str(l)],self.paramsb['b'+str(l)],'relu')
			print("layer",l)
			A_L_1 = AL

		AL,self.caches['Layer' + str(self.layers)] = self.linear_forward_activation(A_L_1,self.paramsW['W'+str(self.layers)],
														self.paramsb['b'+str(self.layers)],'softmax')

		self.prediction = AL

		return np.shape(AL)

	def loss(self):

		n = np.shape(self.y_tr)[1]
		reg_sum = 0
		for weights in self.paramsW:
			reg_sum = reg_sum + np.sum(np.square(self.paramsW[weights]))

		loss = -np.sum(self.y_tr*np.log(self.prediction))/n + self.alpha*reg_sum/(2*n)
		return loss

	def softmax_back_prop(self,prediction,y):
		grad = (prediction - y)/np.shape(y)[1]
		return grad

	def relu_back_prop(self,HL):
		HL = np.where(HL<=0, 0, HL)
		HL = np.where(HL>0, 1, HL)
		return HL

	def back_propagate(self):
		
		n = np.shape(self.y_tr)[1]
		g = 0
		gradW = {}
		gradb = {}
		for l in reversed(range(1,self.layers+1)):

			HL, AWb = self.caches['Layer' + str(l)] 

			A_L_1,WL, bL = AWb

			if l == self.layers:
				g = self.softmax_back_prop(self.prediction,self.y_tr)
			else :
				g  = g*self.relu_back_prop(HL)

		
			dW = g@A_L_1.T + self.alpha*WL/n
			db = np.sum(g,axis=1)
		
			#gradW.append(dW);
			#gradb.append(db)

			gradW['W'+str(l)] = dW
			gradb['b'+str(l)] = db

			g = WL.T@g

		return gradW,gradb

	def parameter_update(self,gradW,gradb):
		for weights in self.paramsW:
			self.paramsW[weights] = self.paramsW[weights] - self.epsilon*gradW[weights]
		for bias in self.paramsb:
			self.paramsb[bias] = self.paramsb[bias] - self.epsilon*np.atleast_2d(gradb[bias]).T




def Model_train():

		# Load data

	X_tr = np.load("fashion_mnist_train_images.npy").T

	X_tr = X_tr/255
	y_tr = (np.load("fashion_mnist_train_labels.npy"))



	#Perform one hot encoding
	y_tr = one_hotencoding(y_tr)

	#Seperating the validation set after permutating the examples

	#Set random seed
	np.random.seed(seed=1)

	indices = np.random.permutation(range(np.shape(X_tr)[1]))
	X_tr = X_tr[:,indices]	
	y_tr = y_tr[:,indices]


	num_ex_val = math.floor(0.2*np.shape(X_tr)[1])

	X_val = X_tr[:,-num_ex_val:]
	X_tr = X_tr[:,0:-num_ex_val]

	y_val = y_tr[:,-num_ex_val:]
	y_tr = y_tr[:,0:-num_ex_val]

	#print(np.shape(X_tr))

 	# Hyparameter sets

	Hypset = [4,[30,30,30,10],0,1]
	model = Model_architeture(Hypset,X_tr,y_tr,X_val,y_val)

	for i in range(1,20):


		print(i)
		model.forward_propagation()
		print(model.loss())

		w_up,b_up = model.back_propagate()
		model.parameter_update(w_up,b_up)
		model.forward_propagation()
		print(model.loss())

	#model_check = Model_architeture(Hypset,X_tr[:,1:5],y_tr[:,1:5],X_val,y_val)

	#print(check_grad(model.forward_propagation(),model.back_propagate(),(model.paramsW,model.paramsb)))

'''
print(scipy.optimize.check_grad(lambda wab: 
forward_prop(np.atleast_2d(trainX[:,0:5]), np.atleast_2d(trainY[:,0:5]), wab)[0], \
                                    lambda wab: 
back_prop(np.atleast_2d(trainX[:,0:5]), np.atleast_2d(trainY[:,0:5]), wab), \
                                    weightsAndBiases))
'''




if __name__ == '__main__':

	Model_train()
	#W,b,alpha_best,epsilon_best,epoch_best,batch_size_best = Model_train()

	""" print('Optimal params are -  epochs of', epoch_best," and mini_batch_size of ", batch_size_best, " with alpha of ",alpha_best, "and epsilon of", epsilon_best)

	X_te = np.load("fashion_mnist_test_images.npy").astype('float32')

	X_te = X_te/255
	yte = (np.load("fashion_mnist_test_labels.npy")).astype('int')

	#Perform one hot encoding
	yte = one_hotencoding(yte).T.astype('float32') """


	

