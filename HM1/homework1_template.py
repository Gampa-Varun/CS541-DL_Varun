import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.stats import ks_2samp

def problem_1a (A, B):
	return A + B

def problem_1b (A, B, C):
	temp = A@B
	return temp - C

def problem_1c (A, B, C):
	temp = A*B
	return temp + np.transpose(C)

def problem_1d (x, y):
	xT = np.transpose(x)
	return xT@y

def problem_1e (A, x):
	return np.linalg.solve(A,x)

def problem_1f (A, x):
	return np.transpose(np.linalg.solve(np.transpose(A),np.transpose(x)))

def problem_1g (A, i):
	temp = A[i,:]
	length = np.shape(A)[1]
	indices = np.arange(0,length,2)
	return np.sum(np.take(A[i,:],indices))

def problem_1h (A, c, d):
	indices = np.nonzero((A >= c)*(A<=d))
	mean = np.mean(A[indices])
	return mean

def problem_1i (A, k):
	try:
		w,v = np.linalg.eig(A)
		print(w)
		kvectors = v[:,0:k]	
	except:
		print(" A is not a square matrix")
		sys.exit(1)
	return kvectors

def problem_1j (x, k, m, s):

	gaussian_matrix = []
	x = x.flatten()
	for i in range(k):
		gaussian_matrix.append(np.random.multivariate_normal(x + m*np.ones((len(x),)),s*np.identity(len(x))))
	return gaussian_matrix

def problem_1k (A):
	indices = np.arange(0,np.shape(A)[0])
	np.random.shuffle(indices)
	A = A[indices]
	return A

def problem_1l (x):
	meanx = np.mean(x,dtype = float)
	stdx = np.std(x)
	y = (x-meanx)/stdx
	return y

def problem_1m (x, k):
	print(len(x))
	x = np.reshape(x,(len(x),1))
	temp = np.repeat(x,k)
	return np.reshape(temp,(len(x),k))

def problem_1n (X):
	temp = np.repeat(X,np.shape(X)[1])
	#print(temp)
	nX = np.reshape(temp,(np.shape(X)[0],np.shape(X)[1],np.shape(X)[1]))
	#print(nX)
	nX = np.swapaxes(nX,1,2)
	#print(nX)
	nX = np.swapaxes(nX,1,0)
	#print(nX)

	#print((nX[0,:,:]))
	#print(np.transpose(nX[:,:,:]) - nX[:,:,:])
	#Difference = np.transpose(nX[:,:,:]) - nX[:,:,:]
	#print("Difference: \n", Difference)
	D = np.sqrt((np.sum(np.square(np.transpose(nX[:,:,:]) - nX[:,:,:]),axis=1)))
	return D

def linear_regression (X_tr, y_tr):
	w = np.random.randn((np.shape(X_tr)[1]))

	Coefficient_matrix = np.transpose(X_tr)@X_tr
	
	y_tr = np.atleast_2d(y_tr)
	y_tr = np.transpose(y_tr)
	
	Dependent_vector =  np.transpose(X_tr)@y_tr
	w = np.linalg.solve(Coefficient_matrix,Dependent_vector)

	return w




def train_age_regressor ():
	# Load data
	X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
	ytr = np.load("age_regression_ytr.npy")
	X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
	yte = np.load("age_regression_yte.npy")

	w = (linear_regression(X_tr, ytr))

	print(w[0])


	# Report fMSE cost on the training and testing data (separately)
	# ...
	#fMSE cost on training data
	ytr = ytr[:,np.newaxis]
	fMSE_training = np.sum((np.square(X_tr@w - ytr))/(2*np.shape(X_tr)[0]))

	#fMSE cost on test data
	yte = yte[:,np.newaxis]
	fMSE_test = np.sum((np.square(X_te@w - yte))/(2*np.shape(X_te)[0]))

	return fMSE_training,fMSE_test

def Poisson_param_est ():

	data = np.load("PoissonX.npy")
	print(np.shape(data))
	fig, ax = plt.subplots(figsize =(7, 4))
	bins = np.arange(13)
	ax.hist(data,bins=bins, density = True,color = 'red',edgecolor ="black", linewidth=1)
	fig.suptitle("Empirical data histogram", fontsize=15)
	plt.savefig("empirical_data.jpg", format="jpg", bbox_inches="tight")
	
	np.random.seed(seed=42424242)
	muVec = [2.5,3.1,3.7,4.3]
	Randomvalues = []

	for mu in muVec:
		Randomvalues.append(poisson.rvs(mu,size = np.shape(data)[0]))

	Randomvalues = np.asarray(Randomvalues)

	fig, axs = plt.subplots(2,2, figsize=(10,7))



	fig.suptitle("Random Variable histograms for different mean values of Poisson distribution", fontsize=15)
	axs[0,0].hist(Randomvalues[0,:],bins=bins, density =True,edgecolor ="black", linewidth=1)
	axs[0,0].set_title("mu = 2.5")
	axs[0,1].hist(Randomvalues[1,:],bins=bins, density =True,edgecolor ="black", linewidth=1)
	axs[0,1].set_title("mu = 3.1")
	axs[1,0].hist(Randomvalues[2,:],bins=bins, density =True,edgecolor ="black", linewidth=1)
	axs[1,0].set_title("mu = 3.7")
	axs[1,1].hist(Randomvalues[3,:],bins=bins, density =True,edgecolor ="black", linewidth=1)
	axs[1,1].set_title("mu = 4.3")
	plt.savefig("generated_data.jpg", format="jpg", bbox_inches="tight")
	plt.show()
	

	return ...

if __name__ == '__main__':
	if len(sys.argv) == 2:
		if str(sys.argv[1]) == 'example':
			np.random.seed(seed=42)
			A = np.random.randint(10, size= (4,4))
			np.random.seed(seed=43)
			B = np.random.randint(4, size= (2,2))
			np.random.seed(seed=44)
			C = np.random.randint(4, size= (2,2))
			np.random.seed(seed=45)
			x = np.random.randint(4, size= (2,1))
			#x= np.array(((3),(2)))
			#x = x[np.newaxis,:]
			np.random.seed(seed=46)
			y = np.random.randint(10, size= (2,1))
			###
			print("A: \n",A)
			print("B: \n",B)
			print("C: \n",C)
			print("x: \n",x)
			print("y: \n", y)
			###
			ans = train_age_regressor()
			print("Answer: \n",ans)