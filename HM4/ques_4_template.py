import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import math
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
# For this assignment, assume that every hidden layer has the same number of 
#neurons.
NUM_HIDDEN_LAYERS = 3
NUM_INPUT = 784
NUM_HIDDEN = 30
NUM_OUTPUT = 10
ALPHA = 0.001
EPSILON = 1
NUM_EPOCHS = 3
MINI_BATCH_SIZE = 64

BEST_NUM_HIDDEN_LAYERS = 3

BEST_NUM_HIDDEN = 30

BEST_ALPHA = 0.001
BEST_EPSILON = 1
BEST_NUM_EPOCHS = 3
BEST_MINI_BATCH_SIZE = 64

LINE_UP = '\033[1A'
LINE_CLEAR = '\033[2K\033[1G'





# Unpack a list of weights and biases into their individual np.arrays.
def unpack (weightsAndBiases):
    # Unpack arguments
    Ws = []
    # Weight matrices
    start = 0
    end = NUM_INPUT*NUM_HIDDEN
    W = weightsAndBiases[start:end]
    Ws.append(W)
    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN*NUM_HIDDEN
        W = weightsAndBiases[start:end]
        Ws.append(W)
    start = end
    end = end + NUM_HIDDEN*NUM_OUTPUT
    W = weightsAndBiases[start:end]
    Ws.append(W)
    Ws[0] = Ws[0].reshape(NUM_HIDDEN, NUM_INPUT)
    for i in range(1, NUM_HIDDEN_LAYERS):
        # Convert from vectors into matrices
        Ws[i] = Ws[i].reshape(NUM_HIDDEN, NUM_HIDDEN)
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, NUM_HIDDEN)
    # Bias terms
    bs = []
    start = end
    end = end + NUM_HIDDEN
    b = weightsAndBiases[start:end]
    bs.append(b)
    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN
        b = weightsAndBiases[start:end]
        bs.append(b)
    start = end
    end = end + NUM_OUTPUT
    b = weightsAndBiases[start:end]
    bs.append(b)
    for i in range(len(bs)):
        # Convert from vectors into matrices
        bs[i] = np.atleast_2d(bs[i]).T
    #bs[-1] = np.atleast_2d(bs[-1]).T
    


    return Ws, bs

def one_hotencoding(y):

    yencode = np.zeros((np.max(y)+1,np.shape(y)[0]))

    columns = np.arange(len(y))

    yencode[y,columns] = 1;

    return yencode


def cost (yhat,y,weightsAndBiases):
    Ws,bs = unpack(weightsAndBiases)
    n = np.shape(y)[1]
    reg_sum = 0
    for weights in Ws:
        reg_sum = reg_sum + np.sum(np.square(weights))
    loss = -np.sum(y*np.log(yhat))/n + ALPHA*reg_sum/(2*n)
    return loss



def forward_prop (x, y, weightsAndBiases):
    Ws, bs = unpack(weightsAndBiases)
    # Return loss, pre-activations, post-activations, and predictions
    H_l_1 = x
    hs = []
    zs = []
        
    for l in range(0,NUM_HIDDEN_LAYERS):
        Zl = Ws[l]@H_l_1 + bs[l]
        Hl = np.where(Zl<=0, 0.0, Zl)
        hs.append(Hl)
        zs.append(Zl)
        H_l_1 = Hl

    
    Zl = Ws[NUM_HIDDEN_LAYERS]@H_l_1 +bs[NUM_HIDDEN_LAYERS]


    den = (np.sum(np.exp(Zl),axis=0))

    num = (np.exp(Zl))

    Hl = num/den

    hs.append(Hl)
    zs.append(Zl)

    yhat = Hl

    n = np.shape(y)[1]
    reg_sum = 0
    for weights in Ws:
        reg_sum = reg_sum + np.sum(np.square(weights))

    loss = -np.sum(y*np.log(yhat))/n + ALPHA*reg_sum/(2*n)

    return loss, zs, hs, yhat



def softmax_back_prop(prediction,y):
    grad = (prediction - y)/(np.shape(y)[1])
    return grad

def relu_back_prop(zl):
    zl = np.where(zl<=0, 0.0, zl)
    zl = np.where(zl>0, 1.0, zl)
    
    return zl
   
def back_prop (x, y, weightsAndBiases):
    Ws, bs = unpack(weightsAndBiases)
    loss, zs, hs, yhat = forward_prop(x, y, weightsAndBiases)
    n = np.shape(y)[1]
    dJdWs = []  # Gradients w.r.t. weights
    dJdbs = []  # Gradients w.r.t. biases
    dJdzL = softmax_back_prop(yhat,y)

    

    dJdWL = dJdzL@hs[-2].T + ALPHA*Ws[-1]/(n)
    dJdbL = np.atleast_2d(np.sum(dJdzL,axis=1)).T

    dJdhi = Ws[-1].T@dJdzL

    
    dJdWs.append(dJdWL)
    dJdbs.append(dJdbL)
    Ws[-1] = Ws[-1] - EPSILON*dJdWL
    bs[-1] = bs[-1] - EPSILON*dJdbL

    

    for i in range(NUM_HIDDEN_LAYERS-1, -1, -1):
        dh_dzi = relu_back_prop(zs[i])
        dJdzi = dJdhi*dh_dzi

        if i >0:
            dJdWi = dJdzi@hs[i-1].T + ALPHA*Ws[i]/(n)
            dJdbi = np.atleast_2d(np.sum(dJdzi,axis=1)).T
            dJdWs.append(dJdWi)
            dJdbs.append(dJdbi)
       
        else:
            dJdWi = dJdzi@x.T + ALPHA*Ws[i]/(n)
            dJdbi = np.atleast_2d(np.sum(dJdzi,axis=1)).T
            dJdWs.append(dJdWi)
            dJdbs.append(dJdbi)
          

        #Ws[i] = Ws[i] - EPSILON*dJdWi
        #bs[i] = bs[i] - EPSILON*dJdbi

        dJdhi = Ws[i].T@dJdzi

        
        
    # Concatenate gradients

    reverse_dJdWs = []
    reverse_dJdbs = []

    for i in reversed(dJdWs):
        reverse_dJdWs.append(i)
    for j in reversed(dJdbs):
        reverse_dJdbs.append(j)
        
    #print(dJdbi)

    return np.hstack([ dJdW.flatten() for dJdW in reverse_dJdWs ] + [ dJdb.flatten() for dJdb in reverse_dJdbs ])





def findBestHyperparameters(NUM_HIDDEN_LAYERS_SET,NUM_HIDDEN_SET,ALPHA_SET,EPSILON_SET,NUM_EPOCHS_SET,MINI_BATCH_SIZE_SET,X_tr,y_tr,X_val,y_val):
    global NUM_HIDDEN_LAYERS 
    global NUM_HIDDEN
    global ALPHA
    global EPSILON
    global NUM_EPOCHS
    global MINI_BATCH_SIZE

    loss_min = 100
    start_best = 0
    start_normal = 0
    bestweightsandbias = []
    best_trajectory = []

    for NUM_HIDDEN_LAYERS in NUM_HIDDEN_LAYERS_SET:
        for NUM_HIDDEN in NUM_HIDDEN_SET:
            for ALPHA in ALPHA_SET:
                for EPSILON in EPSILON_SET:
                    for  NUM_EPOCHS in NUM_EPOCHS_SET:
                        for MINI_BATCH_SIZE in MINI_BATCH_SIZE_SET:
                            index = MINI_BATCH_SIZE_SET.index(MINI_BATCH_SIZE) + NUM_EPOCHS_SET.index(NUM_EPOCHS)*len(MINI_BATCH_SIZE_SET)*NUM_EPOCHS/65 \
                             + EPSILON_SET.index(EPSILON)*len(MINI_BATCH_SIZE_SET)*len(NUM_EPOCHS_SET) \
                             + ALPHA_SET.index(ALPHA)*len(MINI_BATCH_SIZE_SET)*len(NUM_EPOCHS_SET)*len(EPSILON_SET) \
                             + NUM_HIDDEN_SET.index(NUM_HIDDEN)*len(MINI_BATCH_SIZE_SET)*len(NUM_EPOCHS_SET)*len(EPSILON_SET)*len(ALPHA_SET)\
                              +NUM_HIDDEN_LAYERS_SET.index(NUM_HIDDEN_LAYERS)*len(MINI_BATCH_SIZE_SET)*len(NUM_EPOCHS_SET)*len(EPSILON_SET)*len(ALPHA_SET)*len(NUM_HIDDEN_SET)

                            percentage = index/(len(NUM_HIDDEN_LAYERS_SET)*len(MINI_BATCH_SIZE_SET)*len(NUM_EPOCHS_SET)*len(EPSILON_SET)*len(ALPHA_SET)*len(NUM_HIDDEN_SET))

                            if start_normal == 1:
                                print(" percentage: ", percentage*100, "%")

                            weightsAndBiases = initWeightsAndBiases()

                            weightsAndBiases, trajectory,loss,val_loss = train(X_tr, y_tr, weightsAndBiases, X_val, y_val)

                            if start_normal == 1:
                                print(LINE_UP,LINE_CLEAR,end="")
                            start_normal = 1
                            if val_loss < loss_min:     
                                BEST_NUM_HIDDEN_LAYERS = NUM_HIDDEN_LAYERS
                                BEST_NUM_HIDDEN = NUM_HIDDEN 
                                BEST_ALPHA = ALPHA
                                BEST_EPSILON = EPSILON
                                BEST_NUM_EPOCHS = NUM_EPOCHS
                                BEST_MINI_BATCH_SIZE = MINI_BATCH_SIZE
                                best_trajectory = trajectory
                                bestweightsandbias = weightsAndBiases
                                loss_min = val_loss
                                if start_best ==1:
                                    for i in range(9):
                                        print(LINE_UP,LINE_CLEAR,end="")
                                start_best = 1
                                print(" current best validation loss", val_loss)
                                print(" corresonding training loss  ", loss)
                                print( " At minibatch_size: ", MINI_BATCH_SIZE)
                                print( " At Epoch_Length: ", NUM_EPOCHS)
                                print( " At learning rate: ", EPSILON)
                                print( " At regularization: ", ALPHA)
                                print( " At units in hidden layers: ", NUM_HIDDEN)
                                print(" At hidden layers: ", NUM_HIDDEN_LAYERS)
                                print("--------------------")


    NUM_HIDDEN_LAYERS = BEST_NUM_HIDDEN_LAYERS 
    NUM_HIDDEN = BEST_NUM_HIDDEN 
    ALPHA = BEST_ALPHA 
    EPSILON = BEST_EPSILON
    NUM_EPOCHS = BEST_NUM_EPOCHS
    MINI_BATCH_SIZE = BEST_MINI_BATCH_SIZE                        


    return bestweightsandbias, best_trajectory




def train (trainX, trainY, weightsAndBiases, testX, testY):

    trajectory = []

    global EPSILON

    eps = EPSILON

    print(" training_loss: inf")
    print( " for minibatch_size: ", MINI_BATCH_SIZE)
    print( " for Epoch_Length: ", NUM_EPOCHS)
    print( " for learning rate: ", eps)
    print( " for regularization: ", ALPHA)
    print( " for units in hidden layers: ", NUM_HIDDEN)
    print(" for hidden layers: ", NUM_HIDDEN_LAYERS)
    print( " At Epoch: 0" )
    
    for epoch in range(NUM_EPOCHS):

        if epoch == 60:
            eps = 0.005
        elif epoch == 240:
            eps = 0.0005


        indices = np.random.permutation(np.arange(np.shape(trainX)[1]))
        X_tr_shuf = trainX[:,indices] 
        ytr_shuf = trainY[:,indices]   

        for minibatch in range(0,math.floor(np.shape(X_tr)[1]/MINI_BATCH_SIZE)) :

            X_minibatch = X_tr_shuf[:,(minibatch)*MINI_BATCH_SIZE:(minibatch+1)*MINI_BATCH_SIZE]
            y_minibatch = ytr_shuf[:,(minibatch)*MINI_BATCH_SIZE:(minibatch+1)*MINI_BATCH_SIZE]


        # implement SGD.
            grad = back_prop (X_minibatch, y_minibatch, weightsAndBiases)
            weightsAndBiases = weightsAndBiases - eps*grad

        for i in range(8):
            print(LINE_UP,LINE_CLEAR,end="")

        loss,_,_,_ = forward_prop(trainX,trainY,weightsAndBiases)
        print(" training_loss:", loss)
        print( " for minibatch_size: ", MINI_BATCH_SIZE)
        print( " for Epoch_Length: ", NUM_EPOCHS)
        print( " for learning rate: ", eps)
        print( " for regularization: ", ALPHA)
        print( " for units in hidden layers: ", NUM_HIDDEN)
        print(" for hidden layers: ", NUM_HIDDEN_LAYERS)
        print( " At Epoch: ", epoch+1)

        

        trajectory.append(weightsAndBiases)


        # TODO: save the current set of weights and biases into trajectory; this is
        # useful for visualizing the SGD trajectory.
    for i in range(8):
        print(LINE_UP,LINE_CLEAR,end="")
    val_loss,_,_,_ = forward_prop(testX,testY,weightsAndBiases)
        
    return weightsAndBiases, trajectory,loss,val_loss
# Performs a standard form of random initialization of weights and biases
def initWeightsAndBiases ():
    Ws = []
    bs = []
    np.random.seed(42)
    W = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    Ws.append(W)
    b = 0.01 * np.ones((NUM_HIDDEN,1))
    bs.append(b)
    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2*(np.random.random(size=(NUM_HIDDEN, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
        Ws.append(W)
        b = 0.01 * np.ones((NUM_HIDDEN,1))
        bs.append(b)
    W = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    Ws.append(W)
    b = 0.01 * np.ones((NUM_OUTPUT,1))
    bs.append(b)
    return np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])



def plotSGDPath (trainX, trainY, trajectory):
    pca = PCA(n_components=2)

    traj = np.array(trajectory)

    #print("traj",(traj))
    transW_B = pca.fit_transform(traj)

    def toyFunction (x1, x2):
        invW_B = pca.inverse_transform([x1, x2])
        #print("invW_B", np.shape(invW_B))
        return forward_prop(trainX, trainY, invW_B)[0] #First return value of forward_prop is loss.

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Compute the CE loss on a grid of points (corresonding to different w).
    #axis1 = np.arange(-8, +8, 3)  
    #axis2 = np.arange(-8, +8, 3)
    #Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    #Zaxis = np.zeros((len(axis1), len(axis2)))
    #for i in range(len(axis1)):
    #    for j in range(len(axis2)):
    #        Zaxis[i,j] = toyFunction(Xaxis[i,j], Yaxis[i,j])
    #ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)  # Keep alpha < 1 so we can see the scatter plot too.


    # # Now superimpose a scatter plot showing the weights during SGD.

    

    
    Zaxis = np.zeros(np.shape(transW_B)[0])
    for i in range(len(transW_B)):
        Zaxis[i] = toyFunction(transW_B[i,0],transW_B[i,1])

    ax.scatter(np.atleast_2d(transW_B[:,0]).T, np.atleast_2d(transW_B[:,1]).T, np.atleast_2d(Zaxis).T,  color ='r') 
    #Zaxis = toyFunction(Xaxis, Yaxis)
    #ax.scatter(Xaxis, Yaxis, Zaxis, color='r')

    plt.show()





if __name__ == "__main__":
    # Load data and split into train, validation, test sets

    X_tr = np.load("fashion_mnist_train_images.npy").T

    X_tr = X_tr/255
    y_tr = (np.load("fashion_mnist_train_labels.npy"))



    #Perform one hot encoding
    y_tr = one_hotencoding(y_tr)

    #Seperating the validation set after permutating the examples

    #Set random seed
    np.random.seed(seed=42)

    indices = np.random.permutation(range(np.shape(X_tr)[1]))
    X_tr = X_tr[:,indices]  
    y_tr = y_tr[:,indices]


    num_ex_val = math.floor(0.2*np.shape(X_tr)[1])

    X_val = X_tr[:,-num_ex_val:]
    X_tr = X_tr[:,0:-num_ex_val]

    y_val = y_tr[:,-num_ex_val:]
    y_tr = y_tr[:,0:-num_ex_val]



    # Initialize weights and biases randomly
    weightsAndBiases = initWeightsAndBiases()

        



    # Perform gradient check on 5 training examples
    print("check_grad", scipy.optimize.check_grad(lambda wab: 
forward_prop(np.atleast_2d(X_tr[:,0:5]), np.atleast_2d(y_tr[:,0:5]), wab)[0], \
                                    lambda wab: 
back_prop(np.atleast_2d(X_tr[:,0:5]), np.atleast_2d(y_tr[:,0:5]), wab), \
                                    weightsAndBiases))


    NUM_HIDDEN_LAYERS_SET = [5]
    NUM_INPUT = 784
    NUM_HIDDEN_SET = [40]#[35, 45]
    NUM_OUTPUT = 10
    ALPHA_SET = [0.3]
    EPSILON_SET = [0.05]
    NUM_EPOCHS_SET = [400]
    MINI_BATCH_SIZE_SET = [128]

    ws,trajectories = findBestHyperparameters(NUM_HIDDEN_LAYERS_SET,NUM_HIDDEN_SET,ALPHA_SET,EPSILON_SET,NUM_EPOCHS_SET,MINI_BATCH_SIZE_SET,X_tr,y_tr,X_val,y_val)



    X_te = np.load("fashion_mnist_test_images.npy").T

    X_te = X_te/255
    yte = (np.load("fashion_mnist_test_labels.npy"))

    #Perform one hot encoding
    yte = one_hotencoding(yte)


    #Find error on test set with optimal hyperparameter set and corresponding weights
    test_set_cost,_,_,yhat = forward_prop(X_te, yte, ws)
    count = 0

    print("Test set cost: ", test_set_cost)

    for i in range (np.shape(yte)[1]):
        if np.argmax(yte[:,i]) == np.argmax(yhat[:,i]):
            count = count + 1

    print("Accuracy : ", 100*count/np.shape(yte)[1])
    
















    
    # Plot the SGD trajectory
    plotSGDPath(X_tr, y_tr, trajectories)
    
    # Plot the SGD trajectory
   # plotSGDPath(trainX, trainY, ws)