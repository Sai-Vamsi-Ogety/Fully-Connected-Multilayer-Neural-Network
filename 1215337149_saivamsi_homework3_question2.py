'''
This file implements a multi layer neural network for a multiclass classifier

Hemanth Venkateswara
hkdv1@asu.edu
Oct 2018
'''
import numpy as np
from load_mnist import mnist,one_hot
import matplotlib.pyplot as plt
import pdb
import sys, ast

def relu(Z):
    '''
    computes relu activation of Z

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = np.maximum(0,Z)
    cache = {}
    cache["Z"] = Z
    return A, cache

def relu_der(dA, cache):
    '''
    computes derivative of relu activation

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    dZ = np.array(dA, copy=True)
    Z = cache["Z"]
    dZ[Z<0] = 0
    return dZ

def linear(Z):
    '''
    computes linear activation of Z
    This function is implemented for completeness

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = Z
    cache = {}
    return A, cache

   
 
def linear_der(dA, cache):
    '''
    computes derivative of linear activation
    This function is implemented for completeness

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    dZ = np.array(dA, copy=True)
    return dZ

def softmax_cross_entropy_loss(Z, Y=np.array([])):
    '''
    Computes the softmax activation of the inputs Z
    Estimates the cross entropy loss

    Inputs: 
        Z - numpy.ndarray (n, m)
        Y - numpy.ndarray (1, m) of labels
            when y=[] loss is set to []
    
    Returns:
        A - numpy.ndarray (n, m) of softmax activations
        cache -  a dictionary to store the activations later used to estimate derivatives
        loss - cost of prediction
    '''
    ### CODE HERE
    maxz = np.max(Z, axis = 0 , keepdims = True)
    A = np.exp(Z - maxz)
    A = A / np.sum( A, axis = 0, keepdims = True)
    #print("Activations")
    cache_Activation = {}
    cache_Activation["Activation"] = A

        
    one_hot_vector = one_hot(Y.astype(int),10)
    #print("one hot vector of Y")
    #print(one_hot_vector)
    #print(one_hot_vector.shape)
    #print(A.shape)

    loss = -np.sum(one_hot_vector.T*np.log(A))/(Y.shape[1])
    #print(Y.shape[1])
    #print("cross entropy loss")
    #print(loss)

    return A, cache_Activation, loss

def softmax_cross_entropy_loss_der(Y, cache):
    '''
    Computes the derivative of softmax activation and cross entropy loss

    Inputs: 
        Y - numpy.ndarray (1, m) of labels
        cache -  a dictionary with cached activations A of size (n,m)

    Returns:
        dZ - numpy.ndarray (n, m) derivative for the previous layer
    '''
    ### CODE HERE 
    one_hot_vector = one_hot(Y.astype(int),10)

    #print(one_hot_vector.shape)

    dZ = (cache["Activation"] - one_hot_vector.T)/Y.shape[1]
    
    return dZ

def initialize_multilayer_weights(net_dims):
    '''
    Initializes the weights of the multilayer network

    Inputs: 
        net_dims - tuple of network dimensions

    Returns:
        dictionary of parameters
    '''
    np.random.seed(0)
    numLayers = len(net_dims)
    parameters = {}
    for l in range(numLayers-1):
        parameters["W"+str(l+1)] = np.random.randn(net_dims[l+1],net_dims[l])*0.01#np.sqrt(2/net_dims[l+1])#CODE HERE
        parameters["b"+str(l+1)] = np.zeros((net_dims[l+1],1))*0.01#np.sqrt(2/net_dims[l+1])#CODE HERE
    return parameters

def linear_forward(A, W, b):
    '''
    Input A propagates through the layer 
    Z = WA + b is the output of this layer. 

    Inputs: 
        A - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer

    Returns:
        Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        cache - a dictionary containing the inputs A
    '''
    ### CODE HERE
    #print(W.shape)
    #print(A.shape)
    #print(b.shape)

    Z = np.dot(W, A) + b 
    cache = {}
    cache["A"] = A
    return Z, cache

def layer_forward(A_prev, W, b, activation):
    '''
    Input A_prev propagates through the layer and the activation

    Inputs: 
        A_prev - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer
        activation - is the string that specifies the activation function

    Returns:
        A = g(Z), where Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        g is the activation function
        cache - a dictionary containing the cache from the linear and the nonlinear propagation
        to be used for derivative
    '''
    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "relu":
        A, act_cache = relu(Z)
    elif activation == "linear":
        A, act_cache = linear(Z)
    
    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache
    return A, cache

def multi_layer_forward(X, parameters):
    '''
    Forward propgation through the layers of the network

    Inputs: 
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}
    Returns:
        AL - numpy.ndarray (c,m)  - outputs of the last fully connected layer before softmax
            where c is number of categories and m is number of samples in the batch
        caches - a dictionary of associated caches of parameters and network inputs
    '''
    L = len(parameters)//2  
    A = X
    #print("checking input")
    #print(A)
    caches = []
    for l in range(1,L):  # since there is no W0 and b0
        A, cache = layer_forward(A, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
        caches.append(cache)

    AL, cache = layer_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "linear")
    caches.append(cache)
    return AL, caches

def linear_backward(dZ, cache, W, b):
    '''
    Backward prpagation through the linear layer

    Inputs:
        dZ - numpy.ndarray (n,m) derivative dL/dz 
        cache - a dictionary containing the inputs A, for the linear layer
            where Z = WA + b,    
            Z is (n,m); W is (n,p); A is (p,m); b is (n,1)
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)

    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    ## CODE HERE
    A_prev = cache["A"]
    #print ("checking linear backward")
    #print (A_prev)
    dA_prev =  np.dot(W.T,dZ)
    #print (dA_prev)
    dW = np.dot(dZ,A_prev.T)
    #print (dW)
    #print("printing sum")
    #print(np.sum(dW))
    db = np.sum(dZ,axis = 1, keepdims = True)
    return dA_prev, dW, db

def layer_backward(dA, cache, W, b, activation):
    '''
    Backward propagation through the activation and linear layer

    Inputs:
        dA - numpy.ndarray (n,m) the derivative to the previous layer
        cache - dictionary containing the linear_cache and the activation_cache
        activation - activation of the layer
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)
    
    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "sigmoid":
        dZ = sigmoid_der(dA, act_cache)
    elif activation == "tanh":
        dZ = tanh_der(dA, act_cache)
    elif activation == "relu":
        dZ = relu_der(dA, act_cache)
    elif activation == "linear":
        dZ = linear_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db

def multi_layer_backward(dAL, caches, parameters):
    '''
    Back propgation through the layers of the network (except softmax cross entropy)
    softmax_cross_entropy can be handled separately

    Inputs: 
        dAL - numpy.ndarray (n,m) derivatives from the softmax_cross_entropy layer
        caches - a dictionary of associated caches of parameters and network inputs
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}

    Returns:
        gradients - dictionary of gradient of network parameters 
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
    '''
    L = len(caches)  # with one hidden layer, L = 2
    gradients = {}
    dA = dAL
    activation = "linear"
    for l in reversed(range(1,L+1)):
        dA, gradients["dW"+str(l)], gradients["db"+str(l)] = \
                    layer_backward(dA, caches[l-1], \
                    parameters["W"+str(l)],parameters["b"+str(l)],\
                    activation)
        activation = "relu"
    return gradients

def classify(X, parameters):
    '''
    Network prediction for inputs X

    Inputs: 
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters 
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
    Returns:
        YPred - numpy.ndarray (1,m) of predictions
    '''
    ### CODE HERE 
    # Forward propagate X using multi_layer_forward
    AL,caches = multi_layer_forward(X,parameters)

    # Get predictions using softmax_cross_entropy_loss
    maxX = np.max(AL, axis = 0, keepdims = True)
    e = np.exp(AL- maxX)
    AL = e / np.sum(e, axis = 0, keepdims = True)
    # Estimate the class labels using predictions
    Ypred = np.zeros(AL.shape)
    temp = AL.argmax(axis = 0)
    for i in range(0,AL.shape[1]):
        Ypred[temp[i] , i] = 1
     



    return Ypred

def update_parameters(parameters, gradients, epoch, learning_rate, decay_rate=0.0):
    '''
    Updates the network parameters with gradient descent

    Inputs:
        parameters - dictionary of network parameters 
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
        gradients - dictionary of gradient of network parameters 
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
        epoch - epoch number
        learning_rate - step size for learning
        decay_rate - rate of decay of step size - not necessary - in case you want to use
    '''
    alpha = learning_rate*(1/(1+decay_rate*epoch))
    L = len(parameters)//2
    #print (L)
    ### CODE HERE 
    for i in range(1,L+1):

        parameters["W" + str(i)] = parameters["W" + str(i)] - alpha * gradients["dW" + str(i)]
        #print("checking gradients working")
        #print(gradients["dW" + str(i)])
        parameters["b" + str(i)] = parameters["b" + str(i)] - alpha * gradients["db" + str(i)]

    
     
    return parameters, alpha

def multi_layer_network(X, Y,valid_data,valid_label, net_dims, num_iterations, learning_rate, decay_rate = 0.01): # changed the default values
    '''
    Creates the multilayer network and trains the network

    Inputs:
        X - numpy.ndarray (n,m) of training data
        Y - numpy.ndarray (1,m) of training data labels
        net_dims - tuple of layer dimensions
        num_iterations - num of epochs to train
        learning_rate - step size for gradient descent
    
    Returns:
        costs - list of costs over training
        parameters - dictionary of trained network parameters
    '''
    parameters = initialize_multilayer_weights(net_dims)
    #print("shape of the parameters")
    #print (len(parameters))
    A0 = X
    costs = []
    costs_valid = []
    for ii in range(num_iterations):
        ### CODE HERE
        # Forward Prop
        ## call to multi_layer_forward to get activations
        AL,caches =multi_layer_forward(X,parameters)
        #print("cache is")
        #print(len(cache))
        #print(cache[2])
        #print("A last layer")    
        #print(AL)
        ## call to softmax cross entropy loss
        A,cache_Activation,cost = softmax_cross_entropy_loss(AL,Y)

        #print("Activation")
        #print(A)
        #print("cross entropy loss")
        #print(loss)

        dZ = softmax_cross_entropy_loss_der(Y,cache_Activation)
        #print("softmax derivative")
        #print(dZ)


        grad = multi_layer_backward(dZ,caches,parameters)

        #print(grad)


        parameters,alpha = update_parameters(parameters, grad, num_iterations, learning_rate , decay_rate=0.01)

        #print("after update parameters")
        #print(parameters)
        #print(gradients)













        # Backward Prop
        ## call to softmax cross entropy loss der
        ## call to multi_layer_backward to get gradients
        ## call to update the parameters
        
        if ii % 10 == 0:
            costs.append(cost)
        if ii % 10 == 0:
            AL_valid,caches_valid =multi_layer_forward(valid_data,parameters)
            valid_pred = classify(valid_data,parameters)
            x1,cache_Activation,vaild_loss =  softmax_cross_entropy_loss(AL_valid,valid_label)
            costs_valid.append(vaild_loss)
            valid_label_new = one_hot(valid_label.astype(int),10)
            result_valid = np.sum(np.abs(valid_pred - valid_label_new.T))
            print("validation error is : %f"%(result_valid))



            print("Cost at iteration %i is: %.05f, learning rate: %.05f" %(ii, cost, learning_rate))
    
    return costs, parameters , costs_valid

def main():
    '''
    Trains a multilayer network for MNIST digit classification (all 10 digits)
    To create a network with 1 hidden layer of dimensions 800
    Run the progam as:
        python deepMultiClassNetwork_starter.py "[784,800]"
    The network will have the dimensions [784,800,10]
    784 is the input size of digit images (28pix x 28pix = 784)
    10 is the number of digits

    To create a network with 2 hidden layers of dimensions 800 and 500
    Run the progam as:
        python deepMultiClassNetwork_starter.py "[784,800,500]"
    The network will have the dimensions [784,800,500,10]
    784 is the input size of digit images (28pix x 28pix = 784)
    10 is the number of digits
    '''
    
    net_dims = ast.literal_eval( sys.argv[1] )
    net_dims.append(10) # Adding the digits layer with dimensionality = 10
    print("Network dimensions are:" + str(net_dims))

    # getting the subset dataset from MNIST
    train_data, train_label, test_data, test_label = \
            mnist(noTrSamples=6000,noTsSamples=1000,\
            digit_range=[0,1,2,3,4,5,6,7,8,9],\
            noTrPerClass=600, noTsPerClass=100)

    valid_data , valid_label = train_data[:,5000:6000], train_label[:,5000:6000]
          
    # initialize learning rate and num_iterations
    learning_rate = 10.0
    num_iterations = 1000
    num_train_samples = 5000
    num_validate_samples = 1000
    num_test_samples = 1000
    
    
    costs, parameters ,costs_valid = multi_layer_network(train_data, train_label,valid_data,valid_label, net_dims, \
            num_iterations=num_iterations, learning_rate= learning_rate, decay_rate = 0.0)
    
    # compute the accuracy for training set and testing set
    train_Pred = classify(train_data, parameters)
    test_Pred = classify(test_data, parameters)

    #print (train_label.shape)
    #print(train_Pred.shape)
    #print (train_label)
    #print(train_Pred)
    train_label_new = one_hot(train_label.astype(int),10)
    test_label_new = one_hot(test_label.astype(int),10)
    
    result_train = np.sum(abs(train_Pred - train_label_new.T))
    print("Train error is : %f"%(result_train))
    result_test = np.sum(abs(test_Pred - test_label_new.T))
    print("Test error is : %f"%(result_test))


    
    trAcc = 1/num_train_samples * np.sum(num_train_samples - result_train) * 100
    
    teAcc = 1/num_test_samples * np.sum(num_test_samples - result_test) * 100


    print("Accuracy for training set is {0:0.3f} %".format(trAcc))
    print("Accuracy for testing set is {0:0.3f} %".format(teAcc))
    
    ### CODE HERE to plot costs
    x = range(0,int(num_iterations/10))
    plt.plot(x,costs)
    plt.plot(x,costs_valid)
    plt.xlabel('iterations')
    plt.ylabel('Costs')
    plt.title('Training and validation')
    plt.show()


if __name__ == "__main__":
    main()