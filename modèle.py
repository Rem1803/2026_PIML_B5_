import numpy as np 
import matplotlib as plt
from PIL import Image


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def eval_forward(x, w, b, verb=0):
    """
    Evaluate a MLP on an input vector.
    Args:
        x: Input vector.
        w: List containing for each layer its matrix of weights.
        b: List containing for each layer its vector of biases.
        verb: verbosity. Default 0, no message.
    """

    L = len(w) - 1
    a = [np.copy(x)] # layer 0 is the input vector


    for l in range(1,L+1):
        z_l = np.matmul(w[l], a[l-1]) + b[l]
        a_l = sigmoid(z_l)
        a.append(a_l)
            
    return a


def init_parameters(n_feats, hidden_layer_sizes, rng=None):
    """
    Random initialization of weights and biases.
    Args:
        n_feats: Number of features (i.e., size of layer 0).
        hidden_layer_sizes: List of the sizes of each of the hidden layers.
        rng: Random generator if any.
    Returns:
        Lists of weights and biases initialized of each layer.
        """
    
    if rng is None:
        rng = np.random.default_rng()

    L = 1 + len(hidden_layer_sizes)
    # Initialization of parameters for layer 0 (input layer) as an empty matrix
    # and an empty vector since there is no weight and bias associated to this layer.
    w = [np.zeros((0,0))]
    b = [np.zeros(0)]

    # Initialization of parameters for layer 1, i.e., first hidden layer
    # that takes values from all features of the input layer.
    # Biases are initialized to zero. Weights are picked at random
    # in [-0.5; 0.5[ in a uniform way.
    w.append(rng.normal(0,np.sqrt(2 / n_feats),(hidden_layer_sizes[0], n_feats)))
    b.append(np.zeros((hidden_layer_sizes[0])))
    
    # Initialization of parameters for layers 2 to L-1.
    for l in range(2, L):
        fan_in = hidden_layer_sizes[l-2]
        w.append(rng.normal(0,np.sqrt(2 / fan_in),(hidden_layer_sizes[l-1], fan_in)))
        b.append(np.zeros((hidden_layer_sizes[l-1])))

    # Initialization of parameters for layer L (output layer)
    fan_in = hidden_layer_sizes[L-2]

    w.append(rng.normal(0,np.sqrt(2 / fan_in),(1, fan_in)))
    b.append(np.zeros((1)))

    return w, b


def mlp_error(data, target, w, b):
    E = 0
    for x in range(len(data)):
        a=eval_forward(data[x],w,b)
        pred = a [-1]
        e = np.sum((target[x]-pred)**2)
        E+=e
    return E


def mlp_fit(data, target, n_epochs=10, hidden_layer_sizes=[3,2], learning_rate=0.2,
            random_state=None, verb=0):
    
    if random_state != None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()

    n_objs, n_feats = data.shape
    L = len(hidden_layer_sizes) + 1

    w, b = init_parameters(n_feats, hidden_layer_sizes, rng)
    

    for epoch in range(n_epochs):
        err = [None] * (L+1)
        for x in range(n_objs) :
            #prediction
            a = eval_forward(data[x],w,b)
            #erreur en sortie
            da_output = a[-1]*(1-a[-1])
            err[-1] = 2*(a[-1]-target[x])*da_output
                
            #Backpropagation in hidden layers 
            for l in range(L-1,0,-1):
                da = a[l]*(1-a[l])
                err_next = err[l+1]
                err_l = np.matmul(w[l+1].T, err_next)*da
                err[l] = err_l
            
            #mettre à jour le poids et les biais 
            for l in range(1, L+1):
                dw = np.outer(err[l], a[l-1])
                w[l] -= learning_rate * dw
                b[l] -= learning_rate * err[l]

    # end for epoch
    
    return w, b

def predict_proba(x, w, b):
    a = eval_forward(x, w, b)
    return a[-1][0]


def predict(x, w, b):
    proba = predict_proba(x, w, b)

    if proba >= 0.5:
        return 1
    else:
        return 0
    

def relu(z):
    return np.maximum(0,z)

def eval_forward_relu(x, w, b, verb=0):
    """
    Evaluate a MLP on an input vector.
    Args:
        x: Input vector.
        w: List containing for each layer its matrix of weights.
        b: List containing for each layer its vector of biases.
        verb: verbosity. Default 0, no message.
    """

    L = len(w) - 1
    a = [np.copy(x)] # layer 0 is the input vector


    for l in range(1,L+1):
        z_l = np.matmul(w[l], a[l-1]) + b[l]

        if l == L :
            a_l = sigmoid(z_l)   # sortie
        else :
            a_l = relu(z_l)
        
        a.append(a_l)
            
    return a

def mlp_error_relu(data, target, w, b):
    E = 0
    for x in range(len(data)):
        a=eval_forward_relu(data[x],w,b)
        pred = a [-1]
        e = np.sum((target[x]-pred)**2)
        E+=e
    return E

def mlp_fit_relu(data, target, n_epochs=10, hidden_layer_sizes=[3,2], learning_rate=0.2,
            random_state=None, verb=0):
    
    if random_state != None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()

    n_objs, n_feats = data.shape
    L = len(hidden_layer_sizes) + 1
    

    w, b = init_parameters(n_feats, hidden_layer_sizes, rng)
    losses = []
   

    for epoch in range(n_epochs):
        err = [None] * (L+1)
        for x in range(n_objs) :
            #prediction
            a = eval_forward_relu(data[x],w,b)
            #erreur en sortie
            da_output = a[-1] * (1 - a[-1])
            err[-1] = 2*(a[-1]-target[x])*da_output
                
            #Backpropagation in hidden layers 
            for l in range(L-1,0,-1):
                da = (a[l]>0).astype(float)
                err_next = err[l+1]
                err_l = np.matmul(w[l+1].T, err_next)*da
                err[l] = err_l
            
            #mettre à jour le poids et les biais 
            for l in range(1, L+1):
                dw = np.outer(err[l], a[l-1])
                w[l] -= learning_rate * dw
                b[l] -= learning_rate * err[l]

        loss = mlp_error(data, target, w, b)
        losses.append(loss)
                            
            
                     
        

    # end for epoch
    
    return w, b, losses

def predict_proba_relu(x, w, b):
    a = eval_forward_relu(x, w, b)
    return a[-1][0]

def predict_relu(x, w, b):
    proba = predict_proba_relu(x, w, b)

    if proba >= 0.5:
        return 1
    else:
        return 0
    

def mlp_error_entropy(data, target, w, b):
    '''Binary Cross Entropy'''
    E = 0
    epsilon = 1e-15
    
    for x in range(len(data)):
        a = eval_forward_relu(data[x], w, b)
        pred = a[-1][0]
        pred = np.clip(pred, epsilon, 1 - epsilon)
        y = target[x]
        e = - (y * np.log(pred) + (1-y) * np.log(1-pred))
        E += e

    return E

def mlp_fit_bce(data, target, n_epochs=10, hidden_layer_sizes=[3,2], learning_rate=0.001,
            random_state=None, verb=0):
    
    if random_state != None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()

    n_objs, n_feats = data.shape
    L = len(hidden_layer_sizes) + 1
    

    w, b = init_parameters(n_feats, hidden_layer_sizes, rng)
    losses = []
    
  

    for epoch in range(n_epochs):
        err = [None] * (L+1)
        for x in range(n_objs) :
            #prediction
            a = eval_forward_relu(data[x],w,b)
            #erreur en sortie
            err[-1] = a[-1] - target[x]
                
            #Backpropagation in hidden layers 
            for l in range(L-1,0,-1):
                da = (a[l]>0).astype(float)
                err_next = err[l+1]
                err_l = np.matmul(w[l+1].T, err_next)*da
                err[l] = err_l
            
            #mettre à jour le poids et les biais 
            for l in range(1, L+1):
                dw = np.outer(err[l], a[l-1])
                w[l] -= learning_rate * dw
                b[l] -= learning_rate * err[l]
        
        loss = mlp_error_entropy(data, target, w, b)
        losses.append(loss)
                            
            
                     
        

    # end for epoch
    
    return w, b, losses
