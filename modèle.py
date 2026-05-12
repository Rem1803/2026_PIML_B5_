import numpy as np 
import matplotlib as plt
from PIL import Image

img = np.asarray(Image.open(cell_images\Parasitized\C33P1thinF_IMG_20150619_114756a_cell_179.png))

def load_image(uninfected, parasited, image_size=())

def ptrace(message):
    print(message)


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

    if verb>0:
        ptrace(f"Forward evaluation {L=}")

    for l in range(1,L+1):
        z_l = np.matmul(w[l], a[l-1]) + b[l]
        a_l = sigmoid(z_l)
        if verb>1:
            ptrace(f"Layer {l=} {a_l=}")
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
    w.append(rng.random((hidden_layer_sizes[0], n_feats)) - 0.5)
    b.append(np.zeros((hidden_layer_sizes[0])))
    
    # Initialization of parameters for layers 2 to L-1.
    for l in range(2, L):
        w.append(rng.random((hidden_layer_sizes[l-1], hidden_layer_sizes[l-2])) - 0.5)
        b.append(np.zeros((hidden_layer_sizes[l-1])))

    # Initialization of parameters for layer L (output layer)
    w.append(rng.random((1, hidden_layer_sizes[L-2])) - 0.5)
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


def mlp_fit(data, target, n_epochs=5000, hidden_layer_sizes=[3,2], learning_rate=0.2,
            random_state=None, verb=0):
    
    if random_state != None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()

    n_objs, n_feats = data.shape
    L = len(hidden_layer_sizes) + 1
    if verb>0:
        ptrace("="*40)
        ptrace(f"Fitting {n_objs=}, {n_feats=}, {L=}")

    w, b = init_parameters(n_feats, hidden_layer_sizes, rng)
    
    if verb>1:
            ptrace(f"Initial error {mlp_error(data, target, w ,b)}")

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
            for l in range(1,L+1) :
                for i in range(len(a[l])) :
                    for j in range(len(a[l-1])) :
                        dw = err[l][i]* a[l-1][j]
                        w[l][i,j] = w[l][i,j] - learning_rate * dw

                    db = err[l][i]
                    b[l][i] = b[l][i] - learning_rate * db
                
            
                     
        if verb>1:
            ptrace(f"End of {epoch=}, new error {mlp_error(data, target, w ,b)}")

    # end for epoch
    
    return w, b


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    iris = load_iris()

    #obj_indices = [0,1,2,50,51,52]
    obj_indices = list(range(100))

    data = iris.data[iris.target<2][obj_indices]
    #data = stats.zscore(data)
    #print(data.mean(axis=0))
    
    target = iris.target[iris.target<2][obj_indices]

    #mlp_fit(data, target, n_epochs=100, hidden_layer_sizes=[4,2],
    #        learning_rate=0.2, random_state=42, verb=2)
    mlp_fit(data, target, n_epochs=10000, hidden_layer_sizes=[4,2],
            learning_rate=0.2, random_state=42, verb=2)


