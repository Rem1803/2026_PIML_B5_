# ==============================================================================================
# Chargement des images 
# ==============================================================================================
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import os
from Partie_1_Pre_Traitement import *

def load_images(uninfected, parasitized, image_size=(16,16), max_per_class=200):
    data = []
    target = []

    count = 0
    for filename in os.listdir(uninfected):
        if filename.endswith(".png"):
            path = uninfected + "/" + filename

            img = Image.open(path).convert("RGB")
            img = img.resize(image_size)
            arr = np.array(img) / 255.0
            arr_gray = rgb_to_grayscale(arr)

            # Traitements des images
            arr_gray = equalize_histogram(arr_gray * 255.0) / 255.0
            arr_gray = standardize_image(arr_gray)

            data.append(arr_gray.flatten())
            target.append(0)

            count += 1
            if count == max_per_class:
                break

    count = 0
    for filename in os.listdir(parasitized):
        if filename.endswith(".png"):
            path = parasitized + "/" + filename

            img = Image.open(path).convert("RGB")
            img = img.resize(image_size)
            arr = np.array(img) / 255.0
            arr_gray = rgb_to_grayscale(arr)

            # Traitements ajoutés
            arr_gray = equalize_histogram(arr_gray * 255.0) / 255.0
            arr_gray = standardize_image(arr_gray)

            data.append(arr_gray.flatten())
            target.append(1)

            count += 1
            if count == max_per_class:
                break

    return np.array(data), np.array(target)

# ==============================================================================================
# Modèle simple avec la fonction sigmoïde
# ==============================================================================================

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
    
    losses = []

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

        loss = mlp_error(data, target, w, b)
        losses.append(loss)
    # end for epoch
    
    return w, b, losses

# ==============================================================================================
# Calcul des prédictions
# ==============================================================================================

def predict_proba(x, w, b):
    a = eval_forward(x, w, b)
    return a[-1][0]


def predict(x, w, b):
    proba = predict_proba(x, w, b)

    if proba >= 0.5:
        return 1
    else:
        return 0
    
# ==============================================================================================
# Modèle amélioré avec la fonction ReLu sur les couches cachées
# ==============================================================================================

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

# ==============================================================================================
# Calcul des nouvelles prédictions
# ==============================================================================================

def predict_proba_relu(x, w, b):
    a = eval_forward_relu(x, w, b)
    return a[-1][0]

def predict_relu(x, w, b):
    proba = predict_proba_relu(x, w, b)

    if proba >= 0.5:
        return 1
    else:
        return 0
    
# ==============================================================================================
# Modèle amélioré avec la Binary Cross Entropy à la place de la MSE
# ==============================================================================================

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

# ==============================================================================================
# Modèle amélioré avec Dropout
# ==============================================================================================

def eval_forward_dropout(x, w, b, dropout_rate=0.0, training=False):
    L = len(w) - 1
    a = [np.copy(x)]
    masks = [None]

    for l in range(1, L + 1):
        z_l = np.matmul(w[l], a[l-1]) + b[l]
        a_l = sigmoid(z_l)

        # Dropout seulement sur les couches cachées, pas sur la sortie
        if training and l < L and dropout_rate > 0:
            mask = (np.random.rand(*a_l.shape) > dropout_rate).astype(float)
            a_l = (a_l * mask) / (1 - dropout_rate)
        else:
            mask = None

        a.append(a_l)
        masks.append(mask)

    return a, masks

def mlp_error_dropout(data, target, w, b):
    '''Binary Cross Entropy'''
    E = 0
    epsilon = 1e-15
    
    for x in range(len(data)):
        a, masks = eval_forward_dropout(data[x], w, b, training=False)
        pred = a[-1][0]
        pred = np.clip(pred, epsilon, 1 - epsilon)
        y = target[x]
        e = - (y * np.log(pred) + (1-y) * np.log(1-pred))
        E += e

    return E

def mlp_fit_dropout(data, target, n_epochs=10, hidden_layer_sizes=[3,2], learning_rate=0.001,
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
            a,masks = eval_forward_dropout(data[x],w,b,dropout_rate=0.2,training=True)
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
        
        loss = mlp_error_dropout(data, target, w, b)
        losses.append(loss)
                            
    # end for epoch
    return w, b, losses

# ==============================================================================================
# Calcul des nouvelles prédictions 
# ==============================================================================================

def predict_proba_dropout(x, w, b):
    a,masks = eval_forward_dropout(x, w, b, training = False)
    return a[-1][0]

def predict_dropout(x, w, b):
    proba = predict_proba_dropout(x, w, b)

    if proba >= 0.5:
        return 1
    else:
        return 0
    
# ==============================================================================================
# Modèle amélioré avec mini-batch gradient descent
# ==============================================================================================
    
def mlp_fit_minibatch(data, target, n_epochs=10, hidden_layer_sizes=[3,2],
                      learning_rate=0.001, batch_size=32, random_state=None,
                      verb=0):
    """
    Entraine le MLP avec une descente de gradient par mini-batch.
    Les poids sont mis a jour apres chaque groupe de batch_size exemples.
    """
    if random_state != None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()

    n_objs, n_feats = data.shape
    L = len(hidden_layer_sizes) + 1

    w, b = init_parameters(n_feats, hidden_layer_sizes, rng)

    losses = []

    for epoch in range(n_epochs):
        indices = rng.permutation(n_objs)

        for start in range(0, n_objs, batch_size):
            batch_indices = indices[start:start + batch_size]
            current_batch_size = len(batch_indices)

            grad_w = [np.zeros_like(w_l) for w_l in w]
            grad_b = [np.zeros_like(b_l) for b_l in b]

            for i in batch_indices:
                a = eval_forward(data[i], w, b)

                err = [None] * (L + 1)
                da_output = a[-1] * (1 - a[-1])
                err[-1] = 2 * (a[-1] - target[i]) * da_output

                for l in range(L - 1, 0, -1):
                    da = a[l] * (1 - a[l])
                    err_next = err[l + 1]
                    err[l] = np.matmul(w[l + 1].T, err_next) * da

                for l in range(1, L + 1):
                    grad_w[l] += np.outer(err[l], a[l - 1])
                    grad_b[l] += err[l]

            for l in range(1, L + 1):
                w[l] -= learning_rate * grad_w[l] / current_batch_size
                b[l] -= learning_rate * grad_b[l] / current_batch_size

            loss = mlp_error(data, target, w, b)
            losses.append(loss)

    return w, b, losses
    

    
# ==========================================
# Sauvegarder et charger les modèles
# ==========================================  

def save_model(filename, w, b):
    np.savez(filename, w=np.array(w, dtype=object), b=np.array(b, dtype=object))


def load_model(filename):
    params = np.load(filename, allow_pickle=True)
    w = list(params["w"])
    b = list(params["b"])
    return w, b

# ==========================================
# Cross-validation
# ==========================================

def cross_validation(data, target,train_func, predict_func,  n_folds=5, learning_rate=0.001, random_state=42):
    indices = np.arange(len(data))
    np.random.seed(random_state) #pour la reproductibilité
    np.random.shuffle(indices)

    folds = np.array_split(indices, n_folds)

    accuracies = []

    for k in range(n_folds):

        print(f"\n===== Fold {k+1} =====")

        # fold de test
        test_idx = folds[k]

        # folds d'entraînement
        train_idx = np.concatenate(
            [folds[i] for i in range(n_folds) if i != k]
        )

        # séparation des données
        X_train = data[train_idx]
        y_train = target[train_idx]

        X_test = data[test_idx]
        y_test = target[test_idx]

        result = train_func(X_train, y_train, n_epochs=20, hidden_layer_sizes=[16, 8], learning_rate=learning_rate, random_state=random_state)

        w,b,losses = result
        correct = sum(predict_func(X_test[i], w, b) == y_test[i] for i in range(len(X_test)))
        accuracy = correct / len(X_test)
        accuracies.append(accuracy)
        print(f"Accuracy: {accuracy:.4f}")

        accuracies.append(accuracy)

    print(f"\nAccuracy moyenne: {np.mean(accuracies):.4f}")
    return {'accuracies': accuracies, 'mean': np.mean(accuracies), 'w': w, 'b': b, 'losses': losses}

# ==========================================
# Visualisation de l'évolution de la loss
# ==========================================

def plot_losses(losses, title="Evolution de la loss"):
    """
    Affiche le graphique de l'évolution de la loss.
    
    Parameters:
    losses (list): Liste des losses pour chaque époque.
    title (str): Titre du graphique.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()