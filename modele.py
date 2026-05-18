import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import os

# ==============================================================================================
# Fonctions pour traitement des images 
# ==============================================================================================

def rgb_to_grayscale(image):
    """
    Convertit une image RGB en niveaux de gris.
    
    Parameters:
    image (numpy.ndarray): Image RGB à convertir.
    
    Returns:
    numpy.ndarray: Image en niveaux de gris.
    """
    # Vérifier si l'image est déjà en niveaux de gris
    if len(image.shape) == 2:
        return image
    
    # Utiliser la formule de luminosité pour convertir en niveaux de gris
    grayscale = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    
    return grayscale.astype(np.float64)



def standardize_image(image):
    """
    Normalise une image en soustrayant la moyenne et en divisant par l'écart type.
    
    Parameters:
    image (numpy.ndarray): Image à normaliser.
    
    Returns:
    numpy.ndarray: Image normalisée.
    """
    mean = np.mean(image)
    std = np.std(image)
    
    if std == 0:
        return image - mean  # Éviter la division par zéro
    
    standardized_image = (image - mean) / std
    return standardized_image



def equalize_histogram(image):
    """
    Égalise l'histogramme d'une image en niveaux de gris.
    
    Parameters:
    image (numpy.ndarray): Image en niveaux de gris à égaliser.
    
    Returns:
    numpy.ndarray: Image avec histogramme égalisé.
    """
    # Calculer l'histogramme de l'image
    histogram, bin_edges = np.histogram(image.flatten(), bins=256, range=(0, 255))
    
    # Calculer la fonction de distribution cumulative (CDF)
    cdf = histogram.cumsum()
    
    # Normaliser la CDF
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    
    # Appliquer la transformation d'égalisation
    equalized_image = np.interp(image.flatten(), bin_edges[:-1], cdf_normalized)
    
    return equalized_image.reshape(image.shape).astype(np.float64)

# ==============================================================================================
# Chargement des images 
# ==============================================================================================

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

    return w, b

    
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
# Évaluation des performances
# ==========================================

def matrice_confusion(y_vrai, y_predit):
    """
    Calcule la matrice de confusion pour une classification binaire.

    Paramètres :
    y_vrai (list) : Étiquettes réelles (0 ou 1).
    y_predit (list) : Étiquettes prédites (0 ou 1).

    Retourne :
    dict : Dictionnaire contenant les Vrais Positifs (VP), Vrais Négatifs (VN),
           Faux Positifs (FP) et Faux Négatifs (FN).
    """
    vp = sum((y_vrai[i] == 1 and y_predit[i] == 1) for i in range(len(y_vrai)))
    vn = sum((y_vrai[i] == 0 and y_predit[i] == 0) for i in range(len(y_vrai)))
    fp = sum((y_vrai[i] == 0 and y_predit[i] == 1) for i in range(len(y_vrai)))
    fn = sum((y_vrai[i] == 1 and y_predit[i] == 0) for i in range(len(y_vrai)))

    return {'VP': vp, 'VN': vn, 'FP': fp, 'FN': fn}

def exactitude(matrice_conf):
    """
    Calcule l'exactitude (accuracy) globale du modèle.
    """
    vp, vn = matrice_conf['VP'], matrice_conf['VN']
    fp, fn = matrice_conf['FP'], matrice_conf['FN']
    
    total = vp + vn + fp + fn
    return (vp + vn) / total if total > 0 else 0.0

def precision(matrice_conf):
    """
    Calcule la précision (pertinence des prédictions positives).
    """
    vp, fp = matrice_conf['VP'], matrice_conf['FP']
    return vp / (vp + fp) if (vp + fp) > 0 else 0.0
    
def rappel(matrice_conf):
    """
    Calcule le rappel (capacité à trouver tous les cas positifs).
    """
    vp, fn = matrice_conf['VP'], matrice_conf['FN']
    return vp / (vp + fn) if (vp + fn) > 0 else 0.0

def score_f1(valeur_precision, valeur_rappel):
    """
    Calcule le score F1 (moyenne harmonique entre précision et rappel).
    """
    somme = valeur_precision + valeur_rappel
    return 2 * (valeur_precision * valeur_rappel) / somme if somme > 0 else 0.0

# ==========================================
# Visualisation de la matrice
# ==========================================

def afficher_matrice_confusion(mc, classes, normaliser=False, titre='Matrice de Confusion', cmap=plt.cm.Blues):
    """
    Affiche graphiquement la matrice de confusion.
    """
    if normaliser:
        mc = mc.astype('float') / mc.sum(axis=1)[:, np.newaxis]
        print("Matrice de confusion normalisée")
    else:
        print('Matrice de confusion (valeurs brutes)')

    plt.imshow(mc, interpolation='nearest', cmap=cmap)
    plt.title(titre)
    plt.colorbar()
    
    marques = np.arange(len(classes))
    plt.xticks(marques, classes, rotation=45)
    plt.yticks(marques, classes)

    format_texte = '.2f' if normaliser else 'd'
    seuil = mc.max() / 2.
    
    for i in range(len(mc)):
        for j in range(len(mc[i])):
            plt.text(j, i, format(mc[i, j], format_texte),
                        horizontalalignment="center",
                        color="white" if mc[i, j] > seuil else "black")

    plt.ylabel('Réalité')
    plt.xlabel('Prédictions du Modèle')
    plt.tight_layout()

# ==========================================
# Courbe ROC et AUC
# ==========================================

def roc_auc(y_vrai, y_scores):
    """
    Calcule l'aire sous la courbe ROC (AUC) en utilisant la vectorisation NumPy.
    """
    y_vrai = np.asarray(y_vrai)
    y_scores = np.asarray(y_scores)
    
    scores_pos = y_scores[y_vrai == 1]
    scores_neg = y_scores[y_vrai == 0]
    
    if len(scores_pos) == 0 or len(scores_neg) == 0:
        raise ValueError("Il faut au moins un exemple de la classe 0 et de la classe 1.")
        
    diff = scores_pos[:, np.newaxis] - scores_neg
    
    paires_concordantes = np.sum(diff > 0)
    egalites = np.sum(diff == 0) * 0.5
    
    total_paires = len(scores_pos) * len(scores_neg)
    
    return (paires_concordantes + egalites) / total_paires

def courbe_roc(y_vrai, y_scores):
    """
    Calcule les points de la courbe ROC : Taux de Faux Positifs (FPR) 
    et Taux de Vrais Positifs (TPR) en fonction des seuils.
    """
    y_vrai = np.asarray(y_vrai)
    y_scores = np.asarray(y_scores)
    
    indices_tri = np.argsort(y_scores)[::-1]
    scores_tries = y_scores[indices_tri]
    vrai_tries = y_vrai[indices_tri]
    
    total_positifs = np.sum(y_vrai == 1)
    total_negatifs = np.sum(y_vrai == 0)
    
    tpr_liste = [0.0]
    fpr_liste = [0.0]
    seuils = [scores_tries[0] + 0.1] 
    
    vp, fp = 0, 0
    
    for i in range(len(scores_tries)):
        if vrai_tries[i] == 1:
            vp += 1
        else:
            fp += 1
            
        # N'enregistrer le point que si le prochain score est différent (gestion des ex-aequo)
        if i == len(scores_tries) - 1 or scores_tries[i] != scores_tries[i+1]:
            tpr_liste.append(vp / total_positifs)
            fpr_liste.append(fp / total_negatifs)
            seuils.append(scores_tries[i])
            
    return np.array(fpr_liste), np.array(tpr_liste), np.array(seuils)


# ==========================================
# Exemples d'utilisation
# ==========================================
if __name__ == "__main__":
    # --- Test ---
    y_reel = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    
    matrice = matrice_confusion(y_reel, y_pred)
    acc = exactitude(matrice)
    prec = precision(matrice)
    rec = rappel(matrice)
    f1 = score_f1(prec, rec)

    print(f"Matrice de confusion : {matrice}")
    print(f"Exactitude : {acc:.2f}")
    print(f"Précision  : {prec:.2f}")
    print(f"Rappel     : {rec:.2f}")
    print(f"Score F1   : {f1:.2f}\n")

    # --- Test de l'affichage de la matrice ---
    mc_exemple = np.array([[50, 10], [5, 35]])
    classes_noms = ['Cellules Saines', 'Cellules Malades']
    
    plt.figure(figsize=(6, 4))
    afficher_matrice_confusion(mc_exemple, classes_noms, normaliser=True)
    plt.show()

    # --- Test de la courbe ROC et AUC ---
    y_reel_roc = [0, 0, 1, 1]
    y_scores_roc = [0.1, 0.4, 0.35, 0.8]

    fpr, tpr, _ = courbe_roc(y_reel_roc, y_scores_roc)
    score_auc = np.trapezoid(tpr, fpr)
    
    print(f"Score ROC AUC calculé manuellement : {roc_auc(y_reel_roc, y_scores_roc):.3f}")

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'Courbe ROC (AUC = {score_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (FPR)')
    plt.ylabel('Taux de Vrais Positifs (TPR)')
    plt.title('Courbe ROC')
    plt.legend(loc="lower right")
    plt.show()
    
    plt.close("all")
