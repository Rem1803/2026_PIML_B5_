import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
from PIL import Image

# ==============================================================================================
# Exploration et Compréhension des Données (EDA)
# ==============================================================================================

def plot_image_grid(uninfected_dir, parasitized_dir, n_images_per_class=8):
    """
    1. Matrice d'exemples : Affiche une grille (par défaut 4x4) comparant 
    des cellules saines et des cellules infectées.
    """
    # Récupérer une liste de fichiers aléatoires pour chaque classe
    uninfected_files = [f for f in os.listdir(uninfected_dir) if f.endswith('.png')]
    parasitized_files = [f for f in os.listdir(parasitized_dir) if f.endswith('.png')]
    
    s_uninfected = random.sample(uninfected_files, min(n_images_per_class, len(uninfected_files)))
    s_parasitized = random.sample(parasitized_files, min(n_images_per_class, len(parasitized_files)))
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle('Comparaison : Globules Sains (Gauche) vs Infectés (Droite)', fontsize=16)
    
    # Affichage des images (2 premières colonnes = sains, 2 dernières = infectés)
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            ax.axis('off')
            
            # Moitié gauche (sains)
            if j < 2:
                idx = i * 2 + j
                if idx < len(s_uninfected):
                    img_path = os.path.join(uninfected_dir, s_uninfected[idx])
                    img = Image.open(img_path)
                    ax.imshow(img)
                    if i == 0 and j == 0:
                        ax.set_title("Sain")
            # Moitié droite (infectés)
            else:
                idx = i * 2 + (j - 2)
                if idx < len(s_parasitized):
                    img_path = os.path.join(parasitized_dir, s_parasitized[idx])
                    img = Image.open(img_path)
                    ax.imshow(img)
                    if i == 0 and j == 2:
                        ax.set_title("Infecté (Parasité)")
                        
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


def plot_color_histograms(uninfected_dir, parasitized_dir, sample_size=100):
    """
    2. Histogrammes de distribution des couleurs : Analyse les canaux RGB et la 
    Teinte (Hue) en HSV pour mettre en évidence la coloration violette/bleue des parasites.
    """
    uninfected_files = [f for f in os.listdir(uninfected_dir) if f.endswith('.png')]
    parasitized_files = [f for f in os.listdir(parasitized_dir) if f.endswith('.png')]
    
    s_uninfected = random.sample(uninfected_files, min(sample_size, len(uninfected_files)))
    s_parasitized = random.sample(parasitized_files, min(sample_size, len(parasitized_files)))
    
    # Initialisation des listes pour stocker les valeurs
    hue_uninfected, blue_uninfected = [], []
    hue_parasitized, blue_parasitized = [], []
    
    # Extraction pour les cellules saines
    for f in s_uninfected:
        img = np.array(Image.open(os.path.join(uninfected_dir, f)).convert('RGB')) / 255.0
        hsv_img = rgb_to_hsv(img)
        hue_uninfected.extend(hsv_img[:, :, 0].flatten())
        blue_uninfected.extend(img[:, :, 2].flatten()) # Canal bleu
        
    # Extraction pour les cellules infectées
    for f in s_parasitized:
        img = np.array(Image.open(os.path.join(parasitized_dir, f)).convert('RGB')) / 255.0
        hsv_img = rgb_to_hsv(img)
        hue_parasitized.extend(hsv_img[:, :, 0].flatten())
        blue_parasitized.extend(img[:, :, 2].flatten()) # Canal bleu

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogramme du canal Bleu (RGB)
    ax1.hist(blue_uninfected, bins=50, alpha=0.5, label='Sain', color='green', density=True)
    ax1.hist(blue_parasitized, bins=50, alpha=0.5, label='Infecté', color='red', density=True)
    ax1.set_title("Distribution du canal Bleu (RGB)")
    ax1.set_xlabel("Intensité du Bleu")
    ax1.set_ylabel("Densité")
    ax1.legend()
    
    # Histogramme de la Teinte (HSV)
    ax2.hist(hue_uninfected, bins=50, alpha=0.5, label='Sain', color='green', density=True)
    ax2.hist(hue_parasitized, bins=50, alpha=0.5, label='Infecté', color='red', density=True)
    ax2.set_title("Distribution de la Teinte (HSV)")
    ax2.set_xlabel("Teinte (Hue)")
    ax2.set_ylabel("Densité")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def plot_class_balance(uninfected_dir, parasitized_dir):
    """
    3. Graphique à barres du balancement des classes : Vérifie si le dataset 
    est équilibré entre les images saines et infectées.
    """
    n_uninfected = len([f for f in os.listdir(uninfected_dir) if f.endswith('.png')])
    n_parasitized = len([f for f in os.listdir(parasitized_dir) if f.endswith('.png')])
    
    classes = ['Saines', 'Infectées']
    counts = [n_uninfected, n_parasitized]
    
    plt.figure(figsize=(6, 5))
    bars = plt.bar(classes, counts, color=['#2ca02c', '#d62728'], alpha=0.8)
    plt.title("Équilibre des classes dans le Dataset", fontsize=14)
    plt.ylabel("Nombre d'images")
    
    # Ajout des valeurs au-dessus des barres
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(counts)*0.02), 
                 int(yval), ha='center', va='bottom', fontweight='bold')
                 
    plt.ylim(0, max(counts) * 1.15) # Laisser un peu d'espace pour le texte
    plt.show()

   
from sklearn.decomposition import PCA
from matplotlib.colors import rgb_to_hsv

# ==============================================================================================
# Nouveau Chargement des images (Optimisé avec HSV + Stats globales)
# ==============================================================================================

import scipy.stats as stats

def extract_advanced_features(arr_rgb, arr_hsv):
    """
    Extrait 6 statistiques globales de l'image (Variance, Entropie, etc.)
    pour aider le réseau de neurones à comprendre la texture.
    """
    hue = arr_hsv[:, :, 0]
    saturation = arr_hsv[:, :, 1]
    blue_channel = arr_rgb[:, :, 2]
    
    # 1. Variance de l'image brute
    variance = np.var(blue_channel)
    
    # 2. Proportion de pixels violets (Teinte entre 0.75 et 0.95)
    purple_mask = (hue > 0.75) & (hue < 0.95)
    purple_proportion = np.mean(purple_mask) 
    
    # 3. Saturation moyenne
    mean_saturation = np.mean(saturation)
    
    # 4. Entropie
    hist, _ = np.histogram(blue_channel.flatten(), bins=256, range=(0, 1))
    entropy = stats.entropy(hist + 1e-9) 
    
    # 5. Skewness et Kurtosis
    skewness = stats.skew(blue_channel.flatten())
    kurtosis = stats.kurtosis(blue_channel.flatten())
    
    return np.array([variance, purple_proportion, mean_saturation, entropy, skewness, kurtosis])


def global_standardize(X):
    """
    Standardise l'ensemble du dataset (X) d'un seul coup.
    C'est la vraie méthode en Machine Learning.
    """
    mean_global = np.mean(X, axis=0)
    std_global = np.std(X, axis=0)
    
    # Éviter la division par zéro (si une zone de l'image est toujours noire sur toutes les photos)
    std_global[std_global == 0] = 1 
    
    X_standardized = (X - mean_global) / std_global
    return X_standardized


def load_images_hsv(uninfected_dir, parasitized_dir, image_size=(32, 32), max_per_class=200):
    """
    Charge les images, extrait les pixels HSV et les statistiques avancées.
    """
    data = []
    target = []

    # Sous-fonction pour traiter un dossier (sain ou infecté)
    def process_folder(folder_path, label):
        count = 0
        for filename in os.listdir(folder_path):
            if filename.endswith(".png"):
                path = os.path.join(folder_path, filename)

                img = Image.open(path).convert("RGB")
                img = img.resize(image_size) # On passe en 32x32 pour garder les détails !
                arr_rgb = np.array(img) / 255.0 
                
                # 1. Conversion RGB vers HSV
                arr_hsv = rgb_to_hsv(arr_rgb)
                
                # 2. Extraction des statistiques avancées AVANT de modifier l'image
                advanced_feats = extract_advanced_features(arr_rgb, arr_hsv)
                
                # 3. Création de la "Feature map" (Teinte * Saturation pour enlever le fond blanc)
                arr_feature = arr_hsv[:, :, 0] * arr_hsv[:, :, 1]
                
                # 4. Concaténation : On colle les pixels + les 6 stats à la fin
                combined_features = np.concatenate([arr_feature.flatten(), advanced_feats])
                
                # On ajoute à la liste globale
                data.append(combined_features)
                target.append(label)

                count += 1
                if count == max_per_class:
                    break
    # ----------------------------------------------

    # On utilise notre sous-fonction pour les deux dossiers !
    process_folder(uninfected_dir, 0) # Dossier Sain = Label 0
    process_folder(parasitized_dir, 1) # Dossier Infecté = Label 1

    # On transforme les listes Python en matrices mathématiques Numpy
    X_brut = np.array(data)
    y = np.array(target)
    
    # 5. Standardisation Globale
    X_final = global_standardize(X_brut)

    return X_final, y

def plot_advanced_eda(data, target):
    """
    Explore les données via l'analyse de texture (Variance) et la PCA.
    Note : 'data' doit être la matrice aplatie (X) et 'target' les labels (y) 
    retournés par votre fonction load_images_hsv.
    """
    # 1. Analyse de la Variance (Texture)
    # On calcule la variance pour chaque image (chaque ligne de 'data')
    variances = np.var(data, axis=1)
    
    var_saines = variances[target == 0]
    var_infectees = variances[target == 1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.boxplot([var_saines, var_infectees], labels=['Saines', 'Infectées'])
    ax1.set_title("Comparaison de la Texture (Variance des pixels)")
    ax1.set_ylabel("Variance")
    
    # 2. Analyse en Composantes Principales (PCA)
    # On réduit nos 256 pixels en seulement 2 variables (composantes)
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    
    # On sépare les points pour l'affichage
    pca_saines = data_pca[target == 0]
    pca_infectees = data_pca[target == 1]
    
    ax2.scatter(pca_saines[:, 0], pca_saines[:, 1], alpha=0.5, label='Saines', edgecolors='none')
    ax2.scatter(pca_infectees[:, 0], pca_infectees[:, 1], alpha=0.5, label='Infectées', edgecolors='none')
    ax2.set_title("Projection PCA (Espace à 2 Dimensions)")
    ax2.set_xlabel(f"Composante Principale 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax2.set_ylabel(f"Composante Principale 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


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

def save_model(model_or_filename, w=None, b=None):
    """Save model weights and biases to a file.

    Supports both calling styles:
      save_model(filename, w, b)
      save_model(model_dict, filename)
    """
    if isinstance(model_or_filename, dict):
        model_dict = model_or_filename
        if w is None:
            raise ValueError("save_model(model_dict, filename) requires a filename argument")
        filename = w
        w = model_dict.get("w")
        b = model_dict.get("b")
    else:
        filename = model_or_filename

    if w is None or b is None:
        raise ValueError("save_model(filename, w, b) requires both w and b")

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


# ==============================================================================================
# SCRIPT PRINCIPAL (Exécution)
# ==============================================================================================

if __name__ == "__main__":
    # ---------------------------------------------------------
    # CONFIGURATION (Modifiez ces variables selon vos besoins)
    # ---------------------------------------------------------
    TAILLE_IMAGE = (32, 32)    # Essayez (16, 16) ou (48, 48) plus tard !
    MAX_IMAGES = 500           # Nombre d'images par classe à charger
    TAUX_APPRENTISSAGE = 0.05  # Le learning_rate pour le MLP
    
    # Définition des chemins
    uninfected_path = r"C:\Users\noevm\Documents\INSA_Lyon\3A_2025-2026\S2\UE3_PIML\Projet\cell_images\Uninfected"
    parasitized_path = r"C:\Users\noevm\Documents\INSA_Lyon\3A_2025-2026\S2\UE3_PIML\Projet\cell_images\Parasitized"
        
    # ---------------------------------------------------------
    # PHASE 1 : Exploration visuelle brute (Optionnelle)
    # ---------------------------------------------------------
    print("1. Affichage de la matrice d'exemples...")
    plot_image_grid(uninfected_path, parasitized_path)
    plot_color_histograms(uninfected_path, parasitized_path, sample_size=50)
    plot_class_balance(uninfected_path, parasitized_path)
    
    # ---------------------------------------------------------
    # PHASE 2 : Préparation des données pour le Modèle
    # ---------------------------------------------------------
    print(f"Chargement de {MAX_IMAGES} images par classe en dimension {TAILLE_IMAGE}...")
    
    # On passe nos variables ici !
    X, y = load_images_hsv(uninfected_path, parasitized_path, image_size=TAILLE_IMAGE, max_per_class=MAX_IMAGES)
    
    print(f"Données chargées : {X.shape[0]} images, {X.shape[1]} features par image.")
    # Explication : X.shape[1] sera égal à (TAILLE_IMAGE[0] * TAILLE_IMAGE[1]) + 6 stats avancées
    
    # ---------------------------------------------------------
    # PHASE 3 : Validation de la séparation (PCA)
    # ---------------------------------------------------------
    print("Génération de l'analyse PCA...")
    plot_advanced_eda(X, y)
    
    # ---------------------------------------------------------
    # PHASE 4 : Entraînement et Évaluation du MLP
    # ---------------------------------------------------------
    print("\nLancement de la Validation Croisée (Entraînement du modèle)...")
    
    # Appel de la validation croisée (assurez-vous d'avoir bien copié vos fonctions MLP avant !)
    resultats = cross_validation(
        X, y, 
        train_func=mlp_fit_minibatch, 
        predict_func=predict, 
        n_folds=5, 
        learning_rate=TAUX_APPRENTISSAGE, 
        random_state=42
    )
    
    print("\n--- Évaluation globale avec les poids du dernier Fold ---")
    w_final = resultats['w']
    b_final = resultats['b']
    
    y_pred = [predict(x, w_final, b_final) for x in X]
    
    matrice = matrice_confusion(y, y_pred)
    print("Matrice de confusion :", matrice)
    print(f"Exactitude (Accuracy) : {exactitude(matrice):.4f}")