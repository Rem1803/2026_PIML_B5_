"""
Chargement des données, implémentation du modèle MLP Mini-batch gradient descent avec activations ReLU, fonctions de coût,
fonctions de prédiction associées, cross-validation pour évaluer les modèles, sauvegarde des modèles entraînés.
"""
import Module.Pre_traitement as pre_traitement
import Module.Evaluation as eval

import os
import random
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import shutil
from matplotlib.colors import rgb_to_hsv
from PIL import Image
from sklearn.decomposition import PCA

# =======================
# CONFIGURATION 
# =======================

TAILLE_IMAGE = (32, 32)    
MAX_IMAGES = 1000           
TAUX_APPRENTISSAGE = 0.01  
BATCH_SIZE = 32
EPOCHS = 150              
PATIENCE = 5      
SEUIL_DECISION = 0.35
TAUX_DROPOUT = 0.2   
LAMBDA_L2 = 0.0001   # Régularisation L2
COMPOSANTES_PCA = 50       

# =======================
# Chemins des dossiers 
# =======================

UNINFECTED_PATH = r"Data\Uninfected"
PARASITIZED_PATH = r"Data\Parasitized"

# =======================
# Fonctions utilitaires
# =======================

def load_images(uninfected_dir, parasitized_dir, image_size=(32, 32), max_per_class=1000):
    """
    Charge les images et extrait les features pour les deux classes (uninfected et parasitized).
    Retourne X (features) et y (labels).
    """
    data = []
    target = []

    def process_folder(folder_path, label):
        count = 0
        for filename in sorted(os.listdir(folder_path)): #trie des images pour la reproductibilité
            if filename.endswith(".png"): #pour éviter les fichiers non image
                path = os.path.join(folder_path, filename)
                combined_features = pre_traitement.transformer_image_en_features(path, image_size) #pour récupérer les features de l'image
                data.append(combined_features)
                target.append(label)

                count += 1
                if count == max_per_class: break

    process_folder(uninfected_dir, 0)
    process_folder(parasitized_dir, 1)

    X_brut = np.array(data)
    y = np.array(target)
    
    return X_brut, y


def sigmoid(z):
    """Activation sigmoïde (sortie entre 0 et 1)."""
    z = np.clip(z, -500, 500) # pour éviter overflow
    return 1 / (1 + np.exp(-z))

def relu(z):
    """Activation ReLU (sortie positive)."""    
    return np.maximum(0, z)

def init_parameters(n_feats, hidden_layer_sizes, rng=None):
    """
    Initialise poids et biais du réseau de neurones avec He initialization pour ReLU et un léger bruit pour les biais.
    Arguments : - n_feats : nombre de features en entrée
                - hidden_layer_sizes : liste des tailles des couches cachées
    Retourne : listes de poids et biais pour chaque couche.
    """
    if rng is None: rng = np.random.default_rng() #utilisation d'un générateur de nombres aléatoires pour la reproductibilité
    L = 1 + len(hidden_layer_sizes)
    
    w = [np.zeros((0,0))]
    b = [np.zeros(0)]

    w.append(rng.normal(0, np.sqrt(2 / n_feats), (hidden_layer_sizes[0], n_feats))) # He initialization pour la première couche
    b.append(rng.normal(0, 0.01, size=hidden_layer_sizes[0]))  # ajout d'un léger bruit pour les biais pour éviter les symétries
    
    for l in range(2, L): #couches cachées 
        fan_in = hidden_layer_sizes[l-2] # nombre de neurones dans la couche précédente
        w.append(rng.normal(0, np.sqrt(2 / fan_in), (hidden_layer_sizes[l-1], fan_in))) 
        b.append(rng.normal(0, 0.01, size=hidden_layer_sizes[l-1]))

    fan_in = hidden_layer_sizes[L-2]
    w.append(rng.normal(0, np.sqrt(2 / fan_in), (1, fan_in)))
    b.append(rng.normal(0, 0.01, size=1))

    return w, b

def eval_forward_relu(x, w, b, dropout_rate=0.0, training=False, rng=None):
    """
    Propagation avant du MLP avec activations ReLU et option de Dropout.
    Arguments : - x : vecteur d'entrée
                - w, b : poids et biais du réseau
                - dropout_rate : taux de dropout
                - training : booléen indiquant si on est en phase d'entraînement (applique le dropout) ou d'inférence (pas de dropout)
                - rng : générateur de nombres aléatoires pour le dropout
    Retourne : - a : liste des activations de chaque couche
                - masks : liste des masques de dropout appliqués à chaque couche (None si pas de dropout)
    """
    L = len(w) - 1
    a = [np.copy(x)] 
    masks = [None]

    for l in range(1, L+1):
        z_l = np.matmul(w[l], a[l-1]) + b[l] #calcul du pré-activation pour la couche l
        if l == L: #couche de sortie avec sigmoïde pour obtenir une probabilité
            a_l = sigmoid(z_l)  
            mask = None
        else:
            a_l = relu(z_l) #activation ReLU pour les couches cachées    
            
            if training and dropout_rate > 0.0: #application du dropout pendant l'entraînement
                mask = (rng.random(a_l.shape) > dropout_rate).astype(float)
                a_l = (a_l * mask) / (1.0 - dropout_rate)
            else:
                mask = None
                
        a.append(a_l)
        masks.append(mask)
        
    return a, masks

def mlp_error_entropy(data, target, w, b):
    """
    Calcule l'erreur d'entropie croisée (loss) moyenne sur l'ensemble des données.
    """
    E = 0
    epsilon = 1e-15
    for x in range(len(data)):
        a, _ = eval_forward_relu(data[x], w, b) 
        pred = np.clip(a[-1][0], epsilon, 1 - epsilon) #pour éviter log(0) qui est indéfini
        y = target[x]
        e = - (y * np.log(pred) + (1-y) * np.log(1-pred)) 
        E += e
    return E / len(data)

def mlp_fit_minibatch(data, target, n_epochs=20, hidden_layer_sizes=[32, 16],
                             learning_rate=0.01, batch_size=32, random_state=42, 
                             dropout_rate=0.2, patience=5, lambda_reg=0.0001):
    """
    Entraîne un MLP avec mini-batch gradient descent, activations ReLU, régularisation L2, et early stopping.
    Arguments : - data, target : données d'entraînement
                - n_epochs : nombre maximum d'époques
                - hidden_layer_sizes : liste des tailles des couches cachées
                - learning_rate : taux d'apprentissage
                - batch_size : taille des mini-batchs
                - random_state : graine pour la reproductibilité
                - dropout_rate : taux de dropout à appliquer pendant l'entraînement
                - patience : nombre d'époques sans amélioration avant d'arrêter l'entraînement
                - lambda_reg : coefficient de régularisation L2
    Retourne : - w, b : poids et biais du modèle entraîné
                - losses : liste des pertes (loss) à la fin de chaque époque
    """
    rng = np.random.default_rng(random_state) #générateur de nombres aléatoires pour la reproductibilité
    n_objs, n_feats = data.shape
    L = len(hidden_layer_sizes) + 1

    w, b = init_parameters(n_feats, hidden_layer_sizes, rng)
    losses = []
    
    # Variables pour l'early stopping
    best_loss = np.inf
    patience_counter = 0
    best_w = None
    best_b = None

    for epoch in range(n_epochs):
        indices = rng.permutation(n_objs)
        for start in range(0, n_objs, batch_size):
            batch_indices = indices[start:start + batch_size] #
            current_batch_size = len(batch_indices)

            grad_w = [np.zeros_like(w_l) for w_l in w]
            grad_b = [np.zeros_like(b_l) for b_l in b]

            for i in batch_indices:
                a, masks = eval_forward_relu(data[i], w, b, dropout_rate=dropout_rate, training=True, rng=rng)
                err = [None] * (L + 1)
                
                err[-1] = a[-1] - target[i] #erreur de la couche de sortie (probabilité prédite - label réel)
                
                for l in range(L-1, 0, -1):
                    da = (a[l] > 0).astype(float)
                    if masks[l] is not None:
                        da = (da * masks[l]) / (1.0 - dropout_rate) #ajustement du gradient pour le dropout
                    
                    err_next = err[l+1]
                    err[l] = np.matmul(w[l+1].T, err_next) * da #calcul de l'erreur pour la couche l en utilisant la règle de la chaîne et en tenant compte de l'activation ReLU et du dropout
                
                for l in range(1, L + 1):
                    grad_w[l] += np.outer(err[l], a[l-1]) #gradient de la perte par rapport aux poids de la couche l
                    grad_b[l] += err[l]

            for l in range(1, L + 1):
                # Mise à jour avec Régularisation L2 
                w[l] -= learning_rate * (grad_w[l] / current_batch_size + lambda_reg * w[l])
                b[l] -= learning_rate * (grad_b[l] / current_batch_size)

        # Calcul de l'erreur en fin d'époque 
        loss = mlp_error_entropy(data, target, w, b)
        losses.append(loss)
        
        # Early Stopping : on sauvegarde les meilleurs poids et biais si la loss s'améliore, sinon on incrémente le compteur de patience
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
            best_w = [w_i.copy() for w_i in w]
            best_b = [b_i.copy() for b_i in b]
        else:
            patience_counter += 1
        
        # Si le compteur de patience atteint le seuil, on arrête l'entraînement et on restaure les meilleurs poids et biais
        if patience_counter >= patience:
            print(f"    -> Early Stopping à l'époque {epoch+1} (Loss opt: {best_loss:.4f})")
            w = [w_i.copy() for w_i in best_w]
            b = [b_i.copy() for b_i in best_b]
            break

    # Si l'entraînement s'est terminé normalement sans atteindre le nombre maximum d'époques, on restaure les meilleurs poids et biais trouvés pendant l'entraînement
    if patience_counter < patience:
        w = [w_i.copy() for w_i in best_w]
        b = [b_i.copy() for b_i in best_b]

    return w, b, losses

def predict_relu(x, w, b, seuil=0.35):
    """ 
    Prédit la classe pour un échantillon donné en utilisant le modèle MLP avec activation ReLU.
    Arguments : - x : vecteur d'entrée
                - w, b : poids et biais du modèle
                - seuil : seuil de décision pour classer en 0 ou 1
    Retourne : - 1 si la probabilité prédite est supérieure ou égale au seuil, sinon 0
    """
    proba = eval_forward_relu(x, w, b)[-1][0]
    if proba is None: return 0 
    return 1 if proba >= seuil else 0


def cross_validation(data, target, train_func, predict_func, n_folds=5, learning_rate=0.01,
    batch_size=32, hidden_layer_sizes=[32,16], n_epochs=EPOCHS, random_state=42, seuil=0.5,
    dropout_rate=0.0):
    """
    Effectue une validation croisée k-fold pour évaluer les performances d'un modèle de MLP.
    Arguments : - data, target : données et labels
                - train_func : fonction d'entraînement du MLP
                - predict_func : fonction de prédiction du MLP
                - n_folds : nombre de folds pour la validation croisée
                - learning_rate, batch_size, hidden_layer_sizes, n_epochs : hyperparamètres pour l'entraînement
                - random_state : graine pour la reproductibilité
                - seuil : seuil de décision pour la classification
                - dropout_rate : taux de dropout à appliquer pendant l'entraînement
    Retourne : - un dictionnaire contenant la précision moyenne et la matrice de confusion globale
    """

    indices = np.arange(len(data))
    np.random.seed(random_state)
    np.random.shuffle(indices) # mélange des indices pour créer des folds aléatoires
    folds = np.array_split(indices, n_folds)

    accuracies = []
    mc_globale = {'VP': 0, 'VN': 0, 'FP': 0, 'FN': 0} #initialisation de la matrice de confusion globale pour accumuler les résultats de tous les folds

    n_pixels = data.shape[1] - 8 

    for k in range(n_folds):
        print(f"--- Fold {k+1}/{n_folds} ---")

        test_idx = folds[k]
        train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != k])

        X_train, y_train = data[train_idx], target[train_idx]
        X_test, y_test = data[test_idx], target[test_idx]

        mean_train = np.mean(X_train, axis=0)
        std_train = np.std(X_train, axis=0)
        std_train = np.where(std_train < 1e-8, 1e-8, std_train)

        X_train_scaled = (X_train - mean_train) / std_train
        X_test_scaled = (X_test - mean_train) / std_train

        X_train_scaled = np.clip(X_train_scaled, -5, 5)
        X_test_scaled = np.clip(X_test_scaled, -5, 5)

        X_train_pixels, X_train_expert = X_train_scaled[:, :n_pixels], X_train_scaled[:, n_pixels:]
        X_test_pixels, X_test_expert = X_test_scaled[:, :n_pixels], X_test_scaled[:, n_pixels:]

        pca = PCA(n_components=COMPOSANTES_PCA, random_state=random_state)
        X_train_pixels_pca = pca.fit_transform(X_train_pixels)
        X_test_pixels_pca = pca.transform(X_test_pixels)

        X_train_final = np.concatenate([X_train_pixels_pca, X_train_expert], axis=1)
        X_test_final = np.concatenate([X_test_pixels_pca, X_test_expert], axis=1)

        w, b, losses = train_func(
            X_train_final, y_train,
            n_epochs=n_epochs,
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate=learning_rate,
            batch_size=batch_size,
            random_state=random_state,
            dropout_rate=dropout_rate,
            patience=PATIENCE,
            lambda_reg=LAMBDA_L2
        )

        y_pred = [predict_func(x, w, b, seuil) for x in X_test_final]

        mc_fold = eval.matrice_confusion(y_test, y_pred)
        mc_globale['VP'] += mc_fold['VP']
        mc_globale['VN'] += mc_fold['VN']
        mc_globale['FP'] += mc_fold['FP']
        mc_globale['FN'] += mc_fold['FN']

        acc = eval.exactitude(mc_fold)
        accuracies.append(acc)

        print(f"Accuracy fold {k+1} : {acc:.4f} | Loss : {losses[-1]:.4f}")

    print(f"\nAccuracy moyenne : {np.mean(accuracies):.4f}")

    return {
        "mean": np.mean(accuracies),
        "confusion_matrix": mc_globale
    }

def random_search_hyperparameters(data, target, train_func, predict_func, hidden_layer_configs,
    batch_sizes, learning_rate_range=(0.001,0.05), n_trials=8, random_state=42):
    """
    Effectue une recherche aléatoire d'hyperparamètres pour le MLP en utilisant la validation croisée.
    Arguments : - data, target : données et labels
                - train_func, predict_func : fonctions d'entraînement et de prédiction
                - hidden_layer_configs, batch_sizes : configurations possibles pour les couches cachées et la taille des batches
                - learning_rate_range : plage de valeurs pour le taux d'apprentissage
                - n_trials : nombre de configurations à tester
                - random_state : graine pour la reproductibilité
    Retourne : - la meilleure configuration trouvée et un résumé de tous les résultats
    """

    rng = np.random.default_rng(random_state)

    results = []
    best_result = None

    # SPEED SETTINGS
    n_folds = 3          # rapide
    n_epochs = 60        # suffisant avec early stopping
    prune_threshold = 0.55  # un peu plus permissif

    for trial in range(n_trials):

        hidden_layers = hidden_layer_configs[
            rng.integers(len(hidden_layer_configs))
        ]

        batch_size = batch_sizes[
            rng.integers(len(batch_sizes))
        ]

        learning_rate = float(
            np.exp(
                rng.uniform(
                    np.log(learning_rate_range[0]),
                    np.log(learning_rate_range[1])
                )
            )
        )

        print("\n====================")
        print(f"TRIAL {trial+1}/{n_trials}")
        print("====================")
        print(f"hidden_layers : {hidden_layers}")
        print(f"batch_size    : {batch_size}")
        print(f"learning_rate : {learning_rate:.6f}")

        #  CROSS VALIDATION
        cv_result = cross_validation(
            data,
            target,
            train_func=train_func,
            predict_func=predict_func,
            n_folds=n_folds,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            hidden_layer_sizes=hidden_layers,
            random_state=random_state
        )

        mean_acc = cv_result["mean"]

        print(f"→ accuracy = {mean_acc:.4f}")

        # pruning simple
        if mean_acc < prune_threshold:
            print("PRUNED")
            continue

        result = {
            "hidden_layers": hidden_layers,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "mean_accuracy": mean_acc
        }

        results.append(result)

        if best_result is None or mean_acc > best_result["mean_accuracy"]:
            best_result = result

    print("\n BEST CONFIG:")
    print(best_result if best_result else "Aucune config correcte trouvée")

    return best_result, results

def save_model(filename, w, b, mean_train, std_train, pca_comp, pca_mean):
    """
    Sauvegarde les paramètres du modèle (poids, biais, normalisation, PCA) dans un fichier .npz compressé.
    Arguments : - filename : nom du fichier de sauvegarde
                - w, b : poids et biais du modèle
                - mean_train, std_train : paramètres de normalisation des données
                - pca_comp, pca_mean : composantes et moyenne pour la PCA
    """
    np.savez(filename, 
             w=np.array(w, dtype=object), 
             b=np.array(b, dtype=object), 
             mean=mean_train, 
             std=std_train,
             pca_comp=pca_comp,
             pca_mean=pca_mean)

def load_model(filename):
    """
    Charge les paramètres du modèle à partir d'un fichier .npz.
    """
    params = np.load(filename, allow_pickle=True)
    return list(params["w"]), list(params["b"]), params["mean"], params["std"], params["pca_comp"], params["pca_mean"]
