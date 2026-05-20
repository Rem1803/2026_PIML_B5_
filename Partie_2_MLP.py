import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import os

"""
Partie_2_MLP.py
Chargement des données, implémentation de plusieurs modèles de MLP (sigmoïde, ReLU, Binary Cross Entropy, Dropout, Mini-batch gradient), 
fonctions de prédiction associées, cross-validation pour évaluer les modèles, sauvegarde des modèles entraînés 
et visualtisation des courbes d'apprentissage.
"""
import Partie_1_Pre_Traitement as pre_traitement
import Partie_3_Evaluation as eval

import os
import random
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import shutil
from matplotlib.colors import rgb_to_hsv
from PIL import Image
from sklearn.decomposition import PCA

# --- CONFIGURATION ---
TAILLE_IMAGE = (32, 32)    
MAX_IMAGES = 1000           
TAUX_APPRENTISSAGE = 0.01  
BATCH_SIZE = 32
EPOCHS = 150               # Augmenté car l'Early Stopping arrêtera le modèle au bon moment
PATIENCE = 15              # Nombre d'époques sans amélioration avant arrêt
SEUIL_DECISION = 0.35
TAUX_DROPOUT = 0.2   
LAMBDA_L2 = 0.0001         # Régularisation L2
COMPOSANTES_PCA = 50       

# Chemins des dossiers (à adapter si besoin)
UNINFECTED_PATH = r"Uninfected"
PARASITIZED_PATH = r"Parasitized"

def load_images_hsv(uninfected_dir, parasitized_dir, image_size=(32, 32), max_per_class=200):
    data = []
    target = []

    def process_folder(folder_path, label):
        count = 0
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".png"):
                path = os.path.join(folder_path, filename)
                combined_features = transformer_image_en_features(path, image_size)
                data.append(combined_features)
                target.append(label)

                count += 1
                if count == max_per_class: break

    process_folder(uninfected_dir, 0)
    process_folder(parasitized_dir, 1)

    X_brut = np.array(data)
    y = np.array(target)
    
    return X_brut, y


# ==============================================================================================
# Etape 4 : Le Modèle Mathématique (MLP)
# ==============================================================================================

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def init_parameters(n_feats, hidden_layer_sizes, rng=None):
    if rng is None: rng = np.random.default_rng()
    L = 1 + len(hidden_layer_sizes)
    
    w = [np.zeros((0,0))]
    b = [np.zeros(0)]

    w.append(rng.normal(0, np.sqrt(2 / n_feats), (hidden_layer_sizes[0], n_feats)))
    b.append(rng.normal(0, 0.01, size=hidden_layer_sizes[0]))  # Bruit léger pour init (Amélioration 2)
    
    for l in range(2, L):
        fan_in = hidden_layer_sizes[l-2]
        w.append(rng.normal(0, np.sqrt(2 / fan_in), (hidden_layer_sizes[l-1], fan_in)))
        b.append(rng.normal(0, 0.01, size=hidden_layer_sizes[l-1]))

    fan_in = hidden_layer_sizes[L-2]
    w.append(rng.normal(0, np.sqrt(2 / fan_in), (1, fan_in)))
    b.append(rng.normal(0, 0.01, size=1))
    return w, b

def eval_forward_relu(x, w, b, dropout_rate=0.0, training=False, rng=None):
    L = len(w) - 1
    a = [np.copy(x)] 
    masks = [None]

    for l in range(1, L+1):
        z_l = np.matmul(w[l], a[l-1]) + b[l]
        if l == L:
            a_l = sigmoid(z_l)  
            mask = None
        else:
            a_l = relu(z_l)      
            
            if training and dropout_rate > 0.0:
                mask = (rng.random(a_l.shape) > dropout_rate).astype(float)
                a_l = (a_l * mask) / (1.0 - dropout_rate)
            else:
                mask = None
                
        a.append(a_l)
        masks.append(mask)
        
    return a, masks

def mlp_error_entropy(data, target, w, b):
    E = 0
    epsilon = 1e-15
    for x in range(len(data)):
        a, _ = eval_forward_relu(data[x], w, b) 
        pred = np.clip(a[-1][0], epsilon, 1 - epsilon)
        y = target[x]
        e = - (y * np.log(pred) + (1-y) * np.log(1-pred))
        E += e
    return E / len(data)

def mlp_fit_minibatch_ultime(data, target, n_epochs=20, hidden_layer_sizes=[32, 16],
                             learning_rate=0.01, batch_size=32, random_state=42, 
                             dropout_rate=0.2, patience=15, lambda_reg=0.0001):
    rng = np.random.default_rng(random_state)
    n_objs, n_feats = data.shape
    L = len(hidden_layer_sizes) + 1

    w, b = init_parameters(n_feats, hidden_layer_sizes, rng)
    losses = []
    
    # --- MÉCANIQUE D'EARLY STOPPING ---
    best_loss = np.inf
    patience_counter = 0
    best_w = None
    best_b = None

    for epoch in range(n_epochs):
        indices = rng.permutation(n_objs)
        for start in range(0, n_objs, batch_size):
            batch_indices = indices[start:start + batch_size]
            current_batch_size = len(batch_indices)

            grad_w = [np.zeros_like(w_l) for w_l in w]
            grad_b = [np.zeros_like(b_l) for b_l in b]

            for i in batch_indices:
                a, masks = eval_forward_relu(data[i], w, b, dropout_rate=dropout_rate, training=True, rng=rng)
                err = [None] * (L + 1)
                
                err[-1] = a[-1] - target[i]
                
                for l in range(L-1, 0, -1):
                    da = (a[l] > 0).astype(float)
                    if masks[l] is not None:
                        da = (da * masks[l]) / (1.0 - dropout_rate)
                    
                    err_next = err[l+1]
                    err[l] = np.matmul(w[l+1].T, err_next) * da
                
                for l in range(1, L + 1):
                    grad_w[l] += np.outer(err[l], a[l-1])
                    grad_b[l] += err[l]

            for l in range(1, L + 1):
                # Mise à jour avec Régularisation L2 (Amélioration 2)
                w[l] -= learning_rate * (grad_w[l] / current_batch_size + lambda_reg * w[l])
                b[l] -= learning_rate * (grad_b[l] / current_batch_size)

        # Calcul de l'erreur en fin d'époque (sans dropout)
        loss = mlp_error_entropy(data, target, w, b)
        losses.append(loss)
        
        # --- VÉRIFICATION EARLY STOPPING ---
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
            best_w = [w_i.copy() for w_i in w]
            best_b = [b_i.copy() for b_i in b]
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"    -> Early Stopping à l'époque {epoch+1} (Loss opt: {best_loss:.4f})")
            w = [w_i.copy() for w_i in best_w]
            b = [b_i.copy() for b_i in best_b]
            break

    # Si on n'a jamais déclenché le break, on s'assure de renvoyer le meilleur
    if patience_counter < patience:
        w = [w_i.copy() for w_i in best_w]
        b = [b_i.copy() for b_i in best_b]

    return w, b, losses

def predict_proba_relu(x, w, b):
    a, _ = eval_forward_relu(x, w, b)
    return a[-1][0]

def predict_relu(x, w, b, seuil=0.5):
    proba = predict_proba_relu(x, w, b)
    return 1 if proba >= seuil else 0


# ==============================================================================================
# Etape 5 : Évaluation, Métriques, et Gestion du Modèle
# ==============================================================================================

def cross_validation(data, target, train_func, predict_func, n_folds=5, learning_rate=0.01, random_state=42, seuil=0.5, dropout_rate=0.0):
    indices = np.arange(len(data))
    np.random.seed(random_state) 
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    accuracies = []
    mc_globale = {'VP': 0, 'VN': 0, 'FP': 0, 'FN': 0}

    n_pixels = data.shape[1] - 8

    for k in range(n_folds):
        print(f"--- Fold {k+1}/{n_folds} ---")
        test_idx = folds[k]
        train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != k])

        X_train, y_train = data[train_idx], target[train_idx]
        X_test, y_test = data[test_idx], target[test_idx]

        # 1. Normalisation Globale Sécurisée
        mean_train = np.mean(X_train, axis=0)
        std_train = np.std(X_train, axis=0)
        std_train = np.where(std_train < 1e-8, 1e-8, std_train) # Amélioration Sécurité 1
        
        X_train_scaled = (X_train - mean_train) / std_train
        X_test_scaled = (X_test - mean_train) / std_train
        
        # 2. Clipping pour écrêter les valeurs extrêmes (Amélioration Sécurité 2)
        X_train_scaled = np.clip(X_train_scaled, -5, 5)
        X_test_scaled = np.clip(X_test_scaled, -5, 5)

        # 3. Séparation Pixels / Features expertes
        X_train_pixels, X_train_expert = X_train_scaled[:, :n_pixels], X_train_scaled[:, n_pixels:]
        X_test_pixels, X_test_expert = X_test_scaled[:, :n_pixels], X_test_scaled[:, n_pixels:]

        # 4. ACP (PCA) Ciblée UNIQUEMENT sur les pixels
        pca = PCA(n_components=COMPOSANTES_PCA, random_state=random_state)
        X_train_pixels_pca = pca.fit_transform(X_train_pixels)
        X_test_pixels_pca = pca.transform(X_test_pixels)

        # 5. Recombinaison finale
        X_train_final = np.concatenate([X_train_pixels_pca, X_train_expert], axis=1)
        X_test_final = np.concatenate([X_test_pixels_pca, X_test_expert], axis=1)

        # 6. Entraînement et Prédiction
        w, b, losses = train_func(X_train_final, y_train, n_epochs=EPOCHS, hidden_layer_sizes=[32, 16], 
                                  learning_rate=learning_rate, batch_size=BATCH_SIZE, random_state=random_state, 
                                  dropout_rate=dropout_rate, patience=PATIENCE, lambda_reg=LAMBDA_L2)

        y_pred = [predict_func(x, w, b, seuil) for x in X_test_final]
        
        mc_fold = eval.matrice_confusion(y_test, y_pred)
        mc_globale['VP'] += mc_fold['VP']
        mc_globale['VN'] += mc_fold['VN']
        mc_globale['FP'] += mc_fold['FP']
        mc_globale['FN'] += mc_fold['FN']
        
        accuracy = eval.exactitude(mc_fold)
        accuracies.append(accuracy)
        print(f"Accuracy du fold : {accuracy:.4f} | Loss finale : {losses[-1]:.4f}")

    print(f"\nAccuracy moyenne Cross-Validation : {np.mean(accuracies):.4f}")
    return mc_globale

def random_search_hyperparameters(
    data,
    target,
    train_func,
    predict_func,
    hidden_layer_configs,
    batch_sizes,
    learning_rate_range=(1e-4, 1e-2),
    n_trials=10,
    random_state=42,
    verbose=True
):
    """
    Random Search RAPIDE avec :
    - 3-fold CV
    - epochs fixes
    - early stopping
    - pruning des mauvais modèles
    """

    rng = np.random.default_rng(random_state)

    results = []
    best_result = None

    # =========================
    # SPEED SETTINGS
    # =========================

    n_folds = 3
    n_epochs = 40
    prune_threshold = 0.60

    # =========================
    # SUBSET RAPIDE
    # =========================

    max_samples = min(2000, len(data))

    subset_idx = rng.choice(
        len(data),
        size=max_samples,
        replace=False
    )

    data_small = data[subset_idx]
    target_small = target[subset_idx]

    for trial in range(n_trials):

        hidden_layers = hidden_layer_configs[
            rng.integers(len(hidden_layer_configs))
        ]

        batch_size = batch_sizes[
            rng.integers(len(batch_sizes))
        ]

        learning_rate = np.exp(
            rng.uniform(
                np.log(learning_rate_range[0]),
                np.log(learning_rate_range[1])
            )
        )

        if verbose:
            print(f"""
=========================
TRIAL {trial+1}/{n_trials}
=========================
hidden_layers = {hidden_layers}
batch_size    = {batch_size}
learning_rate = {learning_rate:.5f}
""")

        # =========================
        # CROSS VALIDATION
        # =========================

        cv_result = cross_validation(
            data_small,
            target_small,
            train_func=train_func,
            predict_func=predict_func,
            n_folds=n_folds,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            hidden_layer_sizes=hidden_layers,
            verbose=False,
            random_state=random_state
        )

        mean_acc = cv_result["mean"]

        # =========================
        # PRUNING
        # =========================

        if mean_acc < prune_threshold:

            if verbose:
                print(f"PRUNED : accuracy={mean_acc:.4f}")

            continue

        result = {
            "hidden_layers": hidden_layers,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "mean_accuracy": mean_acc
        }

        results.append(result)

        if verbose:
            print(f"✅ accuracy={mean_acc:.4f}")

        if (
            best_result is None
            or mean_acc > best_result["mean_accuracy"]
        ):
            best_result = result

    print("\nBEST CONFIG ")

    if best_result is not None:
        print(best_result)
    else:
        print("Aucun modèle correct trouvé.")

    return best_result, results

def save_model(filename, w, b, mean_train, std_train, pca_comp, pca_mean):
    np.savez(filename, 
             w=np.array(w, dtype=object), 
             b=np.array(b, dtype=object), 
             mean=mean_train, 
             std=std_train,
             pca_comp=pca_comp,
             pca_mean=pca_mean)

def load_model(filename):
    params = np.load(filename, allow_pickle=True)
    return list(params["w"]), list(params["b"]), params["mean"], params["std"], params["pca_comp"], params["pca_mean"]
