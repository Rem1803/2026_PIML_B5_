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

# ==============================================================================================
# Chargement des images 
# ==============================================================================================

def load_images(
    uninfected,
    parasitized,
    image_size=(32, 32),
    max_per_class=1000,
    return_combined_features=False
):
    """
    Charge et prétraite les images de deux classes pour l'entraînement.

    Args:
        uninfected (str): chemin du dossier des images non infectées.
        parasitized (str): chemin du dossier des images infectées.
        image_size (tuple): taille de redimensionnement des images.
        max_per_class (int): nombre maximal d'images par classe.
        return_combined_features (bool): retourne aussi les features combinées.

    Returns:
        tuple:
            - data (np.ndarray): vecteurs de features.
            - target (np.ndarray): labels.
            - combined_features (optionnel)
    """

    images = []
    target = []

    # Chargement des images
    for folder, label in [(uninfected, 0), (parasitized, 1)]:

        count = 0

        for filename in os.listdir(folder):

            if filename.lower().endswith(".png"):

                path = os.path.join(folder, filename)

                img = Image.open(path).convert("RGB")
                arr = np.array(img)

                images.append(arr)
                target.append(label)

                count += 1

                if count >= max_per_class:
                    break

    # Prétraitement
    resized_images = pre_traitement.resize_images(
        images,
        target_size=image_size
    )

    hsv_images = pre_traitement.RGB_to_HSV(resized_images)

    combined_features_list = []

    # Extraction des features
    for rgb_img, hsv_img in zip(resized_images, hsv_images):

        arr_rgb = rgb_img.astype(np.float32) / 255.0

        # Features avancées
        advanced_feats = pre_traitement.extract_advanced_features(
            arr_rgb,
            hsv_img
        )

        # Sécurisation
        advanced_feats = np.asarray(
            advanced_feats,
            dtype=np.float32
        ).flatten()

        # Remplace les None éventuels
        advanced_feats = np.array([
            0 if feat is None else feat
            for feat in advanced_feats
        ], dtype=np.float32)

        # Remplace les Nan éventuels
        advanced_feats = np.nan_to_num(
        advanced_feats,
        nan=0.0,
        posinf=0.0,
        neginf=0.0)

        # Feature H*S
        arr_feature = hsv_img[:, :, 0] * hsv_img[:, :, 1]

        # Concaténation
        combined_features = np.concatenate([
            arr_feature.flatten(),
            advanced_feats
        ])

        combined_features_list.append(combined_features)

    # Conversion finale
    data = np.asarray(combined_features_list, dtype=np.float32)
    target = np.asarray(target, dtype=np.int32)

    if return_combined_features:
        return data, target, combined_features_list

    return data, target


# ==============================================================================================
# Modèle simple avec la fonction sigmoïde
# ==============================================================================================

def sigmoid(z):
    """Sigmoid activation function.

    Applies elementwise on scalars or NumPy arrays.

    Args:
        z: input value or array

    Returns:
        Sigmoid(z) computed elementwise.
    """
    # np.clip évite l'overflow dans l'exponentielle
    z = np.clip(z, -500, 500)
    return 1/(1 + np.exp(-z))



def init_parameters(n_feats, hidden_layer_sizes, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    L = len(hidden_layer_sizes) + 1

    w = [None]
    b = [None]

    prev = n_feats

    for h in hidden_layer_sizes:
        w.append(rng.normal(0, np.sqrt(2 / prev), (h, prev)))
        b.append(rng.normal(0, 0.01, size=h))
        prev = h

    w.append(rng.normal(0, np.sqrt(2 / prev), (1, prev)))
    b.append(rng.normal(0, 0.01, size=1))

    return w, b
# ==============================================================================================
# Modèle amélioré avec la fonction ReLu sur les couches cachées
# ==============================================================================================

def relu(z):
    return np.maximum(0,z)

def eval_forward(x, w, b):
    L = len(w) - 1
    a = [np.copy(x)] 

    for l in range(1, L+1):
        z_l = np.matmul(w[l], a[l-1]) + b[l]
        if l == L:
            a_l = sigmoid(z_l)   # Sortie
        else:
            a_l = relu(z_l)      # Couches cachées
        a.append(a_l)
    return a 


def mlp_error(data, target, w, b):
    '''Binary Cross Entropy Globale'''
    E = 0
    epsilon = 1e-15
    for x in range(len(data)):
        a = eval_forward(data[x], w, b)
        pred = np.clip(a[-1][0], epsilon, 1 - epsilon)
        y = target[x]
        e = - (y * np.log(pred) + (1-y) * np.log(1-pred))
        E += e
    return E / len(data) # Moyenne de la loss
 
# ==============================================================================================
# Modèle amélioré avec mini-batch gradient descent
# ==============================================================================================
    
def mlp_fit_minibatch_ultime(data, target, n_epochs=20, hidden_layer_sizes=[64, 32],
                             learning_rate=0.001, batch_size=32, random_state=42, patience=15):
    rng = np.random.default_rng(random_state)
    n_objs, n_feats = data.shape
    L = len(hidden_layer_sizes) + 1
    lambda_reg = 0.0001

    w, b = init_parameters(n_feats, hidden_layer_sizes, rng)
    losses = []
    best_loss = np.inf
    patience_counter = 0

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
                
                # Erreur en sortie (BCE + Sigmoïde)
                err[-1] = a[-1] - target[i]
                
                # Backpropagation (ReLU)
                for l in range(L-1, 0, -1):
                    da = (a[l] > 0).astype(float)
                    err_next = err[l+1]
                    err[l] = np.matmul(w[l+1].T, err_next) * da
                
                for l in range(1, L + 1):
                    grad_w[l] += np.outer(err[l], a[l-1])
                    grad_b[l] += err[l]

            for l in range(1, L + 1):
                w[l] -= learning_rate * (grad_w[l] / current_batch_size + lambda_reg * w[l])    
                b[l] -= learning_rate * (grad_b[l] / current_batch_size)

        loss = mlp_error(data, target, w, b)
        losses.append(loss)
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
            best_w = [w_i.copy() if w_i is not None else None for w_i in w]
            best_b = [b_i.copy() if b_i is not None else None for b_i in b]
        else:
            patience_counter += 1

        if patience_counter >= patience:
            w, b = best_w, best_b
            break

    return w, b, losses

# ==============================================================================================
# Calcul des prédictions
# ==============================================================================================

def predict_generic(x, w, b, seuil=0.35):
    proba = eval_forward(x, w, b)[-1][0]
    return 1 if proba >= seuil else 0

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

def cross_validation(data, target, train_func, predict_func, n_folds=5,
                     n_epochs=20, learning_rate=0.001, random_state=42, batch_size=32, patience=10,
                     hidden_layer_sizes=[64, 32], verbose=True, **train_kwargs):
    indices = np.arange(len(data))
    np.random.seed(random_state) #pour la reproductibilité
    np.random.shuffle(indices)

    folds = np.array_split(indices, n_folds)

    accuracies = []

    for k in range(n_folds):

        if verbose:
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

        #caclul des stats sur l'entrainement pour éviter data leakage
        mean_train = np.mean(X_train, axis=0)
        std_train = np.std(X_train, axis=0)
        std_train = np.where(std_train < 1e-8, 1e-8, std_train)

        X_train = (X_train - mean_train) / (std_train + 1e-8)
        X_test = (data[test_idx] - mean_train) / (std_train + 1e-8)

        X_train = np.clip(X_train, -5, 5)
        X_test = np.clip(X_test, -5, 5)

        y_test = target[test_idx]

        result = train_func(
            X_train,
            y_train,
            n_epochs=n_epochs,
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate=learning_rate,
            random_state=random_state,
            **train_kwargs
        )

        w,b,losses = result
        correct = sum(predict_func(X_test[i], w, b) == y_test[i] for i in range(len(X_test)))
        accuracy = correct / len(X_test)
        accuracies.append(accuracy)
        if verbose:
            print(f"Accuracy: {accuracy:.4f}")

    if verbose:
        print(f"\nAccuracy moyenne: {np.mean(accuracies):.4f}")
    return {'accuracies': accuracies, 'mean': np.mean(accuracies), 'w': w, 'b': b, 'losses': losses}


def random_search_hyperparameters(
    data, target,
    train_func, predict_func,
    hidden_layer_configs,
    batch_sizes,
    learning_rate_range=(0.001, 0.05),
    n_epochs_range=(30, 200),
    n_trials=10,
    n_folds=5,
    random_state=42,
    verbose=False,
    **train_kwargs
):
    rng = np.random.default_rng(random_state)

    results = []
    best_result = None

    for t in range(n_trials):

        hidden_layers = hidden_layer_configs[rng.integers(len(hidden_layer_configs))]
        batch_size = batch_sizes[rng.integers(len(batch_sizes))]

        learning_rate = 10 ** rng.uniform(
            np.log10(learning_rate_range[0]),
            np.log10(learning_rate_range[1])
        )

        n_epochs = rng.integers(n_epochs_range[0], n_epochs_range[1])

        if verbose:
            print(f"""
Trial {t+1}/{n_trials}
hidden_layers={hidden_layers}
batch_size={batch_size}
learning_rate={learning_rate:.5f}
n_epochs={n_epochs}
""")

        cv_result = cross_validation(
            data,
            target,
            train_func=train_func,
            predict_func=predict_func,
            n_folds=n_folds,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            hidden_layer_sizes=hidden_layers,
            batch_size=batch_size,
            random_state=random_state,
            verbose=False,
            **train_kwargs
        )

        result = {
            "hidden_layers": hidden_layers,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "n_epochs": n_epochs,
            "mean_accuracy": cv_result["mean"]
        }

        results.append(result)

        if best_result is None or result["mean_accuracy"] > best_result["mean_accuracy"]:
            best_result = result

    print("\n MEILLEURS PARAMÈTRES")
    print(best_result)

    return best_result, results
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
    
