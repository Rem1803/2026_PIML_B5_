# ==============================================================================================
# Etape 1 : Imports et Configuration Globale
# ==============================================================================================

import os
import random
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
from PIL import Image
from sklearn.decomposition import PCA

# --- CONFIGURATION ---
TAILLE_IMAGE = (32, 32)    
MAX_IMAGES = 1000           # Nombre d'images max à charger par classe
TAUX_APPRENTISSAGE = 0.01  
BATCH_SIZE = 32
EPOCHS = 40
SEUIL_DECISION = 0.5  # Seuil de classification (ajustable pour optimiser précision/rappel)

# Chemins des dossiers (à adapter si besoin)
UNINFECTED_PATH = r"C:\Users\noevm\Documents\INSA_Lyon\3A_2025-2026\S2\UE3_PIML\Projet\cell_images\Uninfected"
PARASITIZED_PATH = r"C:\Users\noevm\Documents\INSA_Lyon\3A_2025-2026\S2\UE3_PIML\Projet\cell_images\Parasitized"


# ==============================================================================================
# Etape 2 : Exploration Visuelle (EDA)
# ==============================================================================================

def plot_image_grid(uninfected_dir, parasitized_dir, n_images_per_class=8):
    uninfected_files = [f for f in os.listdir(uninfected_dir) if f.endswith('.png')]
    parasitized_files = [f for f in os.listdir(parasitized_dir) if f.endswith('.png')]
    
    s_uninfected = random.sample(uninfected_files, min(n_images_per_class, len(uninfected_files)))
    s_parasitized = random.sample(parasitized_files, min(n_images_per_class, len(parasitized_files)))
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle('Comparaison : Globules Sains (Gauche) vs Infectés (Droite)', fontsize=16)
    
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            ax.axis('off')
            if j < 2:
                idx = i * 2 + j
                if idx < len(s_uninfected):
                    img = Image.open(os.path.join(uninfected_dir, s_uninfected[idx]))
                    ax.imshow(img)
                    if i == 0 and j == 0: ax.set_title("Sain")
            else:
                idx = i * 2 + (j - 2)
                if idx < len(s_parasitized):
                    img = Image.open(os.path.join(parasitized_dir, s_parasitized[idx]))
                    ax.imshow(img)
                    if i == 0 and j == 2: ax.set_title("Infecté (Parasité)")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

def plot_color_histograms(uninfected_dir, parasitized_dir, sample_size=100):
    uninfected_files = [f for f in os.listdir(uninfected_dir) if f.endswith('.png')]
    parasitized_files = [f for f in os.listdir(parasitized_dir) if f.endswith('.png')]
    
    s_uninfected = random.sample(uninfected_files, min(sample_size, len(uninfected_files)))
    s_parasitized = random.sample(parasitized_files, min(sample_size, len(parasitized_files)))
    
    hue_uninfected, blue_uninfected = [], []
    hue_parasitized, blue_parasitized = [], []
    
    for f in s_uninfected:
        img = np.array(Image.open(os.path.join(uninfected_dir, f)).convert('RGB')) / 255.0
        hsv_img = rgb_to_hsv(img)
        hue_uninfected.extend(hsv_img[:, :, 0].flatten())
        blue_uninfected.extend(img[:, :, 2].flatten()) 
        
    for f in s_parasitized:
        img = np.array(Image.open(os.path.join(parasitized_dir, f)).convert('RGB')) / 255.0
        hsv_img = rgb_to_hsv(img)
        hue_parasitized.extend(hsv_img[:, :, 0].flatten())
        blue_parasitized.extend(img[:, :, 2].flatten()) 

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(blue_uninfected, bins=50, alpha=0.5, label='Sain', color='green', density=True)
    ax1.hist(blue_parasitized, bins=50, alpha=0.5, label='Infecté', color='red', density=True)
    ax1.set_title("Distribution du canal Bleu (RGB)")
    ax1.legend()
    
    ax2.hist(hue_uninfected, bins=50, alpha=0.5, label='Sain', color='green', density=True)
    ax2.hist(hue_parasitized, bins=50, alpha=0.5, label='Infecté', color='red', density=True)
    ax2.set_title("Distribution de la Teinte (HSV)")
    ax2.legend()
    plt.tight_layout()
    plt.show()

def plot_class_balance(uninfected_dir, parasitized_dir):
    n_uninfected = len([f for f in os.listdir(uninfected_dir) if f.endswith('.png')])
    n_parasitized = len([f for f in os.listdir(parasitized_dir) if f.endswith('.png')])
    
    classes = ['Saines', 'Infectées']
    counts = [n_uninfected, n_parasitized]
    
    plt.figure(figsize=(6, 5))
    bars = plt.bar(classes, counts, color=['#2ca02c', '#d62728'], alpha=0.8)
    plt.title("Équilibre des classes dans le Dataset", fontsize=14)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(counts)*0.02), 
                 int(yval), ha='center', va='bottom', fontweight='bold')
    plt.ylim(0, max(counts) * 1.15) 
    plt.show()

def plot_advanced_eda(data, target):
    mean_data = np.mean(data, axis=0)
    std_data = np.std(data, axis=0)
    std_data[std_data == 0] = 1 # Sécurité pour éviter la division par zéro
    
    data_scaled = (data - mean_data) / std_data
    
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)
    
    pca_saines = data_pca[target == 0]
    pca_infectees = data_pca[target == 1]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_saines[:, 0], pca_saines[:, 1], alpha=0.5, label='Saines', edgecolors='none', color='green')
    plt.scatter(pca_infectees[:, 0], pca_infectees[:, 1], alpha=0.5, label='Infectées', edgecolors='none', color='red')
    plt.title("Projection PCA (Espace à 2 Dimensions)")
    plt.xlabel(f"Composante Principale 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"Composante Principale 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ==============================================================================================
# Etape 3 : Pipeline de Données et Extraction de Caractéristiques Avancées
# ==============================================================================================

def extract_advanced_features(arr_rgb, arr_hsv):
    hue = arr_hsv[:, :, 0]
    saturation = arr_hsv[:, :, 1]
    blue_channel = arr_rgb[:, :, 2]
    
    variance = np.var(blue_channel)
    purple_mask = (hue > 0.75) & (hue < 0.95)
    purple_proportion = np.mean(purple_mask) 
    mean_saturation = np.mean(saturation)
    
    hist, _ = np.histogram(blue_channel.flatten(), bins=256, range=(0, 1))
    entropy = stats.entropy(hist + 1e-9) 
    skewness = stats.skew(blue_channel.flatten())
    kurtosis = stats.kurtosis(blue_channel.flatten())
    
    return np.array([variance, purple_proportion, mean_saturation, entropy, skewness, kurtosis])


def load_images_hsv(uninfected_dir, parasitized_dir, image_size=(32, 32), max_per_class=200):
    data = []
    target = []

    def process_folder(folder_path, label):
        count = 0
        for filename in os.listdir(folder_path):
            if filename.endswith(".png"):
                path = os.path.join(folder_path, filename)
                img = Image.open(path).convert("RGB").resize(image_size)
                arr_rgb = np.array(img) / 255.0 
                
                arr_hsv = rgb_to_hsv(arr_rgb)
                advanced_feats = extract_advanced_features(arr_rgb, arr_hsv)
                arr_feature = arr_hsv[:, :, 0] * arr_hsv[:, :, 1]
                
                combined_features = np.concatenate([arr_feature.flatten(), advanced_feats])
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
    # np.clip évite l'overflow dans l'exponentielle
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
    b.append(np.zeros((hidden_layer_sizes[0])))
    
    for l in range(2, L):
        fan_in = hidden_layer_sizes[l-2]
        w.append(rng.normal(0, np.sqrt(2 / fan_in), (hidden_layer_sizes[l-1], fan_in)))
        b.append(np.zeros((hidden_layer_sizes[l-1])))

    fan_in = hidden_layer_sizes[L-2]
    w.append(rng.normal(0, np.sqrt(2 / fan_in), (1, fan_in)))
    b.append(np.zeros((1)))

    return w, b

def eval_forward_relu(x, w, b):
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

def mlp_error_entropy(data, target, w, b):
    '''Binary Cross Entropy Globale'''
    E = 0
    epsilon = 1e-15
    for x in range(len(data)):
        a = eval_forward_relu(data[x], w, b)
        pred = np.clip(a[-1][0], epsilon, 1 - epsilon)
        y = target[x]
        e = - (y * np.log(pred) + (1-y) * np.log(1-pred))
        E += e
    return E / len(data) # Moyenne de la loss

def mlp_fit_minibatch_ultime(data, target, n_epochs=20, hidden_layer_sizes=[16, 8],
                             learning_rate=0.01, batch_size=32, random_state=42):
    rng = np.random.default_rng(random_state)
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
                a = eval_forward_relu(data[i], w, b)
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
                w[l] -= learning_rate * (grad_w[l] / current_batch_size)
                b[l] -= learning_rate * (grad_b[l] / current_batch_size)

        loss = mlp_error_entropy(data, target, w, b)
        losses.append(loss)

    return w, b, losses

def predict_proba_relu(x, w, b):
    a = eval_forward_relu(x, w, b)
    return a[-1][0]

def predict_relu(x, w, b, seuil=0.5):
    proba = predict_proba_relu(x, w, b)
    return 1 if proba >= seuil else 0


# ==============================================================================================
# Etape 5 : Évaluation et Métriques
# ==============================================================================================

def cross_validation(data, target, train_func, predict_func, n_folds=5, learning_rate=0.01, random_state=42, seuil=0.5):
    indices = np.arange(len(data))
    np.random.seed(random_state) 
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    accuracies = []
    # On initialise une matrice de confusion globale pour les données de test
    mc_globale = {'VP': 0, 'VN': 0, 'FP': 0, 'FN': 0}

    for k in range(n_folds):
        print(f"--- Fold {k+1}/{n_folds} ---")
        test_idx = folds[k]
        train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != k])

        X_train, y_train = data[train_idx], target[train_idx]
        X_test, y_test = data[test_idx], target[test_idx]

        mean_train = np.mean(X_train, axis=0)
        std_train = np.std(X_train, axis=0)
        std_train[std_train == 0] = 1 
        
        X_train = (X_train - mean_train) / std_train
        X_test = (X_test - mean_train) / std_train

        w, b, losses = train_func(X_train, y_train, n_epochs=EPOCHS, hidden_layer_sizes=[32, 16], 
                                  learning_rate=learning_rate, batch_size=BATCH_SIZE, random_state=random_state)

        # Prédictions UNIQUEMENT sur le test set de ce fold
        y_pred = [predict_func(x, w, b, seuil) for x in X_test]
        
        # On calcule la matrice de ce fold et on l'ajoute à la globale
        mc_fold = matrice_confusion(y_test, y_pred)
        mc_globale['VP'] += mc_fold['VP']
        mc_globale['VN'] += mc_fold['VN']
        mc_globale['FP'] += mc_fold['FP']
        mc_globale['FN'] += mc_fold['FN']
        
        accuracy = exactitude(mc_fold)
        accuracies.append(accuracy)
        print(f"Accuracy du fold : {accuracy:.4f} | Loss finale : {losses[-1]:.4f}")

    print(f"\nAccuracy moyenne Cross-Validation : {np.mean(accuracies):.4f}")
    
    # On retourne la vraie matrice de confusion évaluée de manière impartiale
    return mc_globale

def matrice_confusion(y_vrai, y_predit):
    vp = sum((y_vrai[i] == 1 and y_predit[i] == 1) for i in range(len(y_vrai)))
    vn = sum((y_vrai[i] == 0 and y_predit[i] == 0) for i in range(len(y_vrai)))
    fp = sum((y_vrai[i] == 0 and y_predit[i] == 1) for i in range(len(y_vrai)))
    fn = sum((y_vrai[i] == 1 and y_predit[i] == 0) for i in range(len(y_vrai)))
    return {'VP': vp, 'VN': vn, 'FP': fp, 'FN': fn}

def exactitude(mc):
    total = mc['VP'] + mc['VN'] + mc['FP'] + mc['FN']
    return (mc['VP'] + mc['VN']) / total if total > 0 else 0.0

def precision(mc):
    return mc['VP'] / (mc['VP'] + mc['FP']) if (mc['VP'] + mc['FP']) > 0 else 0.0
    
def rappel(mc):
    return mc['VP'] / (mc['VP'] + mc['FN']) if (mc['VP'] + mc['FN']) > 0 else 0.0

def score_f1(prec, rap):
    return 2 * (prec * rap) / (prec + rap) if (prec + rap) > 0 else 0.0

def save_model(filename, w, b):
    np.savez(filename, w=np.array(w, dtype=object), b=np.array(b, dtype=object))

def load_model(filename):
    params = np.load(filename, allow_pickle=True)
    return list(params["w"]), list(params["b"])


# ==============================================================================================
# Etape 6 : SCRIPT PRINCIPAL (Exécution)
# ==============================================================================================

if __name__ == "__main__":
    
    # ---------------------------------------------------------
    # PHASE 1 : Exploration visuelle brute
    # ---------------------------------------------------------
    print("1. Génération des graphiques EDA...")
    plot_image_grid(UNINFECTED_PATH, PARASITIZED_PATH)
    plot_color_histograms(UNINFECTED_PATH, PARASITIZED_PATH)
    
    # ---------------------------------------------------------
    # PHASE 2 : Préparation des données pour le Modèle
    # ---------------------------------------------------------
    print(f"Chargement de {MAX_IMAGES} images par classe (Dimensions : {TAILLE_IMAGE})...")
    X, y = load_images_hsv(UNINFECTED_PATH, PARASITIZED_PATH, image_size=TAILLE_IMAGE, max_per_class=MAX_IMAGES)
    print(f"Données prêtes : {X.shape[0]} images, {X.shape[1]} variables par image.")
    
    # ---------------------------------------------------------
    # PHASE 3 : Validation de la séparation (PCA)
    # ---------------------------------------------------------
    # print("\nGénération de l'analyse PCA sur les features extraites...")
    # plot_advanced_eda(X, y)
    
    # ---------------------------------------------------------
    # PHASE 4 : Entraînement et Évaluation du MLP
    # ---------------------------------------------------------
    print("\nLancement de l'entraînement (Validation Croisée)...")
    
    vraie_matrice = cross_validation(
        X, y, 
        train_func=mlp_fit_minibatch_ultime, 
        predict_func=predict_relu, 
        n_folds=5, 
        learning_rate=TAUX_APPRENTISSAGE, 
        random_state=42,
        seuil=SEUIL_DECISION 
    )
    
    print("\n--- ÉVALUATION FINALE ---")
    print(f"Matrice de confusion globale : {vraie_matrice}")
    print(f"Exactitude (Accuracy) : {exactitude(vraie_matrice):.4f}")
    print(f"Précision             : {precision(vraie_matrice):.4f}")
    print(f"Rappel (Recall)       : {rappel(vraie_matrice):.4f}")
    print(f"Score F1              : {score_f1(precision(vraie_matrice), rappel(vraie_matrice)):.4f}")