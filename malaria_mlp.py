# ==============================================================================================
# Etape 1 : Imports et Configuration Globale
# ==============================================================================================

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
EPOCHS = 40
SEUIL_DECISION = 0.35
TAUX_DROPOUT = 0.2   

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
    plt.close()

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
    plt.close()

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
    plt.close()

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
    plt.close()

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
    
    # --- Analyse de la Texture (Filtre de Sobel) ---
    # 1. Conversion en niveaux de gris
    gray = np.mean(arr_rgb, axis=2)
    
    # 2. On ajoute une bordure virtuelle (padding) pour pouvoir calculer les bords
    p = np.pad(gray, 1, mode='edge')
    
    # 3. Convolution avec les noyaux de Sobel pour détecter les bords verticaux et horizontaux
    # Détection des lignes verticales (Noyau Sobel X)
    gx = (p[:-2, 2:] - p[:-2, :-2]) + 2 * (p[1:-1, 2:] - p[1:-1, :-2]) + (p[2:, 2:] - p[2:, :-2])
    
    # Détection des lignes horizontales (Noyau Sobel Y)
    gy = (p[:-2, :-2] - p[2:, :-2]) + 2 * (p[:-2, 1:-1] - p[2:, 1:-1]) + (p[:-2, 2:] - p[2:, 2:])
    
    gradient_mag = np.sqrt(gx**2 + gy**2)
    
    mean_gradient = np.mean(gradient_mag)
    std_gradient = np.std(gradient_mag)
    
    return np.array([variance, purple_proportion, mean_saturation, entropy, skewness, kurtosis, mean_gradient, std_gradient])


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

def eval_forward_relu(x, w, b, dropout_rate=0.0, training=False):
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
                mask = (np.random.rand(*a_l.shape) > dropout_rate).astype(float)
                a_l = (a_l * mask) / (1.0 - dropout_rate)
            else:
                mask = None
                
        a.append(a_l)
        masks.append(mask)
        
    return a, masks

def mlp_error_entropy(data, target, w, b):
    '''Binary Cross Entropy Globale'''
    E = 0
    epsilon = 1e-15
    for x in range(len(data)):
        a, _ = eval_forward_relu(data[x], w, b) 
        pred = np.clip(a[-1][0], epsilon, 1 - epsilon)
        y = target[x]
        e = - (y * np.log(pred) + (1-y) * np.log(1-pred))
        E += e
    return E / len(data)

def mlp_fit_minibatch_ultime(data, target, n_epochs=20, hidden_layer_sizes=[16, 8],
                             learning_rate=0.01, batch_size=32, random_state=42, dropout_rate=0.2):
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
                # --- MODIF ICI : On récupère a ET masks ---
                a, masks = eval_forward_relu(data[i], w, b, dropout_rate=dropout_rate, training=True)
                err = [None] * (L + 1)
                
                err[-1] = a[-1] - target[i]
                
                # Backpropagation rigoureuse
                for l in range(L-1, 0, -1):
                    # 1. Dérivée de base de ReLU
                    da = (a[l] > 0).astype(float)
                    
                    # 2. Application propre du masque du Dropout
                    if masks[l] is not None:
                        da = (da * masks[l]) / (1.0 - dropout_rate)
                    
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
    a, _ = eval_forward_relu(x, w, b) # <-- MODIF ICI (a, _)
    return a[-1][0]

def predict_relu(x, w, b, seuil=0.5):
    proba = predict_proba_relu(x, w, b)
    return 1 if proba >= seuil else 0


# ==============================================================================================
# Etape 5 : Évaluation et Métriques
# ==============================================================================================

def cross_validation(data, target, train_func, predict_func, n_folds=5, learning_rate=0.01, random_state=42, seuil=0.5, dropout_rate=0.0):
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
                                  learning_rate=learning_rate, batch_size=BATCH_SIZE, random_state=random_state, dropout_rate=dropout_rate)

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

def save_model(filename, w, b, mean_train, std_train):
    """Sauvegarde les paramètres et les stats de normalisation."""
    np.savez(filename, 
             w=np.array(w, dtype=object), 
             b=np.array(b, dtype=object), 
             mean=mean_train, 
             std=std_train)

def load_model(filename):
    """Charge les paramètres et les stats de normalisation."""
    params = np.load(filename, allow_pickle=True)
    return list(params["w"]), list(params["b"]), params["mean"], params["std"]

def diagnostiquer_une_cellule(image_path, w, b, mean_train, std_train, image_size=(32, 32), seuil=0.35):
    """Prend une image au hasard, la pré-traite, la classe et l'affiche."""
    if not os.path.exists(image_path):
        print(f"Erreur : Le fichier {image_path} n'existe pas.")
        return

    # 1. Charger l'image originale pour l'affichage visuel
    img_originale = Image.open(image_path).convert("RGB")
    
    # 2. Appliquer EXACTEMENT le même pré-traitement que le pipeline
    img_resized = img_originale.resize(image_size)
    arr_rgb = np.array(img_resized) / 255.0
    arr_hsv = rgb_to_hsv(arr_rgb)
    
    # Extraction des caractéristiques expertes
    advanced_feats = extract_advanced_features(arr_rgb, arr_hsv)
    arr_feature = arr_hsv[:, :, 0] * arr_hsv[:, :, 1]
    
    # Combinaison des descripteurs
    combined_features = np.concatenate([arr_feature.flatten(), advanced_feats])
    
    # 3. Normalisation OBLIGATOIRE avec les statistiques de l'entraînement
    combined_features_scaled = (combined_features - mean_train) / std_train
    
    # 4. Passer l'image dans le réseau de neurones
    proba = predict_proba_relu(combined_features_scaled, w, b)
    prediction = 1 if proba >= seuil else 0
    
    # 5. Visualisation graphique du résultat
    plt.figure(figsize=(6, 6))
    plt.imshow(img_originale)
    plt.axis('off')
    
    # Formatage du diagnostic médical
    if prediction == 1:
        statut = "INFECTÉE (Malaria)"
        couleur = "#d62728"  # Rouge
    else:
        statut = "SAINE (Non infectée)"
        couleur = "#2ca02c"  # Vert
        
    plt.title(f"Diagnostic IA : {statut}\nProbabilité d'infection : {proba*100:.2f}%", 
              color=couleur, fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.show()
    plt.close()

def trier_dossier_images(dossier_entree, w, b, mean_train, std_train, image_size=(32, 32), seuil=0.35):
    """
    Parcourt un dossier d'images inconnues, les analyse, et les trie
    dans deux sous-dossiers : 'Classifiees_Saines' et 'Classifiees_Infectees'.
    """
    if not os.path.exists(dossier_entree):
        print(f"Erreur : Le dossier d'entrée {dossier_entree} n'existe pas.")
        return

    # Création des dossiers de destination
    dossier_saines = os.path.join(dossier_entree, "Resultats_Saines")
    dossier_infectees = os.path.join(dossier_entree, "Resultats_Infectees")
    
    os.makedirs(dossier_saines, exist_ok=True)
    os.makedirs(dossier_infectees, exist_ok=True)

    fichiers = [f for f in os.listdir(dossier_entree) if f.lower().endswith('.png')]
    total = len(fichiers)
    
    if total == 0:
        print("Aucune image .png trouvée dans ce dossier.")
        return

    print(f"\nDébut du tri automatique de {total} images...")
    
    saines_count = 0
    infectees_count = 0

    for idx, filename in enumerate(fichiers):
        path_in = os.path.join(dossier_entree, filename)
        
        try:
            # 1. Pipeline de pré-traitement (Silencieux)
            img = Image.open(path_in).convert("RGB")
            img_resized = img.resize(image_size)
            arr_rgb = np.array(img_resized) / 255.0
            arr_hsv = rgb_to_hsv(arr_rgb)
            
            advanced_feats = extract_advanced_features(arr_rgb, arr_hsv)
            arr_feature = arr_hsv[:, :, 0] * arr_hsv[:, :, 1]
            combined_features = np.concatenate([arr_feature.flatten(), advanced_feats])
            
            # Normalisation avec les stats du modèle
            features_scaled = (combined_features - mean_train) / std_train
            
            # 2. Prédiction
            proba = predict_proba_relu(features_scaled, w, b)
            
            # 3. Copie dans le bon dossier
            if proba >= seuil:
                path_out = os.path.join(dossier_infectees, filename)
                infectees_count += 1
            else:
                path_out = os.path.join(dossier_saines, filename)
                saines_count += 1
                
            shutil.copy2(path_in, path_out)
            
            # Affichage de la progression
            if (idx + 1) % 50 == 0 or (idx + 1) == total:
                print(f"Progression : {idx + 1}/{total} images traitées...")
                
        except Exception as e:
            print(f"Erreur lors du traitement de {filename} : {e}")

    print("\n--- RAPPORT DE TRI ---")
    print(f"Total analysé : {total}")
    print(f"Détectées Saines    : {saines_count} -> copiées dans '{dossier_saines}'")
    print(f"Détectées Infectées : {infectees_count} -> copiées dans '{dossier_infectees}'")
  
   
# ==============================================================================================
# Etape 6 : SCRIPT PRINCIPAL (Exécution)
# ==============================================================================================

if __name__ == "__main__":
    
    # ---------------------------------------------------------
    # PHASE 1 : Exploration visuelle brute
    # ---------------------------------------------------------
    # print("1. Génération des graphiques EDA...")
    # plot_image_grid(UNINFECTED_PATH, PARASITIZED_PATH)
    # plot_color_histograms(UNINFECTED_PATH, PARASITIZED_PATH)
    
    # ---------------------------------------------------------
    # PHASE 2 : Préparation des données pour le Modèle
    # ---------------------------------------------------------
    print(f"Chargement de {MAX_IMAGES} images par classe...")
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
        seuil=SEUIL_DECISION,
        dropout_rate=TAUX_DROPOUT 
    )
    
    print("\n--- ÉVALUATION FINALE ---")
    print(f"Matrice de confusion globale : {vraie_matrice}")
    print(f"Exactitude (Accuracy) : {exactitude(vraie_matrice):.4f}")
    print(f"Précision             : {precision(vraie_matrice):.4f}")
    print(f"Rappel (Recall)       : {rappel(vraie_matrice):.4f}")
    print(f"Score F1              : {score_f1(precision(vraie_matrice), rappel(vraie_matrice)):.4f}")
    
    # ---------------------------------------------------------
    # ENTRAÎNEMENT DU MODÈLE FINAL (Sur 100% des données)
    # ---------------------------------------------------------
    print("\nEntraînement du modèle de production final...")
    
    # On calcule les stats globales sur TOUT le dataset pour la future démo
    mean_final = np.mean(X, axis=0)
    std_final = np.std(X, axis=0)
    std_final[std_final == 0] = 1
    X_scaled = (X - mean_final) / std_final
    
    w_final, b_final, _ = mlp_fit_minibatch_ultime(
        X_scaled, y, n_epochs=EPOCHS, hidden_layer_sizes=[32, 16],
        learning_rate=TAUX_APPRENTISSAGE, batch_size=BATCH_SIZE, random_state=42
    )
    
    # Sauvegarde du package complet
    save_model("modele_malaria_ultime.npz", w_final, b_final, mean_final, std_final)
    print("Modèle de production sauvegardé avec succès sous 'modele_malaria_ultime.npz'.")
    
    # ---------------------------------------------------------
    # INTERFACE UTILISATEUR (Mode Démo INTERACTIF)
    # ---------------------------------------------------------

    print("\n" + "="*50)
    print("MODE DÉMONSTRATION INTERACTIF")
    print("="*50)
    
    # Simulation du chargement futur 
    w_load, b_load, mean_load, std_load = load_model("modele_malaria_ultime.npz")
    
    while True:
        print("\nQue souhaitez-vous faire ?")
        print("1 - Analyser une SEULE image (Mode manuel)")
        print("2 - Trier un DOSSIER complet d'images (Mode automatique)")
        print("q - Quitter")
        
        choix = input("Votre choix : ")
        
        if choix.lower() == 'q':
            print("Fermeture du mode démonstration.")
            break
            
        elif choix == '1':
            chemin = input("Entrez le chemin de l'image (.png) : ").strip('"').strip("'")
            diagnostiquer_une_cellule(chemin, w_load, b_load, mean_load, std_load, 
                                      image_size=TAILLE_IMAGE, seuil=SEUIL_DECISION)
                                      
        elif choix == '2':
            dossier = input("Entrez le chemin du DOSSIER contenant les nouvelles images : ").strip('"').strip("'")
            trier_dossier_images(dossier, w_load, b_load, mean_load, std_load, 
                                 image_size=TAILLE_IMAGE, seuil=SEUIL_DECISION)
                                 
        else:
            print("Choix invalide, veuillez réessayer.")
