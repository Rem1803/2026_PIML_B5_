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
EPOCHS = 150               # Augmenté car l'Early Stopping arrêtera le modèle au bon moment
PATIENCE = 15              # Nombre d'époques sans amélioration avant arrêt
SEUIL_DECISION = 0.35
TAUX_DROPOUT = 0.2   
LAMBDA_L2 = 0.0001         # Régularisation L2
COMPOSANTES_PCA = 50       

# Chemins des dossiers (à adapter si besoin)
UNINFECTED_PATH = r"Uninfected"
PARASITIZED_PATH = r"Parasitized"



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
    gray = np.mean(arr_rgb, axis=2)
    p = np.pad(gray, 1, mode='edge')
    
    gx = (p[:-2, 2:] - p[:-2, :-2]) + 2 * (p[1:-1, 2:] - p[1:-1, :-2]) + (p[2:, 2:] - p[2:, :-2])
    gy = (p[:-2, :-2] - p[2:, :-2]) + 2 * (p[:-2, 1:-1] - p[2:, 1:-1]) + (p[:-2, 2:] - p[2:, 2:])
    
    gradient_mag = np.sqrt(gx**2 + gy**2)
    mean_gradient = np.mean(gradient_mag)
    std_gradient = np.std(gradient_mag)
    
    advanced_feats = np.array([variance, purple_proportion, mean_saturation, entropy, skewness, kurtosis, mean_gradient, std_gradient], dtype=np.float32)
    
    # Sécurité absolue : on remplace les NaNs et les Infinis par 0
    return np.nan_to_num(advanced_feats, nan=0.0, posinf=0.0, neginf=0.0)

def transformer_image_en_features(chemin_image, image_size):
    """Ouvre une image et la transforme en un vecteur prêt pour le réseau de neurones."""
    img = Image.open(chemin_image).convert("RGB")
    img_resized = img.resize(image_size)
    arr_rgb = np.array(img_resized) / 255.0
    arr_hsv = rgb_to_hsv(arr_rgb)
    
    advanced_feats = extract_advanced_features(arr_rgb, arr_hsv)
    arr_feature = arr_hsv[:, :, 0] * arr_hsv[:, :, 1]
    
    combined_features = np.concatenate([arr_feature.flatten(), advanced_feats])
    return combined_features

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
        
        mc_fold = matrice_confusion(y_test, y_pred)
        mc_globale['VP'] += mc_fold['VP']
        mc_globale['VN'] += mc_fold['VN']
        mc_globale['FP'] += mc_fold['FP']
        mc_globale['FN'] += mc_fold['FN']
        
        accuracy = exactitude(mc_fold)
        accuracies.append(accuracy)
        print(f"Accuracy du fold : {accuracy:.4f} | Loss finale : {losses[-1]:.4f}")

    print(f"\nAccuracy moyenne Cross-Validation : {np.mean(accuracies):.4f}")
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

def diagnostiquer_une_cellule(image_path, w, b, mean_train, std_train, pca_comp, pca_mean, image_size=(32, 32), seuil=0.35):
    if not os.path.exists(image_path):
        print(f"Erreur : Le fichier {image_path} n'existe pas.")
        return

    img_originale = Image.open(image_path).convert("RGB")
    
    combined_features = transformer_image_en_features(image_path, image_size)
    combined_features_scaled = (combined_features - mean_train) / std_train
    combined_features_scaled = np.clip(combined_features_scaled, -5, 5) # Clipping Inférence

    n_pixels = image_size[0] * image_size[1]
    pixels_scaled = combined_features_scaled[:n_pixels]
    expert_scaled = combined_features_scaled[n_pixels:]
    
    pixels_pca = np.dot(pixels_scaled - pca_mean, pca_comp.T)
    features_final = np.concatenate([pixels_pca, expert_scaled])

    proba = predict_proba_relu(features_final, w, b)
    prediction = 1 if proba >= seuil else 0
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img_originale)
    plt.axis('off')
    
    if prediction == 1:
        statut = "INFECTÉE (Malaria)"
        couleur = "#d62728"  
    else:
        statut = "SAINE (Non infectée)"
        couleur = "#2ca02c"  
        
    plt.title(f"Diagnostic IA : {statut}\nProbabilité d'infection : {proba*100:.2f}%", 
              color=couleur, fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.show()
    plt.close()

def trier_dossier_images(dossier_entree, w, b, mean_train, std_train, pca_comp, pca_mean, image_size=(32, 32), seuil=0.35):
    if not os.path.exists(dossier_entree):
        print(f"Erreur : Le dossier d'entrée {dossier_entree} n'existe pas.")
        return

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
    n_pixels = image_size[0] * image_size[1]

    for idx, filename in enumerate(fichiers):
        path_in = os.path.join(dossier_entree, filename)
        try:
            combined_features = transformer_image_en_features(path_in, image_size)
            combined_features_scaled = (combined_features - mean_train) / std_train
            combined_features_scaled = np.clip(combined_features_scaled, -5, 5)
            
            pixels_scaled = combined_features_scaled[:n_pixels]
            expert_scaled = combined_features_scaled[n_pixels:]
            
            pixels_pca = np.dot(pixels_scaled - pca_mean, pca_comp.T)
            features_final = np.concatenate([pixels_pca, expert_scaled])
            
            proba = predict_proba_relu(features_final, w, b)
            
            if proba >= seuil:
                path_out = os.path.join(dossier_infectees, filename)
                infectees_count += 1
            else:
                path_out = os.path.join(dossier_saines, filename)
                saines_count += 1
                
            shutil.copy2(path_in, path_out)
            
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
    # PHASE 2 : Préparation des données pour le Modèle
    # ---------------------------------------------------------
    print(f"Chargement de {MAX_IMAGES} images par classe (Taille dynamique : {TAILLE_IMAGE})...")
    X, y = load_images_hsv(UNINFECTED_PATH, PARASITIZED_PATH, image_size=TAILLE_IMAGE, max_per_class=MAX_IMAGES)
    print(f"Données brutes prêtes : {X.shape[0]} images, {X.shape[1]} variables par image.")
    
    # ---------------------------------------------------------
    # PHASE 4 : Entraînement et Évaluation du MLP
    # ---------------------------------------------------------
    print("\nLancement de l'entraînement (Validation Croisée + ACP Ciblée + Early Stopping)...")
    
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
    # ENTRAÎNEMENT DU MODÈLE FINAL
    # ---------------------------------------------------------
    print("\nEntraînement du modèle de production final...")
    
    mean_final = np.mean(X, axis=0)
    std_final = np.std(X, axis=0)
    std_final = np.where(std_final < 1e-8, 1e-8, std_final)
    X_scaled = (X - mean_final) / std_final
    X_scaled = np.clip(X_scaled, -5, 5)
    
    n_pixels = TAILLE_IMAGE[0] * TAILLE_IMAGE[1]
    X_pixels_scaled = X_scaled[:, :n_pixels]
    X_expert_scaled = X_scaled[:, n_pixels:]
    
    pca_finale = PCA(n_components=COMPOSANTES_PCA)
    X_pixels_pca = pca_finale.fit_transform(X_pixels_scaled)
    
    X_final = np.concatenate([X_pixels_pca, X_expert_scaled], axis=1)
    
    w_final, b_final, _ = mlp_fit_minibatch_ultime(
        X_final, y, n_epochs=EPOCHS, hidden_layer_sizes=[32, 16],
        learning_rate=TAUX_APPRENTISSAGE, batch_size=BATCH_SIZE, random_state=42, 
        dropout_rate=TAUX_DROPOUT, patience=PATIENCE, lambda_reg=LAMBDA_L2
    )
    
    save_model("modele_malaria_ultime.npz", w_final, b_final, mean_final, std_final, 
               pca_finale.components_, pca_finale.mean_)
    print(f"Modèle sauvegardé ('modele_malaria_ultime.npz'). Le réseau utilise désormais {X_final.shape[1]} variables au lieu de {X.shape[1]}.")
    
    # ---------------------------------------------------------
    # INTERFACE UTILISATEUR (Mode Démo INTERACTIF)
    # ---------------------------------------------------------

    print("\n" + "="*50)
    print("MODE DÉMONSTRATION INTERACTIF")
    print("="*50)
    
    w_load, b_load, mean_load, std_load, pca_comp, pca_mean = load_model("modele_malaria_ultime.npz")
    
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
            diagnostiquer_une_cellule(chemin, w_load, b_load, mean_load, std_load, pca_comp, pca_mean,
                                      image_size=TAILLE_IMAGE, seuil=SEUIL_DECISION)
                                      
        elif choix == '2':
            dossier = input("Entrez le chemin du DOSSIER contenant les nouvelles images : ").strip('"').strip("'")
            trier_dossier_images(dossier, w_load, b_load, mean_load, std_load, pca_comp, pca_mean,
                                 image_size=TAILLE_IMAGE, seuil=SEUIL_DECISION)
                                 
        else:
            print("Choix invalide, veuillez réessayer.")