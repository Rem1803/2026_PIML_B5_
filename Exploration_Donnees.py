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

    # --- La fameuse "fonction dans la fonction" ---
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
# SCRIPT PRINCIPAL (Exécution)
# ==============================================================================================

if __name__ == "__main__":
    # 1. Définition des chemins
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
    print("Chargement et transformation des images en cours (HSV)...")
    # On commence avec 500 images par classe pour tester rapidement, on peut augmenter ensuite !
    X, y = load_images_hsv(uninfected_path, parasitized_path, max_per_class=500)
    print(f"Données chargées : {X.shape[0]} images, {X.shape[1]} features (pixels) par image.")
    
    # ---------------------------------------------------------
    # PHASE 3 : Validation de la séparation (PCA)
    # ---------------------------------------------------------
    print("Génération de l'analyse PCA...")
    # Note : Le graphique de variance sera plat car X est déjà standardisé (Variance = 1), c'est normal !
    plot_advanced_eda(X, y)
    
    # ---------------------------------------------------------
    # PHASE 4 : Entraînement du MLP
    # ---------------------------------------------------------
    # C'est ici que nous allons appeler la cross-validation et le modèle !
    # Exemple à venir :
    # resultats = cross_validation(X, y, mlp_fit_minibatch, predict, n_folds=5)