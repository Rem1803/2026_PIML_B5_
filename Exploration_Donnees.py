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

# ==========================================
# Exemple d'exécution des fonctions d'exploration
# ==========================================
if __name__ == "__main__":
    uninfected_path = r"C:\Users\noevm\Documents\INSA_Lyon\3A_2025-2026\S2\UE3_PIML\Projet\cell_images\Uninfected"
    parasitized_path = r"C:\Users\noevm\Documents\INSA_Lyon\3A_2025-2026\S2\UE3_PIML\Projet\cell_images\Parasitized"
        
    print("1. Affichage de la matrice d'exemples...")
    plot_image_grid(uninfected_path, parasitized_path)
    
    print("2. Affichage des histogrammes de couleurs...")
    plot_color_histograms(uninfected_path, parasitized_path, sample_size=50)
    
    print("3. Affichage de l'équilibre des classes...")
    plot_class_balance(uninfected_path, parasitized_path)
    