import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from PIL import Image

import Partie_1_Pre_Traitement as pre_traitement
import Partie_2_MLP as mlp


def diagnostiquer_une_cellule(image_path, w, b, mean_train, std_train, pca_comp, pca_mean, image_size=(32, 32), seuil=0.35):
    if not os.path.exists(image_path):
        print(f"Erreur : Le fichier {image_path} n'existe pas.")
        return

    img_originale = Image.open(image_path).convert("RGB")
    
    combined_features = pre_traitement.transformer_image_en_features(image_path, image_size)
    combined_features_scaled = (combined_features - mean_train) / std_train
    combined_features_scaled = np.clip(combined_features_scaled, -5, 5) # Clipping Inférence

    n_pixels = image_size[0] * image_size[1]
    pixels_scaled = combined_features_scaled[:n_pixels]
    expert_scaled = combined_features_scaled[n_pixels:]
    
    pixels_pca = np.dot(pixels_scaled - pca_mean, pca_comp.T)
    features_final = np.concatenate([pixels_pca, expert_scaled])

    proba = mlp.predict_proba_relu(features_final, w, b)
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
            combined_features = pre_traitement.transformer_image_en_features(path_in, image_size)
            combined_features_scaled = (combined_features - mean_train) / std_train
            combined_features_scaled = np.clip(combined_features_scaled, -5, 5)
            
            pixels_scaled = combined_features_scaled[:n_pixels]
            expert_scaled = combined_features_scaled[n_pixels:]
            
            pixels_pca = np.dot(pixels_scaled - pca_mean, pca_comp.T)
            features_final = np.concatenate([pixels_pca, expert_scaled])
            
            proba = mlp.predict_proba_relu(features_final, w, b)
            
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