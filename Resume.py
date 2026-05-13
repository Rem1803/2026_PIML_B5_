import numpy as np
import matplotlib.pyplot as plt

def rgb_to_grayscale(image):
    """
    Convertit une image RGB en niveaux de gris.
    
    Parameters:
    image (numpy.ndarray): Image RGB à convertir.
    
    Returns:
    numpy.ndarray: Image en niveaux de gris.
    """
    # Vérifier si l'image est déjà en niveaux de gris
    if len(image.shape) == 2:
        return image
    
    # Utiliser la formule de luminosité pour convertir en niveaux de gris
    grayscale = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    
    return grayscale.astype(np.float64)



def standardize_image(image):
    """
    Normalise une image en soustrayant la moyenne et en divisant par l'écart type.
    
    Parameters:
    image (numpy.ndarray): Image à normaliser.
    
    Returns:
    numpy.ndarray: Image normalisée.
    """
    mean = np.mean(image)
    std = np.std(image)
    
    if std == 0:
        return image - mean  # Éviter la division par zéro
    
    standardized_image = (image - mean) / std
    return standardized_image



def equalize_histogram(image):
    """
    Égalise l'histogramme d'une image en niveaux de gris.
    
    Parameters:
    image (numpy.ndarray): Image en niveaux de gris à égaliser.
    
    Returns:
    numpy.ndarray: Image avec histogramme égalisé.
    """
    # Calculer l'histogramme de l'image
    histogram, bin_edges = np.histogram(image.flatten(), bins=256, range=(0, 255))
    
    # Calculer la fonction de distribution cumulative (CDF)
    cdf = histogram.cumsum()
    
    # Normaliser la CDF
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    
    # Appliquer la transformation d'égalisation
    equalized_image = np.interp(image.flatten(), bin_edges[:-1], cdf_normalized)
    
    return equalized_image.reshape(image.shape).astype(np.float64)







# ==========================================
# Noé
# =========================================="""

# ==========================================
# 1. Évaluation des performances
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
# 2. Visualisation de la matrice
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
    
    for i in range(nb_lignes):
        for j in range(nb_colonnes):
            plt.text(j, i, format(mc[i, j], format_texte),
                        horizontalalignment="center",
                        color="white" if mc[i, j] > seuil else "black")

    plt.ylabel('Réalité')
    plt.xlabel('Prédictions du Modèle')
    plt.tight_layout()

# ==========================================
# 3. Courbe ROC et AUC
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
    # --- Test des métriques de base ---
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