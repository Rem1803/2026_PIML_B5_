# ==============================================================================================
# Partie 3 — Évaluation, interprétabilité & application finale
# ==============================================================================================

import numpy as np
import matplotlib.pyplot as plt
import itertools

# ==============================================================================================
# 1. Fonctions Mathématiques et Métriques
# ==============================================================================================

def matrice_confusion(y_vrai, y_predit):
    """Calcule la matrice de confusion à partir des prédictions brutes."""
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

def roc_auc(y_true, y_scores):
    """
    Calcule le score ROC AUC en utilisant la vectorisation NumPy.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    
    pos_scores = y_scores[y_true == 1]
    neg_scores = y_scores[y_true == 0]
    
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        raise ValueError("Il faut au moins un exemple de la classe 0 et de la classe 1.")
        
    diff = pos_scores[:, np.newaxis] - neg_scores
    concordant_pairs = np.sum(diff > 0)
    ties = np.sum(diff == 0) * 0.5
    
    total_pairs = len(pos_scores) * len(neg_scores)
    return (concordant_pairs + ties) / total_pairs

def roc_curve_points(y_true, y_scores):
    """
    Calcule les points (FPR, TPR) et les seuils de la courbe ROC.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_scores_sorted = y_scores[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]
    
    total_positives = np.sum(y_true == 1)
    total_negatives = np.sum(y_true == 0)
    
    tpr_list = [0.0]
    fpr_list = [0.0]
    thresholds = [y_scores_sorted[0] + 0.1]
    
    tp = 0
    fp = 0
    
    for i in range(len(y_scores_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
            
        if i == len(y_scores_sorted) - 1 or y_scores_sorted[i] != y_scores_sorted[i+1]:
            tpr_list.append(tp / total_positives)
            fpr_list.append(fp / total_negatives)
            thresholds.append(y_scores_sorted[i])
            
    return np.array(fpr_list), np.array(tpr_list), np.array(thresholds)


# ==============================================================================================
# 2. Fonctions de Visualisation
# ==============================================================================================

def plot_confusion_matrix(cm, classes, normalize=False, title='Matrice de Confusion', cmap=plt.cm.Blues):
    """
    Affiche la matrice de confusion (format Numpy Array).
    Normalisation peut être appliquée en définissant `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Affichage de la Matrice de confusion normalisée...")
    else:
        print('Affichage de la Matrice de confusion (sans normalisation)...')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Réalité')
    plt.xlabel('Prédictions du Modèle')
    plt.tight_layout()

def plot_roc_curve(fpr, tpr, auc_score, title='Courbe ROC (From Scratch)'):
    """
    Génère et affiche le graphique de la courbe ROC.
    """
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'Courbe ROC (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--') # La diagonale du hasard
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (FPR)')
    plt.ylabel('Taux de Vrais Positifs (TPR)')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()


# ==============================================================================================
# 3. Exécution et Tests (Uniquement si le script est lancé directement)
# ==============================================================================================

if __name__ == "__main__":
    
    print("--- TEST DES FONCTIONS DE MÉTRIQUES ---")
    
    # 1. Test de la Matrice de Confusion
    cm_test = np.array([[50, 10], [5, 35]])  
    classes_labels = ['Cellules Saines', 'Cellules Malades']
    
    plt.figure(figsize=(6, 5))
    plot_confusion_matrix(cm_test, classes_labels, normalize=True)
    plt.show()

    # 2. Test du score ROC-AUC et de la Courbe ROC
    y_true_test = [0, 0, 1, 1]
    y_scores_test = [0.1, 0.4, 0.35, 0.8]
    
    # Calculs
    auc_score_vectorized = roc_auc(y_true_test, y_scores_test)
    print(f"ROC AUC Score (Vectorisé) : {auc_score_vectorized:.4f}")
    
    fpr, tpr, thresholds = roc_curve_points(y_true_test, y_scores_test)
    auc_score_trapezoid = np.trapezoid(tpr, fpr)
    print(f"ROC AUC Score (Règle du trapèze) : {auc_score_trapezoid:.4f}")
    
    # Affichage
    plot_roc_curve(fpr, tpr, auc_score_trapezoid)
    plt.show()

    plt.close("all")