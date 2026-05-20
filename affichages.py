import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import Partie_1_Pre_Traitement as pre_traitement
import Partie_3_Evaluation as eval
import itertools

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

def plot_advanced_eda(data, target):
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    
    pca_saines = data_pca[target == 0]
    pca_infectees = data_pca[target == 1]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_saines[:, 0], pca_saines[:, 1], alpha=0.5, label='Saines', edgecolors='none')
    plt.scatter(pca_infectees[:, 0], pca_infectees[:, 1], alpha=0.5, label='Infectées', edgecolors='none')
    plt.title("Projection PCA (Espace à 2 Dimensions)")
    plt.xlabel(f"Composante Principale 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"Composante Principale 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.legend()
    plt.tight_layout()
    plt.show()


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

#Plot de la matrice de confusion
def plot_confusion_matrix(cm, classes, normalize=False, title='Matrice de Confusion', cmap=plt.cm.Blues):
    """
    Cette fonction affiche la matrice de confusion.
    Normalisation peut être appliquée en définissant `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matrice de confusion normalisée")
    else:
        print('Matrice de confusion, sans normalisation')

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