import Module.Application as app
import numpy as np

def test_diagnostiquer_une_cellule():
    # Test avec une image de test
    image_path = "test_image.jpg"
    w = np.random.rand(10, 10)  # Poids fictifs pour le test
    b = np.random.rand(10)       # Biais fictifs pour le test
    mean_train = np.random.rand(10)  # Moyenne fictive pour le test
    std_train = np.random.rand(10)   # Écart-type fictif pour le test
    pca_comp = np.random.rand(10, 10)  # Composantes PCA fictives pour le test
    pca_mean = np.random.rand(10)      # Moyenne PCA fictive pour le test
    
    app.diagnostiquer_une_cellule(image_path, w, b, mean_train, std_train, pca_comp, pca_mean)


def test_trier_dossier_images():
    # Test avec un dossier de test
    dossier_entree = "test_images"
    w = np.random.rand(10, 10)  # Poids fictifs pour le test
    b = np.random.rand(10)       # Biais fictifs pour le test
    mean_train = np.random.rand(10)  # Moyenne fictive pour le test
    std_train = np.random.rand(10)   # Écart-type fictif pour le test
    pca_comp = np.random.rand(10, 10)  # Composantes PCA fictives pour le test
    pca_mean = np.random.rand(10)      # Moyenne PCA fictive pour le test
    
    app.trier_dossier_images(dossier_entree, w, b, mean_train, std_train, pca_comp, pca_mean)