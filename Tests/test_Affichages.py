import Module.Affichages as aff
import numpy as np

def test_plot_image_grid():
    # Test avec des dossiers de test
    uninfected_dir = "test_uninfected"
    parasitized_dir = "test_parasitized"
    
    aff.plot_image_grid(uninfected_dir, parasitized_dir, n_images_per_class=4)



def test_plot_advanced_eda():
    # Test avec des données de test
    data = np.random.rand(100, 10)  # 100 échantillons, 10 caractéristiques
    target = np.random.randint(0, 2, size=100)  # Cibles binaires
    
    aff.plot_advanced_eda(data, target)



def test_plot_losses():
    # Test avec des pertes de test
    losses = [0.9, 0.7, 0.5, 0.3, 0.1]
    aff.plot_losses(losses, title="Test de l'évolution de la loss")