import numpy as np
import cv2


# PIPELINE OPTIMAL :
# 1-resize_images
# 2-rgb_to_grayscale
# 3-equalize_histogram_CLAHE
# 4-standardize_image


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
    Standardise une image en soustrayant la moyenne et en divisant par l'écart type.
    
    Parameters:
    image (numpy.ndarray): Image à standardiser.
    
    Returns:
    numpy.ndarray: Image standardisée.
    """
    mean = np.mean(image)
    std = np.std(image)
    
    if std == 0:
        return image - mean  # Éviter la division par zéro
    
    standardized_image = (image - mean) / std
    return standardized_image



def equalize_histogram_global(image):
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



def equalize_histogram_CLAHE(image):
    # Création du CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # Application
    equalized = clahe.apply(image.astype(np.uint8))
    return equalized.astype(np.float64)



def resize_images(images, target_size=(16, 16)):
    """
    Redimensionne une liste d'images pour qu'elles aient toutes
    les mêmes dimensions.

    Paramètres
    ----------
    images : list
        Liste d'images numpy (grayscale ou RGB)

    target_size : tuple (width, height), optionnel, Taille cible.

    Retour
    ------
    resized_images : list
        Liste des images redimensionnées
    """

    if len(images) == 0:
        return []

    resized_images = []

    for img in images:
        resized = cv2.resize(
            img,
            target_size,
            interpolation=cv2.INTER_AREA
        )
        resized_images.append(resized)

    return resized_images



def add_noise(images, noise_type="gaussian", strength=25):
    """
    Ajoute du bruit à une liste d'images.

    Paramètres
    ----------
    images : list
        Liste d'images numpy

    noise_type : str
        "gaussian" ou "salt_pepper"

    strength : float
        Intensité du bruit

    Retour
    ------
    noisy_images : list
        Liste des images bruitées
    """

    noisy_images = []

    for img in images:

        # Conversion en float pour éviter les dépassements
        noisy = img.astype(np.float32)

        # =========================
        # Bruit gaussien
        # =========================
        if noise_type == "gaussian":

            noise = np.random.normal(
                loc=0,
                scale=strength,
                size=img.shape
            )

            noisy = noisy + noise

        # =========================
        # Bruit sel / poivre
        # =========================
        elif noise_type == "salt_pepper":
            prob = strength
            noisy = noisy.copy()

            # Pixels blancs
            salt_mask = np.random.rand(*img.shape[:2]) < prob / 2

            # Pixels noirs
            pepper_mask = np.random.rand(*img.shape[:2]) < prob / 2

            if img.ndim == 3:
                noisy[salt_mask] = [255] * img.shape[2]
                noisy[pepper_mask] = [0] * img.shape[2]
            else:
                noisy[salt_mask] = 255
                noisy[pepper_mask] = 0

        else:
            raise ValueError("noise_type doit être 'gaussian' ou 'salt_pepper'")

        # Limite les valeurs entre 0 et 255
        noisy = np.clip(noisy, 0, 255)

        # Retour au format image classique
        noisy_images.append(noisy.astype(np.float64))

    return noisy_images



