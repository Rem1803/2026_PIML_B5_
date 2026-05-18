import numpy as np
from PIL import Image
from skimage import exposure
from matplotlib.colors import rgb_to_hsv

# ==============================================================================================
# Fonctions pour traitement des images 
# ==============================================================================================

# Plus utilisée
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

# ============

def standardize_image(images):
    """
    Normalise une image en soustrayant la moyenne et en divisant par l'écart type.
    
    Parameters:
    image (numpy.ndarray): Image à normaliser.
    
    Returns:
    numpy.ndarray: Image normalisée.
    """
    standardized_images = []
    for img in images:

        mean = np.mean(img)
        std = np.std(img)

        if std == 0:
            standardized_images.append(img - mean)
        else:
            standardized_images.append((img - mean) / std)

    return standardized_images

# ============

# Plus utilisée
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

# ============

def resize_images(images, target_size=(16, 16)):
    """
    Redimensionne une liste d'images avec PIL.

    Paramètres
    ----------
    images : list
        Liste d'images numpy

    target_size : tuple (width, height)
        Taille cible.
        Si None : utilise la taille de la première image.

    Retour
    ------
    resized_images : list
    """

    resized_images = []

    for img in images:

        # Conversion numpy -> PIL
        pil_img = Image.fromarray(img)

        # Resize
        resized_pil = pil_img.resize(
            target_size,
            Image.Resampling.LANCZOS
        )

        # Retour PIL -> numpy
        resized = np.array(resized_pil)

        resized_images.append(resized)

    return resized_images

# ============

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

# ============

def RGB_to_HSV(images):
    """
    Convertit une liste d'images RGB en HSV.

    Paramètres
    ----------
    images : list
        Liste d'images RGB numpy uint8

    Retour
    ------
    hsv_images : list
        Liste d'images HSV
        H,S,V dans [0,1]
    """

    hsv_images = []

    for img in images:

        # Conversion en float [0,1]
        img_float = img.astype(np.float32) / 255.0

        # RGB -> HSV
        hsv = rgb_to_hsv(img_float)

        hsv_images.append(hsv)

    return hsv_images

# ============

def HSV_by_HS(hsv_images):
    """
    Remplace chaque pixel HSV (H,S,V)
    par la valeur H*S.

    Paramètres
    ----------
    hsv_images : list
        Liste d'images HSV

    Retour
    ------
    hs_images : list
        Liste d'images 2D
    """

    hs_images = []

    for hsv in hsv_images:

        H = hsv[:, :, 0]
        S = hsv[:, :, 1]

        # Produit H*S
        hs = H * S

        hs_images.append(hs)

    return hs_images