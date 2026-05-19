import numpy as np
from PIL import Image
from skimage import exposure
from matplotlib.colors import rgb_to_hsv
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.measure import label


# PIPELINE OPTIMAL :
# -1-resize_images
# -2-RGB_to_HSV
# -3-HSV_by_HS
# -4-standardize_images

# DESCRIPTEURS ADDITIONNELS : nb de régions segmentées
# -watershed_region_count (à utiliser après le pipeline optimal)

# ==============================================================================================
# Fonctions pour traitement des images 
# ==============================================================================================


def standardize_images(images):
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

def resize_images(images, target_size=(32, 32)):
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

def watershed_region_count(images):
    """
    Applique une segmentation Watershed à une liste d'images
    et retourne le nombre de régions détectées pour chaque image.

    Paramètres
    ----------
    images : list[np.ndarray]
        Liste d'images grayscale (2D).

    Retour
    -------
    counts : list[int]
        counts[i] = nombre de régions détectées
        sur images[i].
    """

    counts = []

    for img in images:

        # Conversion float
        img = img.astype(np.float32)

        # Binarisation automatique (Otsu)
        thresh = threshold_otsu(img)
        binary = img > thresh

        # Distance transform
        distance = ndimage.distance_transform_edt(binary)

        # Détection des marqueurs
        coords = peak_local_max(
            distance,
            footprint=np.ones((3, 3)),
            labels=binary
        )

        # Création image des maxima locaux
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True

        # Labellisation des marqueurs
        markers, _ = ndimage.label(mask)

        # Watershed
        labels_ws = watershed(
            -distance,
            markers,
            mask=binary
        )

        # Nombre de régions
        n_regions = len(np.unique(labels_ws)) - 1
        counts.append(n_regions)

    return counts

# ============
