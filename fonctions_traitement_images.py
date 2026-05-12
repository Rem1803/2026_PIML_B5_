import numpy as np


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

