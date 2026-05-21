
from tkinter import Image
import Module.Pre_traitement as pre_traitement
import numpy as np


def test_standardize_images():
    # Test avec une image simple
    img = np.array([[1, 2], [3, 4]], dtype=np.float32)
    standardized = pre_traitement.standardize_images([img])[0]
    
    expected_mean = 2.5
    expected_std = 1.118033988749895
    expected_standardized = (img - expected_mean) / expected_std
    
    assert np.allclose(standardized, expected_standardized), "Standardization failed for simple image."



def test_resize_images():
    # Test avec une image simple
    img = np.array([[0, 2], [2, 0]], dtype=np.uint8)
    target_size = (1, 1)
    resized = pre_traitement.resize_images([img], target_size=target_size)[0]
    
    expected_resized = np.array([[1]], dtype=np.uint8)  # La moyenne des pixels
    assert np.array_equal(resized, expected_resized), "Resizing failed for simple image."



def test_watershed_region_count():
    """
    Vérifie que Watershed détecte deux régions
    sur une image artificielle simple.
    """

    img = np.zeros((100, 100))

    # disque 1
    y, x = np.ogrid[:100, :100]
    mask1 = (x - 30)**2 + (y - 30)**2 < 10**2

    # disque 2
    mask2 = (x - 70)**2 + (y - 70)**2 < 10**2

    img[mask1] = 255
    img[mask2] = 255

    counts = pre_traitement.watershed_region_count([img])

    assert counts[0] == 2



def test_extract_advanced_features():
    img_rgb = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]], dtype=np.uint8)
    img_hsv = pre_traitement.RGB_to_HSV([img_rgb])[0]
    features = pre_traitement.extract_advanced_features(img_rgb, img_hsv)
    assert len(features)==8, "Advanced feature extraction failed for simple image."

