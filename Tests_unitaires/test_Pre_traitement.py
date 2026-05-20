
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
    img = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    target_size = (1, 1)
    resized = pre_traitement.resize_images([img], target_size=target_size)[0]
    
    expected_resized = np.array([[127]], dtype=np.uint8)  # La moyenne des pixels
    assert np.array_equal(resized, expected_resized), "Resizing failed for simple image."



def test_watershed_region_count():
    # Test avec une image simple
    img = np.array([[0, 255], [255, 0]])
    region_count = pre_traitement.watershed_region_count(img)
    
    expected_region_count = 2  # Il y a deux régions distinctes
    assert region_count == expected_region_count, "Watershed region count failed for simple image."



def test_extract_advanced_features():
    img_rgb = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]], dtype=np.uint8)
    img_hsv = pre_traitement.RGB_to_HSV([img_rgb])[0]
    features = pre_traitement.extract_advanced_features(img_rgb, img_hsv)
    expected_features = np.array([255, 255, 255, 0, 0, 0, 127.5, 127.5])  # Moyennes des canaux
    assert np.allclose(features, expected_features), "Advanced feature extraction failed for simple image."




def transformer_image_en_features():
    # Test avec une image simple
    chemin_image = "test_image.jpg"
    image_size = (32, 32)
    
    # Créer une image de test
    img = Image.new("RGB", (64, 64), color="red")
    img.save(chemin_image)
    
    features = pre_traitement.transformer_image_en_features(chemin_image, image_size)
    
    assert features.shape == (8,), "Feature extraction failed to produce the correct shape."