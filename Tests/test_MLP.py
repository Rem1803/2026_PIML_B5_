import Module.MLP as MLP
import numpy as np


# =======================
# CONFIGURATION 
# =======================

TAILLE_IMAGE = (32, 32)    
MAX_IMAGES = 1000           
TAUX_APPRENTISSAGE = 0.01  
BATCH_SIZE = 32
EPOCHS = 150              
PATIENCE = 5      
SEUIL_DECISION = 0.35
TAUX_DROPOUT = 0.2   
LAMBDA_L2 = 0.0001   # Régularisation L2
COMPOSANTES_PCA = 50       

# =======================
# Chemins des dossiers 
# =======================

UNINFECTED_PATH = r"Uninfected"
PARASITIZED_PATH = r"Parasitized"


def  test_load_images():
    # Test de la fonction load_images
    images, labels = MLP.load_images(UNINFECTED_PATH, PARASITIZED_PATH)
    assert images.shape == (60000, 28, 28), "La forme des images est incorrecte"
    assert labels.shape == (60000,), "La forme des labels est incorrecte"
    assert np.all((images >= 0) & (images <= 255)), "Les valeurs des pixels doivent être entre 0 et 255"
    assert np.all((labels >= 0) & (labels <= 9)), "Les labels doivent être entre 0 et 9"



def test_sigmoid():
    # Test de la fonction sigmoid
    x = np.array([-1, 0, 1])
    expected = np.array([0.26894142, 0.5, 0.73105858])
    assert np.allclose(MLP.sigmoid(x), expected), "La fonction sigmoid ne retourne pas les valeurs attendues"



def test_init_parameters():
    # Test de la fonction init_parameters
    input_size = 10
    hidden_size = 5
    output_size = 2
    parameters = MLP.init_parameters(input_size, hidden_size, output_size)
    assert parameters['W1'].shape == (input_size, hidden_size), "La forme de W1 est incorrecte"
    assert parameters['b1'].shape == (hidden_size,), "La forme de b1 est incorrecte"
    assert parameters['W2'].shape == (hidden_size, output_size), "La forme de W2 est incorrecte"
    assert parameters['b2'].shape == (output_size,), "La forme de b2 est incorrecte"



def test_eval_forward_relu():
    # Test de la fonction eval_forward_relu
    X = np.array([[1, 2], [3, 4]])
    parameters = {
        'W1': np.array([[0.1, 0.2], [0.3, 0.4]]),
        'b1': np.array([0.5, 0.6]),
        'W2': np.array([[0.7, 0.8], [0.9, 1.0]]),
        'b2': np.array([1.1, 1.2])
    }
    A2 = MLP.eval_forward_relu(X, parameters)
    expected_A2 = np.array([[1.26, 1.56], [3.06, 3.36]])
    assert np.allclose(A2, expected_A2), "La fonction eval_forward_relu ne retourne pas les valeurs attendues"



def test_mlp_error_entropy():
    # Test de la fonction mlp_error_entropy
    Y = np.array([[1, 0], [0, 1]])
    A2 = np.array([[0.9, 0.1], [0.2, 0.8]])
    error = MLP.mlp_error_entropy(Y, A2)
    expected_error = -np.mean(np.sum(Y * np.log(A2), axis=1))
    assert np.isclose(error, expected_error), "La fonction mlp_error_entropy ne retourne pas les valeurs attendues"



def test_mlp_fit_mini_batch():
    # Test de la fonction mlp_fit_mini_batch
    X = np.random.rand(100, 10)
    Y = np.random.rand(100, 2)
    parameters = MLP.init_parameters(10, 5, 2)
    parameters = MLP.mlp_fit_mini_batch(X, Y, parameters, TAUX_APPRENTISSAGE, BATCH_SIZE, EPOCHS, PATIENCE)
    assert 'W1' in parameters and 'b1' in parameters and 'W2' in parameters and 'b2' in parameters, "Les paramètres ne sont pas correctement mis à jour"



def test_predict_relu():
    # Test de la fonction predict_relu
    X = np.array([[1, 2], [3, 4]])
    parameters = {
        'W1': np.array([[0.1, 0.2], [0.3, 0.4]]),
        'b1': np.array([0.5, 0.6]),
        'W2': np.array([[0.7, 0.8], [0.9, 1.0]]),
        'b2': np.array([1.1, 1.2])
    }
    predictions = MLP.predict_relu(X, parameters)
    expected_predictions = np.array([[1, 1], [1, 1]])
    assert np.array_equal(predictions, expected_predictions), "La fonction predict_relu ne retourne pas les valeurs attendues"



def test_cross_validation():
    # Test de la fonction cross_validation
    X = np.random.rand(100, 10)
    Y = np.random.rand(100, 2)
    parameters = MLP.init_parameters(10, 5, 2)
    accuracy = MLP.cross_validation(X, Y, parameters, TAUX_APPRENTISSAGE, BATCH_SIZE, EPOCHS, PATIENCE)
    assert 0 <= accuracy <= 1, "L'accuracy doit être entre 0 et 1"



def test_random_search_hyperparameters():
    # Test de la fonction random_search_hyperparameters
    X = np.random.rand(100, 10)
    Y = np.random.rand(100, 2)
    best_params = MLP.random_search_hyperparameters(X, Y, TAUX_APPRENTISSAGE, BATCH_SIZE, EPOCHS, PATIENCE)
    assert 'learning_rate' in best_params and 'batch_size' in best_params and 'epochs' in best_params and 'patience' in best_params, "Les hyperparamètres ne sont pas correctement retournés"



