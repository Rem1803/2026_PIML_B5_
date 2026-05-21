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




def test_sigmoid():
    # Test de la fonction sigmoid
    x = np.array([-1, 0, 1])
    expected = np.array([0.26894142, 0.5, 0.73105858])
    assert np.allclose(MLP.sigmoid(x), expected), "La fonction sigmoid ne retourne pas les valeurs attendues"



def test_init_parameters():
    n_feats = 4
    hidden_layers = [5, 3]

    rng = np.random.default_rng(42)

    w, b = MLP.init_parameters(n_feats, hidden_layers, rng)

    # --- Vérification de la structure ---
    # w et b doivent contenir :
    # [dummy, layer1, layer2, output]

    # --- Vérification des formes des poids ---
    assert w[1].shape == (5, 4)      # première couche cachée
    assert w[2].shape == (3, 5)      # deuxième couche cachée
    assert w[3].shape == (1, 3)      # couche de sortie

    # --- Vérification des biais ---
    assert b[1].shape == (5,)
    assert b[2].shape == (3,)
    assert b[3].shape == (1,)

    # --- Vérification des statistiques (He init approx) ---
    # moyenne proche de 0
    assert np.abs(np.mean(w[1])) < 1.0

    # variance approximative (pas stricte mais sanity check)
    expected_var = 2 / n_feats
    assert np.isclose(np.var(w[1]), expected_var, rtol=0.5)

    # --- Reproductibilité ---
    rng2 = np.random.default_rng(42)
    w2, b2 = MLP.init_parameters(n_feats, hidden_layers, rng2)

    for i in range(len(w)):
        assert np.allclose(w[i], w2[i])
        assert np.allclose(b[i], b2[i])



def test_eval_forward_relu():
    # Réseau simple
    x = np.array([1.0, -1.0, 2.0])
    w = [
        np.zeros((0, 0)),  # dummy
        np.array([[0.5, -0.2, 0.1],
                  [0.3, 0.8, -0.5]]),
        np.array([[1.0, -1.0]])
    ]
    b = [
        np.zeros(0),       # dummy
        np.array([0.1, -0.1]),
        np.array([0.0])
    ]

    rng = np.random.default_rng(42)

    a, masks = MLP.eval_forward_relu(x, w, b, dropout_rate=0.0, training=False, rng=rng)

    # --- Structure ---
    assert len(a) == len(w)
    assert len(masks) == len(w)

    # --- Formes ---
    assert a[0].shape == x.shape
    assert a[-1].shape == (1,)

    # --- Pas de dropout en inference ---
    assert all(m is None for m in masks)

    # --- Reproductibilité (sans dropout) ---
    rng2 = np.random.default_rng(42)
    a2, masks2 = MLP.eval_forward_relu(x, w, b, dropout_rate=0.0, training=False, rng=rng2)

    for i in range(len(a)):
        assert np.allclose(a[i], a2[i])



def test_mlp_error_entropy():
    # --- Données simples ---
    data = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0])
    ]
    target = np.array([1, 0])

    # --- Réseau trivial (prédiction constante ~0.5) ---
    w = [
        np.zeros((0, 0)),
        np.zeros((1, 2))
    ]
    b = [
        np.zeros(0),
        np.zeros(1)
    ]

    loss = MLP.mlp_error_entropy(data, target, w, b)

    # sigmoid(0) = 0.5 => loss = -log(0.5)
    expected = -np.log(0.5)

    assert np.isclose(loss, expected, atol=1e-6)



def test_mlp_fit_minibatch():
    np.random.seed(0)

    data = np.random.randn(20, 4)
    target = (np.random.rand(20) > 0.5).astype(int)

    w, b, losses = MLP.mlp_fit_minibatch(
        data, target,
        n_epochs=3,
        hidden_layer_sizes=[5],
        batch_size=4,
        random_state=42,
        dropout_rate=0.0,
        patience=10
    )

    # --- Structure ---
    assert isinstance(losses, list)
    assert len(losses) <= 3

    # w, b contiennent layers + sortie
    assert len(w) == 3
    assert len(b) == 3

    # --- Shapes cohérentes ---
    assert w[1].shape[0] == 5
    assert w[1].shape[1] == 4
    assert w[1].shape == (5, 4)

    assert w[2].shape[0] == 1
    assert w[2].shape[1] == 5



def test_predict_relu():
    x = np.array([0.2, 0.8])
    w = [
        np.zeros((0, 0)),
        np.array([[0.1, 0.2]]),
        np.array([[0.3]])
    ]
    b = [
        np.zeros(0),
        np.array([0.0]),
        np.array([0.0])
    ]

    pred = MLP.predict_relu(x, w, b)

    assert pred in [0, 1]
    assert isinstance(pred, int)



def test_cross_validation():
    data = np.random.randn(100, 60)
    target = (np.sum(data, axis=1) > 0).astype(int)

    def fake_train(*args, **kwargs):
        return [None, np.zeros((1,1))], [None, np.zeros(1)], [1.0]

    def fake_predict(x, w, b, seuil):
        return 0

    class fake_eval:
        @staticmethod
        def matrice_confusion(y_true, y_pred):
            return {"VP": 0, "VN": len(y_true), "FP": 0, "FN": 0}

        @staticmethod
        def exactitude(mc):
            return mc["VN"] / (mc["VP"] + mc["VN"] + mc["FP"] + mc["FN"])

    result = MLP.cross_validation(
        data, target,
        train_func=fake_train,
        predict_func=fake_predict,
        n_folds=4,
        n_epochs=1,
        hidden_layer_sizes=[2],
        random_state=1
    )

    assert 0.0 <= result["mean"] <= 1.0

