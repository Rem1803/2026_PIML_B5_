import Module.Evaluation as eval
import numpy as np

def test_matrice_confusion():
    y_vrai = np.array([1, 1, 0, 0, 1, 0])
    y_predit = np.array([1, 0, 0, 1, 1, 0])

    result = eval.matrice_confusion(y_vrai, y_predit)

    assert result["VP"] == 2
    assert result["VN"] == 2
    assert result["FP"] == 1
    assert result["FN"] == 1



def test_roc_auc():
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35, 0.8]
    auc = eval.roc_auc(y_true, y_scores)
    
    expected_auc = 0.75
    assert np.isclose(auc, expected_auc), "ROC AUC calculation failed for simple case."



def test_roc_curve_points():
    """
    Vérifie simplement que la fonction
    fonctionne sur des données aléatoires.
    """

    rng = np.random.default_rng(42)

    y_true = rng.integers(0, 2, 100)
    y_scores = rng.random(100)

    fpr, tpr, thresholds = eval.roc_curve_points(
        y_true,
        y_scores
    )

    assert len(fpr) > 0
    assert len(tpr) > 0
    assert len(thresholds) > 0
