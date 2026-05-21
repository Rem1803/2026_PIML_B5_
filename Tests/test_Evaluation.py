import Module.Evaluation as eval

def test_matrice_confusion():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    cm = eval.matrice_confusion(y_true, y_pred)
    
    expected_cm = np.array([[1, 1], [1, 1]])
    assert np.array_equal(cm, expected_cm), "Confusion matrix calculation failed for simple case."



def tes_roc_auc():
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35, 0.8]
    auc = eval.roc_auc(y_true, y_scores)
    
    expected_auc = 0.75
    assert np.isclose(auc, expected_auc), "ROC AUC calculation failed for simple case."



def test_roc_curve_points():
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35, 0.8]
    fpr, tpr, thresholds = eval.roc_curve_points(y_true, y_scores)
    
    expected_fpr = np.array([0.0, 0.5, 1.0])
    expected_tpr = np.array([0.0, 0.5, 1.0])
    
    assert np.allclose(fpr, expected_fpr), "FPR calculation failed for simple case."
    assert np.allclose(tpr, expected_tpr), "TPR calculation failed for simple case."

