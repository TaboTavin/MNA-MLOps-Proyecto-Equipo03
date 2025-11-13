"""
Pruebas unitarias para cálculo de métricas y validación de resultados.
"""

import pytest
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TestMetricsCalculation:
    """Pruebas para validar el cálculo correcto de métricas."""
    
    def test_accuracy_calculation_perfect_predictions(self):
        """Valida accuracy con predicciones perfectas."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        
        accuracy = accuracy_score(y_true, y_pred)
        assert accuracy == 1.0
    
    def test_accuracy_calculation_zero_correct(self):
        """Valida accuracy con cero predicciones correctas."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        
        accuracy = accuracy_score(y_true, y_pred)
        assert accuracy == 0.0
    
    def test_accuracy_calculation_partial_correct(self):
        """Valida accuracy con predicciones parcialmente correctas."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])  # 3/6 correctos
        
        accuracy = accuracy_score(y_true, y_pred)
        assert accuracy == 0.5
    
    def test_precision_weighted_multiclass(self):
        """Valida precision weighted para clasificación multiclase."""
        y_true = np.array(['happy', 'sad', 'happy', 'energetic', 'sad'])
        y_pred = np.array(['happy', 'happy', 'happy', 'energetic', 'sad'])
        
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        
        assert 0.0 <= precision <= 1.0
        assert isinstance(precision, float)
    
    def test_recall_weighted_multiclass(self):
        """Valida recall weighted para clasificación multiclase."""
        y_true = np.array(['happy', 'sad', 'happy', 'energetic', 'sad'])
        y_pred = np.array(['happy', 'happy', 'happy', 'energetic', 'sad'])
        
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        assert 0.0 <= recall <= 1.0
        assert isinstance(recall, float)
    
    def test_f1_score_weighted_multiclass(self):
        """Valida F1-score weighted para clasificación multiclase."""
        y_true = np.array(['happy', 'sad', 'happy', 'energetic', 'sad'])
        y_pred = np.array(['happy', 'happy', 'happy', 'energetic', 'sad'])
        
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        assert 0.0 <= f1 <= 1.0
        assert isinstance(f1, float)
    
    def test_metrics_with_all_classes_predicted_wrong(self):
        """Valida métricas cuando todas las clases se predicen incorrectamente."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        assert accuracy == 0.0
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0
        assert 0.0 <= f1 <= 1.0
    
    def test_metrics_handle_zero_division(self):
        """Valida que las métricas manejan correctamente la división por cero."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        
        # No debe lanzar error con zero_division=0
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        assert isinstance(precision, float)
        assert isinstance(recall, float)
        assert isinstance(f1, float)


class TestMetricsConsistency:
    """Pruebas de consistencia entre diferentes métricas."""
    
    def test_perfect_predictions_all_metrics_equal_one(self):
        """Con predicciones perfectas, todas las métricas deben ser 1.0."""
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        assert accuracy == 1.0
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0
    
    def test_f1_is_harmonic_mean_of_precision_recall(self):
        """F1 debe ser la media armónica de precision y recall."""
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
        
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        
        # F1 = 2 * (precision * recall) / (precision + recall)
        expected_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        assert abs(f1 - expected_f1) < 1e-10
    
    def test_metrics_with_string_labels(self):
        """Valida que las métricas funcionan con etiquetas de tipo string."""
        y_true = ['happy', 'sad', 'energetic', 'calm', 'happy', 'sad']
        y_pred = ['happy', 'sad', 'energetic', 'happy', 'happy', 'sad']
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        assert 0.0 <= accuracy <= 1.0
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0
        assert 0.0 <= f1 <= 1.0


class TestMetricsEdgeCases:
    """Pruebas de casos extremos en el cálculo de métricas."""
    
    def test_single_sample_prediction(self):
        """Valida métricas con una sola muestra."""
        y_true = np.array([1])
        y_pred = np.array([1])
        
        accuracy = accuracy_score(y_true, y_pred)
        assert accuracy == 1.0
    
    def test_binary_classification_imbalanced(self):
        """Valida métricas con clases desbalanceadas."""
        # 90% clase 0, 10% clase 1
        y_true = np.array([0] * 90 + [1] * 10)
        y_pred = np.array([0] * 85 + [1] * 5 + [1] * 10)  # Predice más clase 1
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Todas las métricas deben estar en el rango válido
        for metric in [accuracy, precision, recall, f1]:
            assert 0.0 <= metric <= 1.0
    
    def test_multiclass_with_missing_predicted_class(self):
        """Valida métricas cuando una clase nunca se predice."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 0, 0, 1, 1, 1])  # Clase 2 nunca predicha
        
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0
        assert 0.0 <= f1 <= 1.0
    
    def test_metrics_with_four_classes(self):
        """Valida métricas con las 4 clases de emociones musicales."""
        classes = ['happy', 'sad', 'energetic', 'calm']
        y_true = classes * 10  # 10 muestras de cada clase
        y_pred = (classes * 8) + ['happy'] * 8  # Algunas predicciones incorrectas
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Accuracy debe ser exactamente 32/40 = 0.8
        assert accuracy == 0.8
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0
        assert 0.0 <= f1 <= 1.0
