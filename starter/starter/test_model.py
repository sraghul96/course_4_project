import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pytest


from ml.model import train_model, inference, compute_model_metrics


@pytest.fixture(scope="module")
def sample_data():
    """
    Fixture to provide sample data for testing.
    """
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_repeated=2,
        n_classes=2,
        random_state=42,
    )
    X_test, y_test = X[:10], y[:10]
    return X, y, X_test, y_test


@pytest.fixture(scope="module")
def get_model(sample_data):
    """
    Fixture to provide a trained model for testing.
    """
    X, y, _, _ = sample_data
    model = train_model(X, y, fine_tuning=False)
    return model


def test_train_model(get_model):
    """
    Test the train_model function.
    """
    model = get_model
    assert isinstance(
        model, RandomForestClassifier), "Model should be a RandomForestClassifier"


def test_inference(sample_data, get_model):
    """
    Test the inference function.
    """
    _, _, X_test, y_test = sample_data
    model = get_model
    preds = inference(model, X_test)
    assert len(preds) == len(
        X_test), "Number of predictions should match number of test samples"
    assert all(np.isin(preds, [0, 1])), "Predictions should be binary (0 or 1)"
    assert isinstance(preds, np.ndarray), "Predictions should be a numpy array"
    assert preds.shape == (10,), "Predictions should have shape (10,)"


def test_compute_model_metrics(sample_data, get_model):
    """
    Test the compute_model_metrics function.
    """
    _, _, X_test, y_test = sample_data
    model = get_model
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert isinstance(precision, float), "Precision should be a float"
    assert isinstance(recall, float), "Recall should be a float"
    assert isinstance(fbeta, float), "Fbeta should be a float"
    assert 0 <= precision <= 1, "Precision should be between 0 and 1"
    assert 0 <= recall <= 1, "Recall should be between 0 and 1"
    assert 0 <= fbeta <= 1, "Fbeta should be between 0 and 1"
