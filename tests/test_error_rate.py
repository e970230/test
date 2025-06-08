import numpy as np
from sklearn.metrics import mean_squared_error


def test_error_rate_per_label():
    true_labels = np.array([120, 1100, 1500, 120, 1100, 1500, 120, 1100, 1500])
    predicted_labels = np.array([110, 1005, 1450, 110, 1005, 1450, 100, 1005, 1450])
    unique_labels = np.unique(true_labels)
    mse_per_label = {
        label: mean_squared_error(
            true_labels[true_labels == label],
            predicted_labels[true_labels == label],
        )
        for label in unique_labels
    }
    expected = {120: 200.0, 1100: 9025.0, 1500: 2500.0}
    for label, expected_val in expected.items():
        assert np.isclose(mse_per_label[label], expected_val)
