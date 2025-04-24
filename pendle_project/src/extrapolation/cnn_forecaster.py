# CNN dummy forecast (rebuild marker)

import numpy as np

def dummy_cnn_forecast(pca_sequence, forecast_steps=5):
    direction = np.gradient(pca_sequence[:, 0])[-1]
    forecast = [pca_sequence[-1] + direction * i for i in range(1, forecast_steps + 1)]
    return np.array(forecast)