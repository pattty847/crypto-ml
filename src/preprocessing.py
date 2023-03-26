# Set up logging
import logging
import numpy as np

from sklearn.preprocessing import MinMaxScaler


logging.basicConfig(filename='model_logs.log', level=logging.DEBUG)

# Normalize the data
def normalize(data):
    logging.info("Normalizing data.")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(data), scaler

# Denormalize the data
def denormalize(data, scaler):
    logging.info("Denormalizing data.")
    return scaler.inverse_transform(data.reshape(-1, 1)).flatten()

# Create sliding window dataset
def create_sliding_window_dataset(data, window_size):
    logging.info("Creating sliding window dataset.")
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, :])
        y.append(data[i, 3])  # Use the "close" price as target
    return np.array(X), np.array(y)

# Split the dataset into training and validation sets
def split(data, window_size):
    logging.info("Splitting dataset.")
    X, y = create_sliding_window_dataset(data, window_size)
    train_size = int(len(X) * 0.6)
    val_size = int(len(X) * 0.2)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
    return X_train, y_train, X_val, y_val, X_test, y_test