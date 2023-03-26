import sys
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import plotly.graph_objs as go
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

# Set up logging
logging.basicConfig(filename='model_logs.log', level=logging.DEBUG)

# Create a LSTMModel class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size) -> None:
        super(LSTMModel, self).__init__()
        logging.info("Initializing LSTM model.")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Get the output from the last time step
        out = out[:, -1, :]

        # Pass the output through the linear layer
        out = self.fc(out)

        return out

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

# Train the model
def train(model, train_loader, criterion, optimizer, device):
    logging.info("Training dataset.")
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    logging.info("Validating dataset.")
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def load_data(file):
    original_data = pd.read_csv(file)
    data = original_data.drop(columns=['dates'])  # Drop the 'dates' column
    data.dropna(inplace=True)
    data = data.values  # Convert DataFrame to NumPy array
    # Normalize the data
    scaled_data, scaler = normalize(data)
    return original_data, scaled_data, scaler

def convert_data(X_train, y_train, X_val, y_val, X_test, y_test, model):
    # Convert data to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    # Convert test data to PyTorch tensors
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoader for training, validation, and test data
    batch_size = 32
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = TensorDataset(X_val_t, y_val_t)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the loss function and the optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return train_loader, val_loader, test_loader, criterion, optimizer, device, model

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, early_stopping_patience):
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement == early_stopping_patience:
            print("Early stopping...")
            break

def main():
    original_data, scaled_data, scaler = load_data("data/exchange/coinbasepro/BTC_USD_1d.csv")

    window_size = 30
    X_train, y_train, X_val, y_val, X_test, y_test = split(scaled_data, window_size)

    input_size = X_train.shape[2]
    hidden_size = 64
    num_layers = 2
    output_size = 1

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    train_loader, val_loader, test_loader, criterion, optimizer, device, model = convert_data(X_train, y_train, X_val, y_val, X_test, y_test, model)

    # Train the model
    num_epochs = 100
    early_stopping_patience = 10

    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, early_stopping_patience)

    # Load the best model
    best_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    best_model.load_state_dict(torch.load("best_model.pt"))
    best_model = best_model.to(device)

    # Evaluate the model on the test set
    test_loss = validate(best_model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

    # Make predictions on the test set
    y_pred_list = []
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch[0].to(device)
            y_pred = best_model(X_batch)
            y_pred_list.extend(y_pred.squeeze().cpu().numpy())

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred_list)
    mae = mean_absolute_error(y_test, y_pred_list)
    r2 = r2_score(y_test, y_pred_list)

    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R-squared: {r2:.4f}")


if __name__ == "__main__":
    main()