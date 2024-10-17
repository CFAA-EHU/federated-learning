import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import uniform

# Dispositivo de entrenamiento
DEVICE = torch.device("cpu")  # Cambiar a "cuda" si tienes GPU disponible


# Cargar y preparar el dataset
def load_dataset():
    df = pd.read_csv("C:\\Users\\836582\\Downloads\\federated_working_powerZ_GMTK.csv")  # Actualiza con tu ruta
    X = df[['load_X', 'load_Z', 'power_Z', 'speed_SPINDLE', 'override_SPINDLE','powerDrive_SPINDLE']].values
    y = df['potenciaKW'].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val


# Convertir los datos a tensores y escalarlos
def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                  torch.tensor(y_train_scaled, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32),
                                torch.tensor(y_val_scaled, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, scaler_X, scaler_y


# Definir la red neuronal con una capa oculta
class Net(nn.Module):
    def __init__(self, input_size=6, hidden_size=16, output_size=1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Función de entrenamiento
def train(net, trainloader, epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    net.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(trainloader)}")


# Función de evaluación
def evaluate(net, valloader):
    net.eval()
    criterion = nn.MSELoss()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in valloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = net(inputs)
            loss = criterion(outputs.squeeze(), targets)
            val_loss += loss.item()
    return val_loss / len(valloader)


# Hiperparámetros para optimizar
def objective(hidden_size, learning_rate, batch_size, epochs):
    net = Net(hidden_size=hidden_size).to(DEVICE)
    X_train, X_val, y_train, y_val = load_dataset()
    train_loader, val_loader, scaler_X, scaler_y = create_data_loaders(X_train, y_train, X_val, y_val,
                                                                       batch_size=batch_size)

    train(net, train_loader, epochs, learning_rate)
    val_loss = evaluate(net, val_loader)
    print(f"Validation Loss: {val_loss}")
    return val_loss


# Ejecutar Random Search
from scipy.stats import randint

# Definir los parámetros para la búsqueda
param_grid = {
    'hidden_size': randint(4, 64),  # Neuronas en la capa oculta
    'learning_rate': uniform(0.0001, 0.01),  # Tasa de aprendizaje
    'batch_size': randint(16, 128),  # Tamaño del batch
    'epochs': randint(10, 100),  # Número de épocas
}

# Crear Random Search
n_iter_search = 10  # Número de iteraciones de búsqueda
best_loss = float('inf')
best_params = None

for i in range(n_iter_search):
    # Seleccionar una combinación aleatoria de parámetros
    hidden_size = param_grid['hidden_size'].rvs()
    learning_rate = param_grid['learning_rate'].rvs()
    batch_size = param_grid['batch_size'].rvs()
    epochs = param_grid['epochs'].rvs()

    print(
        f"Trial {i + 1}/{n_iter_search}: hidden_size={hidden_size}, learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}")

    # Ejecutar el entrenamiento y evaluación con los parámetros actuales
    val_loss = objective(hidden_size, learning_rate, batch_size, epochs)

    # Guardar los mejores parámetros
    if val_loss < best_loss:
        best_loss = val_loss
        best_params = (hidden_size, learning_rate, batch_size, epochs)

print(
    f"Best parameters: hidden_size={best_params[0]}, learning_rate={best_params[1]}, batch_size={best_params[2]}, epochs={best_params[3]}")
print(f"Best validation loss: {best_loss}")
