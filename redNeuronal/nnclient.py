import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
from flwr.client import NumPyClient
from typing import List, Tuple
from collections import OrderedDict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

DEVICE = torch.device("cpu")  # Cambiar a "cuda" si tienes GPU disponible
print(f"Training on {DEVICE}")

BATCH_SIZE = 32

# 1. Cargar y preparar el dataset
def load_dataset():
    df = pd.read_csv("/home/ubuntu/aprendizaje_federado/federated_working_IBARMIA.csv")  # Actualiza con tu ruta

    # Seleccionar características (X) y la variable objetivo (y)
    X = df[['load_X', 'load_Z', 'power_X', 'power_Z', 'load_SPINDLE', 'speed_SPINDLE', 'override_SPINDLE', 'powerDrive_SPINDLE']].values
    y = df['potenciaKW'].values

    # Dividir en conjunto de entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val

# 2. Convertir los datos a tensores y cargarlos en DataLoader
def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    # Normalizamos las características
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

    # Convertimos los datos a tensores
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train_scaled, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), torch.tensor(y_val_scaled, dtype=torch.float32))

    # Creamos los DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, scaler_X, scaler_y


# 3. Definición de la red neuronal
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(8, 16)  # 8 características de entrada
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 1)  # 1 valor de salida para regresión

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 4. Función de entrenamiento
def train(net, trainloader, epochs: int, verbose=False):
    """Entrena la red neuronal en el conjunto de entrenamiento."""
    criterion = nn.MSELoss()  # Función de pérdida para regresión
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in trainloader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))  # Ajuste para que las dimensiones coincidan
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(trainloader)
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}")







# 5. Función de validación/test
def test(net, valloader):
    """Evalúa la red neuronal en el conjunto de validación."""
    criterion = nn.MSELoss()
    net.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in valloader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = net(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))  # Ajuste para que las dimensiones coincidan
            val_loss += loss.item()

    val_loss /= len(valloader)
    return val_loss






def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net,parameters)
        train(self.net, self.trainloader, 1)
        print("Returning:", get_parameters(self.net), len(self.trainloader.dataset))
        print('SSSSSSSSSSSSSSSS')
        print(type(get_parameters(self.net)))
        return get_parameters(self.net), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net,parameters)
        loss = test(self.net, self.valloader)
        return float(loss), len(self.valloader.dataset), {"loss": float(loss)}



# Modelo preentrenado
PRETRAINED_MODEL_PATH = "pretrained_regression_model.pth"

if __name__ == "__main__":
    # Carga el modelo preentrenado
    net = Net().to(DEVICE)
    #net.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
    net.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))

    X_train, X_val, y_train, y_val = load_dataset()
    trainloader, valloader, scaler_X, scaler_y = create_data_loaders(X_train, y_train, X_val, y_val)

    # Iniciar el cliente y conectarse al servidor
    client = FlowerClient(net, trainloader, valloader)
    fl.client.start_client(server_address="10.98.101.104:8080", client=client.to_client())
