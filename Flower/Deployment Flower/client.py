from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import flwr as fl
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(f"Training on {DEVICE}")
print(f"Flower {fl.__version__} / PyTorch {torch.__version__}")
disable_progress_bar()



def load_datasets():
    # 1. Cargar y preparar el dataset de entrenamiento desde el CSV
    df_train = pd.read_csv("/home/ubuntu/aprendizaje_federado/alto.csv")

    # Seleccionar características (X) y la variable objetivo (y)
    X_train = df_train[['load_X', 'load_Z', 'power_Z', 'speed_SPINDLE', 'override_SPINDLE', 'powerDrive_SPINDLE']].values
    y_train = df_train['consumo_potencia'].values

    # 2. Dividir en conjunto de entrenamiento y validación (80% entrenamiento, 20% validación)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 3. Normalizar los datos de entrada
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)
    X_val_scaled = scaler.transform(X_val_split)

    # Convertir los datos a tensores para PyTorch
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_split, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_split, dtype=torch.long)

    # Crear datasets de entrenamiento y validación
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # Crear DataLoader para dividir los datos en batches
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    return train_loader, val_loader

'''
# 2. Definir la red neuronal
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 16)  # Capa de entrada con 6 características, y capa oculta con 16 neuronas
        self.fc2 = nn.Linear(16, 3)   # Capa de salida con 3 clases (bajo, medio, alto)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # No aplicamos softmax aquí porque usamos CrossEntropyLoss
        return x
'''


'''
# 2. Definir la red neuronal
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 64)  # Capa de entrada con 6 características, y capa oculta con 16 neuronas
        self.fc2 = nn.Linear(64, 32)   # Capa de salida con 3 clases (bajo, medio, alto)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))  # Activación ReLU en la segunda capa
        x = self.fc3(x)  # Capa de salida (sin softmax, ya que CrossEntropyLoss lo incluye)
        return x
'''


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 64)  # Capa de entrada con 6 características y 64 neuronas
        self.fc2 = nn.Linear(64, 32)  # Capa oculta con 32 neuronas
        self.fc3 = nn.Linear(32, 3)   # Capa de salida con 3 clases (bajo, medio, alto)
        self.dropout = nn.Dropout(0.5)  # Dropout para regularización (apaga el 50% de las neuronas)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activación ReLU en la primera capa
        x = self.dropout(x)  # Aplicamos Dropout
        x = torch.relu(self.fc2(x))  # Activación ReLU en la segunda capa
        x = self.fc3(x)  # Capa de salida (sin softmax, ya que CrossEntropyLoss lo incluye)
        return x




def train(net, trainloader, epochs: int, verbose=False):
    criterion = nn.CrossEntropyLoss()  # Función de pérdida para clasificación multiclase
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in trainloader:  # Iterar sobre el DataLoader
            optimizer.zero_grad()  # Resetear gradientes
            outputs = net(X_batch)  # Hacer predicciones
            loss = criterion(outputs, y_batch)  # Calcular la pérdida
            loss.backward()  # Retropropagar el error
            optimizer.step()  # Actualizar los pesos

            # Acumular la pérdida y las métricas
            epoch_loss += loss.item()  # Acumular la pérdida de este batch
            #total += y_batch.size(0)  # Total de ejemplos en este batch
            #_, predicted = torch.max(outputs, 1)  # Obtener las predicciones
            #correct += (predicted == y_batch).sum().item()  # Comparar con etiquetas reales
            #correct += (torch.max(outputs.data, 1)[1] == y_batch).sum().item()

        # Calcular métricas promedio para la época actual
        epoch_loss /= len(trainloader)  # Promedio de la pérdida
        #epoch_acc = correct / total  # Precisión como proporción de aciertos

        #if verbose:
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {epoch_loss:.4f}")


def test(net, testloader):
    criterion = nn.CrossEntropyLoss()  # Función de pérdida
    val_loss = 0.0
    correct = 0
    total = 0
    net.eval()

    with torch.no_grad():  # Desactivar el cálculo de gradientes para ahorrar memoria y acelerar
        for X_batch, y_batch in testloader:  # Iterar sobre el DataLoader de prueba
            outputs = net(X_batch)  # Hacer predicciones
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

            # Predecir las clases con la mayor probabilidad
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    val_loss /= len(testloader)
    accuracy = correct / total
    print("VAL LOSS: " + str(val_loss))
    print("ACC:" + str(accuracy))
    return val_loss, accuracy



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
        #print("Returning:", get_parameters(self.net), len(self.trainloader.dataset))
        #print('SSSSSSSSSSSSSSSS')
        #print(type(get_parameters(self.net)))
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net,parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}



# Modelo preentrenado
PRETRAINED_MODEL_PATH = "pretrained_model_bnn.pth"

if __name__ == "__main__":
    # Carga el modelo preentrenado
    net = Net().to(DEVICE)
    #net.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
    #net.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))

    # Cargar los datos particionados para este cliente (cambiar el ID según el cliente)
    trainloader, valloader = load_datasets()

    # Iniciar el cliente y conectarse al servidor
    client = FlowerClient(net, trainloader, valloader)
    fl.client.start_client(server_address="10.98.101.104:8080", client=client.to_client())

