import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import syft as sy
from collections import defaultdict, OrderedDict
from typing import Tuple, Optional, List
import numpy.typing as npt

DEVICE = torch.device("cpu")  # Change to "cuda" to use GPU


# Aqui habria que poner la direccion de cada cliente
DATASITE_URLS = {
    "DataSite1": "http://10.98.101.120:54879",
    "DataSite2": "http://10.98.101.125:54880",
    "DataSite3": "http://10.98.101.123:54881",
    "DataSite4": "http://10.98.101.124:54882",
}

# Función para lanzar servidores DataSite
def spawn_server(sid: int):
    name = list(DATASITE_URLS.keys())[sid % len(DATASITE_URLS)]
    #print(DATASITE_URLS[name])
    client = sy.login(url=DATASITE_URLS[name],email="info@openmined.org", password="changethis")
    client.settings.allow_guest_signup(True)
    print(f"{name} está activo en {DATASITE_URLS[name]}")
    return client



def dl_experiment(data, model_params = None):
    def preprocess_data(data):
        X = data[['load_X', 'load_Z', 'power_Z', 'speed_SPINDLE', 'override_SPINDLE', 'powerDrive_SPINDLE']].values
        y = data['consumo_potencia'].values
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
        trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        valloader = DataLoader(val_dataset, batch_size=32)
        return trainloader, valloader

    # Define the neural network
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(6, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 3)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x



    def train(net, trainloader, epochs: int):
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
            # Calcular métricas promedio para la época actual
            epoch_loss /= len(trainloader)  # Promedio de la pérdida
            #epoch_acc = correct / total  # Precisión como proporción de aciertos

            #if verbose:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {epoch_loss:.4f}")
        # es el equivalente a get_parameters
        return {k: t.cpu().numpy() for k, t in net.state_dict().items()}

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
        #print("VAL LOSS: " + str(val_loss))
        print("ACC:" + str(accuracy))
        return val_loss, accuracy

    trloader, teloader = preprocess_data(data)

    net = Net()

    if model_params:  # convert to torch tensor and load
        net.load_state_dict({k: torch.from_numpy(v) for k, v in model_params.items()})

    net.to(DEVICE)
    model_params=train(net, trloader, 1)
    val_loss, accuracy = test(net,teloader)
    return model_params, accuracy





# Setup federated learning
def avg(all_models_params):
    return {param: np.mean([p[param] for p in all_models_params], axis=0)
            for param in all_models_params[0].keys()}


if __name__ == "__main__":
    FL_EPOCHS = 10

    #datasites = [spawn_server(0)[1], spawn_server(1)[1], spawn_server(2)[1], spawn_server(3)[1]]  # Lanzamiento de DataSites y clientes
    datasites = [spawn_server(0), spawn_server(1), spawn_server(2), spawn_server(3)]
    #datasites = [spawn_server(0)]

    model_params = dict()
    fl_metrics = defaultdict(list)  # one entry per epoch as a list
    for epoch in range(FL_EPOCHS):
        accs = []
        for datasite in datasites:
            data_asset = datasite.datasets["Custom Dataset"].assets["Consumo Potencia Data"]
            # print(type(data_asset)) # <class 'syft.service.dataset.dataset.Asset'>
            # Convertir el asset a DataFrame
            data_df = data_asset.data  # Asegúrate de que esto retorne un DataFrame
            #print(data_df) # esto es un dataframe
            params, acc = dl_experiment(data_df,model_params)
            fl_metrics[epoch].append(params)
            accs.append(acc)

        #### FALTA SACAR LA MEDIA DEL ACCURACCY Y EL LOSS ####
        avg_accuracy = np.mean(accs)
        print(f"Epoch {epoch + 1}/{FL_EPOCHS}, Mean Accuracy: {avg_accuracy:.4f}")


        model_params = avg([params for params in fl_metrics[epoch]])


    print("FIN")
