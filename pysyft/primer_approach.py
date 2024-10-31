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



# Cargar datos y URLs de DataSites
def load_data(name: str) -> Optional[pd.DataFrame]:
    dataset_paths = {
        "DataSite1": "C:\\Users\\836582\\Downloads\\alto.csv",
        "DataSite2": "C:\\Users\\836582\\Downloads\\bajo.csv",
    }
    file_path = dataset_paths.get(name)
    if not file_path or not os.path.exists(file_path):
        print(f"No se encontró el archivo de datos para {name}")
        return None
    return pd.read_csv(file_path)

# Aqui habria que poner la direccion de cada cliente
DATASITE_URLS = {
    "DataSite1": "http://localhost:54879",
    "DataSite2": "http://localhost:54880"
}

# Generación de mock data
def generate_mock(data: pd.DataFrame, seed: int = 12345) -> pd.DataFrame:
    np.random.seed(seed=seed)
    mock_data = data.sample(n=len(data), replace=True).reset_index(drop=True)
    for column in data.columns:
        true_na_rate = data[column].isna().mean()
        if true_na_rate > 0:
            na_indices = np.random.choice(mock_data.index, size=int(true_na_rate * len(data)), replace=False)
            mock_data.loc[na_indices, column] = np.nan
    return mock_data

# Creación del dataset Syft
def create_syft_dataset(name: str) -> Optional[sy.Dataset]:
    data = load_data(name=name)
    if data is None:
        return None
    dataset = sy.Dataset(
        name="Custom Dataset",
        summary=f"Dataset de consumo de potencia para {name}",
        description=f"Dataset cargado desde {name} con datos de consumo de potencia.",
    )
    dataset.add_asset(
        sy.Asset(
            name="Consumo Potencia Data",
            data=data,
            mock=generate_mock(data, seed=len(name))
        )
    )
    return dataset

# Función para lanzar servidores DataSite
def spawn_server(sid: int):
    name = list(DATASITE_URLS.keys())[sid % len(DATASITE_URLS)]
    data_site = sy.orchestra.launch(
        name=name,
        port=int(DATASITE_URLS[name].split(":")[-1]),
        reset=True,
        n_consumers=1,
        create_producer=True,
    )
    client = data_site.login(email="info@openmined.org", password="changethis")
    client.settings.allow_guest_signup(True)
    ds = create_syft_dataset(name)
    if ds:
        client.upload_dataset(ds)
    print(f"{name} está activo en {DATASITE_URLS[name]}")
    return data_site, client




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




# Setup federated learning
def avg(all_models_params):
    return {param: np.mean([p[param] for p in all_models_params], axis=0)
            for param in all_models_params[0].keys()}



if __name__ == "__main__":
    FL_EPOCHS = 10

    datasites = [spawn_server(0)[1], spawn_server(1)[1]]  # Lanzamiento de DataSites y clientes

    net = Net().to(DEVICE)

    for epoch in range(FL_EPOCHS):
        for datasite in datasites:
            data_asset = datasite.datasets["Custom Dataset"].assets["Consumo Potencia Data"]
            # print(type(data_asset)) # <class 'syft.service.dataset.dataset.Asset'>
            # Convertir el asset a DataFrame
            data_df = data_asset.data  # Asegúrate de que esto retorne un DataFrame
            print(data_df) # esto es un dataframe
            trloader, teloader = preprocess_data(data_df)
            train(net, trloader, 1)
            acc = test(net, teloader)

    print("FIN")




'''
from collections import defaultdict
from typing import Optional, TypeVar, Union, Any, List, OrderedDict

import syft as sy

import os
import pandas as pd

from threading import Thread, Event, current_thread
import os
from time import sleep
from typing import Optional
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# training (model) using torch
import torch
import torch.nn as nn
import torch.optim as optim
import numpy.typing as npt

# evaluation (metrics)
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import confusion_matrix

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU

# Cargar datos y URLs de DataSites
def load_data(name: str) -> Optional[pd.DataFrame]:
    dataset_paths = {
        "DataSite1": "C:\\Users\\836582\\Downloads\\alto.csv",
        "DataSite2": "C:\\Users\\836582\\Downloads\\bajo.csv",
    }
    file_path = dataset_paths.get(name)
    if not file_path or not os.path.exists(file_path):
        print(f"No se encontró el archivo de datos para {name}")
        return None
    return pd.read_csv(file_path)

DATASITE_URLS = {
    "DataSite1": "http://localhost:54879",
    "DataSite2": "http://localhost:54880"
}

# Generación de mock data
def generate_mock(data: pd.DataFrame, seed: int = 12345) -> pd.DataFrame:
    np.random.seed(seed=seed)
    mock_data = data.sample(n=len(data), replace=True).reset_index(drop=True)
    for column in data.columns:
        true_na_rate = data[column].isna().mean()
        if true_na_rate > 0:
            na_indices = np.random.choice(mock_data.index, size=int(true_na_rate * len(data)), replace=False)
            mock_data.loc[na_indices, column] = np.nan
    return mock_data

# Creación del dataset Syft
def create_syft_dataset(name: str) -> Optional[sy.Dataset]:
    data = load_data(name=name)
    if data is None:
        return None
    dataset = sy.Dataset(
        name="Custom Dataset",
        summary=f"Dataset de consumo de potencia para {name}",
        description=f"Dataset cargado desde {name} con datos de consumo de potencia.",
    )
    dataset.add_asset(
        sy.Asset(
            name="Consumo Potencia Data",
            data=data,
            mock=generate_mock(data, seed=len(name))
        )
    )
    return dataset

# Función para lanzar servidores DataSite
def spawn_server(sid: int):
    name = list(DATASITE_URLS.keys())[sid % len(DATASITE_URLS)]
    data_site = sy.orchestra.launch(
        name=name,
        port=int(DATASITE_URLS[name].split(":")[-1]),
        reset=True,
        n_consumers=2,
        create_producer=True,
    )
    client = data_site.login(email="info@openmined.org", password="changethis")
    client.settings.allow_guest_signup(True)
    ds = create_syft_dataset(name)
    if ds:
        client.upload_dataset(ds)
    print(f"{name} está activo en {DATASITE_URLS[name]}")
    return data_site, client



def dl_experiment(data: DataFrame, model_params: ModelParams = None, training_epochs: int = 10) -> Result:
    # Preprocesamiento y creación de loaders de datos
    def preprocess():
        df = pd.read_csv("C:\\Users\\836582\\Downloads\\alto.csv")
        X = df[['load_X', 'load_Z', 'power_Z', 'speed_SPINDLE', 'override_SPINDLE', 'powerDrive_SPINDLE']].values
        y = df['consumo_potencia'].values
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
        trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        valloader = DataLoader(val_dataset, batch_size=32)
        return trainloader, valloader

    # Modelo
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

    # Entrenamiento
    def train(net, trainloader, epochs=1):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        net.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in trainloader:
                optimizer.zero_grad()
                outputs = net(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        return get_parameters(net), len(trainloader), {}


    # Testing
    def test(net, testloader):
        correct, total = 0, 0
        net.eval()
        with torch.no_grad():
            for X_batch, y_batch in testloader:
                outputs = net(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        accuracy = correct / total
        return accuracy

    def set_parameters(net, parameters: List[np.ndarray]):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def get_parameters(net) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in net.state_dict().items()]







# Función de promedio para parámetros del modelo
def avg(all_models_params):
    return {param: np.mean([p[param] for p in all_models_params], axis=0)
            for param in all_models_params[0].keys()}




if __name__ == "__main__":
    # Configuración de FL
    FL_EPOCHS = 10
    fl_model_params, fl_metrics = None, defaultdict(list)

    datasites = [spawn_server(0)[1], spawn_server(1)[1]]  # Lanzamiento de DataSites y clientes

    for epoch in range(FL_EPOCHS):
        for datasite in datasites:
            data_asset = datasite.datasets["Custom Dataset"].assets["Consumo Potencia Data"]
            metrics, params = datasite.code.dl_experiment(data=data_asset, model_params=fl_model_params).get()
            fl_metrics[epoch].append((metrics, params))
        fl_model_params = avg([params for _, params in fl_metrics[epoch]])

    print(f"Parámetros finales tras {FL_EPOCHS} rondas de FL: ", fl_model_params)
'''






'''
# Configuración de rutas de datasets y URLs de DataSites
def load_data(name: str) -> Optional[pd.DataFrame]:
    dataset_paths = {
        "DataSite1": "C:\\Users\\836582\\Downloads\\alto.csv",
        "DataSite2": "C:\\Users\\836582\\Downloads\\bajo.csv",
    }
    file_path = dataset_paths.get(name)
    if not file_path or not os.path.exists(file_path):
        print(f"No se encontró el archivo de datos para {name}")
        return None
    return pd.read_csv(file_path)

DATASITE_URLS = {
    "DataSite1": "http://localhost:54879",
    "DataSite2": "http://localhost:54880"


}


def generate_mock(data: pd.DataFrame, seed: int = 12345) -> pd.DataFrame:
    """Genera un mock dataset a partir de los datos reales, manteniendo estructura y tasas de valores faltantes."""
    np.random.seed(seed=seed)
    mock_n_samples = len(data)
    mock_data = data.sample(n=mock_n_samples, replace=True).reset_index(drop=True)

    # Mantén valores NaN en las mismas proporciones que en el dataset original
    for column in data.columns:
        true_na_rate = data[column].isna().mean()
        if true_na_rate > 0:
            na_indices = np.random.choice(
                mock_data.index, size=int(true_na_rate * mock_n_samples), replace=False
            )
            mock_data.loc[na_indices, column] = np.nan

    return mock_data
def create_syft_dataset(name: str) -> Optional[sy.Dataset]:
    data = load_data(name=name)
    if data is None:
        return None
    dataset = sy.Dataset(
        name="Custom Dataset",
        summary=f"Dataset de consumo de potencia para {name}",
        description=f"Dataset cargado desde {name} con datos de consumo de potencia.",
    )
    dataset.add_asset(
        sy.Asset(
            name="Consumo Potencia Data",
            data=data,
            mock=generate_mock(data, seed=len(name))
        )
    )
    return dataset
# Función para lanzar servidores DataSite
def spawn_server(sid: int):
    name = list(DATASITE_URLS.keys())[sid % len(DATASITE_URLS)]
    data_site = sy.orchestra.launch(
        name=name,
        port=int(DATASITE_URLS[name].split(":")[-1]),
        reset=True,
        n_consumers=2,
        create_producer=True,
    )
    client = data_site.login(email="info@openmined.org", password="changethis")
    client.settings.allow_guest_signup(True)
    ds = create_syft_dataset(name)
    if ds:
        client.upload_dataset(ds)
    print(f"{name} está activo en {DATASITE_URLS[name]}")
    return data_site, client


# Cargar datos para cada cliente
def preprocess():
    df = pd.read_csv("/home/ubuntu/pysyft/alto.csv")
    X = df[['load_X', 'load_Z', 'power_Z', 'speed_SPINDLE', 'override_SPINDLE', 'powerDrive_SPINDLE']].values
    y = df['consumo_potencia'].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=32)
    return trainloader, valloader

# Definir el modelo
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

# Función para entrenar el modelo
def train(net, trainloader, epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in trainloader:
            optimizer.zero_grad()
            outputs = net(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

def test(net, testloader):
    criterion = nn.CrossEntropyLoss()  # Función de pérdida
    correct, total, test_loss = 0, 0, 0.0
    net.eval()

    with torch.no_grad():  # Desactivar el cálculo de gradientes para ahorrar memoria y acelerar
        for X_batch, y_batch in testloader:  # Iterar sobre el DataLoader de prueba
            outputs = net(X_batch)  # Hacer predicciones
            loss = criterion(outputs, y_batch)  # Calcular la pérdida
            test_loss += loss.item()  # Acumular la pérdida

            # Predecir las clases con mayor probabilidad
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)  # Total de ejemplos procesados
            correct += (predicted == y_batch).sum().item()  # Acumular los aciertos

    # Calcular la pérdida y la precisión promedio
    test_loss /= len(testloader.dataset)  # Promedio de la pérdida
    accuracy = correct / total  # Precisión como proporción de aciertos
    return test_loss, accuracy

# Custom average function for model parameters
def avg(all_models_params):
    return {param: np.average([p[param] for p in all_models_params], axis=0)
            for param in all_models_params[0].keys()}


FL_EPOCHS=10

# Federated training setup
fl_model_params, fl_metrics = None, defaultdict(list)

for epoch in range(FL_EPOCHS):
    for datasite in datasites:
        # Retrieve specific data from each datasite
        data_asset = datasite.datasets["Custom Dataset"].assets["MyData"]

        # Train and retrieve model parameters and metrics
        metrics, params = datasite.code.dl_experiment(data=data_asset, model_params=fl_model_params).get()
        fl_metrics[epoch].append((metrics, params))

    # Aggregate parameters across datasites
    fl_model_params = avg([params for _, params in fl_metrics[epoch]])

print(f"Final aggregated parameters after {FL_EPOCHS} epochs: ", fl_model_params)


if __name__ == "__main__":
    spawn_server(0)
    spawn_server(1)
'''

'''
# Clase para hilos que verifican solicitudes de clientes
class DataSiteThread(Thread):
    def __init__(self, *args, **kwargs):
        super(DataSiteThread, self).__init__(*args, **kwargs)
        self._stop_event = Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

# Lanzamiento de DataSites y monitoreo de solicitudes
def launch_datasites():
    data_sites = []
    client_threads = []

    for sid in range(2):
        data_site, client = spawn_server(sid=sid)
        data_sites.append(data_site)
        client_threads.append(
            DataSiteThread(
                target=check_and_approve_incoming_requests, args=(client,), daemon=True
            )
        )
    for t in client_threads:
        t.start()

    try:
        while True:
            sleep(2)
    except KeyboardInterrupt:
        for data_site in data_sites:
            data_site.land()
        for t in client_threads:
            t.stop()

def check_and_approve_incoming_requests(client):
    while not current_thread().stopped():
        requests = client.requests
        for r in filter(lambda r: r.status.value != 2, requests):  # 2 == APROBADO
            r.approve(approve_nested=True)
        sleep(1)


if __name__ == "__main__":
    launch_datasites()
'''


'''
def load_data(name: str) -> Optional[pd.DataFrame]:
    """Carga el CSV correspondiente según el nombre del DataSite."""
    dataset_paths = {
        "DataSite1": "C:\\Users\\836582\\Downloads\\alto.csv",
        "DataSite2": "C:\\Users\\836582\\Downloads\\bajo.csv",
    }
    file_path = dataset_paths.get(name)

    if not file_path or not os.path.exists(file_path):
        print(f"No se encontró el archivo de datos para {name}")
        return None

    df = pd.read_csv(file_path)
    return df


DATASITE_URLS = {
    "DataSite1": "http://localhost:54879",
    "DataSite2": "http://localhost:54880"
}


def create_syft_dataset(name: str) -> Optional[sy.Dataset]:
    """Crea un dataset de Syft para cada DataSite usando el archivo CSV correspondiente."""
    data = load_data(name=name)
    if data is None:
        return None

    dataset = sy.Dataset(
        name="Custom Dataset",
        summary=f"Dataset de consumo de potencia para {name}",
        description=f"Dataset cargado desde {name} con datos de consumo de potencia.",
    )
    dataset.add_asset(
        sy.Asset(
            name="Consumo Potencia Data",
            data=data,
        )
    )
    return dataset


def spawn_server(sid: int):
    """Lanza una instancia de un DataSite usando el nombre y puerto de DATASITE_URLS."""
    name = list(DATASITE_URLS.keys())[sid % len(DATASITE_URLS)]

    data_site = sy.orchestra.launch(
        name=name,
        port=DATASITE_URLS[name].split(":")[-1],
        reset=True,
        n_consumers=2,
        create_producer=True,
    )
    client = data_site.login(email="info@openmined.org", password="changethis")

    client.settings.allow_guest_signup(True)
    ds = create_syft_dataset(name)
    if ds:
        client.upload_dataset(ds)

    print(f"{name} está activo en {DATASITE_URLS[name]}")
    return data_site, client
'''

'''
import numpy as np
from collections import defaultdict
from syft.client import Client


# Custom average function for model parameters
def avg(all_models_params):
    return {param: np.average([p[param] for p in all_models_params], axis=0)
            for param in all_models_params[0].keys()}


# Connect to datasites
datasite1 = Client("http://datasite1:port")  # Replace with actual IP and port for site 1
datasite2 = Client("http://datasite2:port")  # Replace with actual IP and port for site 2
datasites = [datasite1, datasite2]

# Federated training setup
fl_model_params, fl_metrics = None, defaultdict(list)

for epoch in range(FL_EPOCHS):
    for datasite in datasites:
        # Retrieve specific data from each datasite
        data_asset = datasite.datasets["Custom Dataset"].assets["MyData"]

        # Train and retrieve model parameters and metrics
        metrics, params = datasite.code.ml_experiment(data=data_asset, model_params=fl_model_params).get()
        fl_metrics[epoch].append((metrics, params))

    # Aggregate parameters across datasites
    fl_model_params = avg([params for _, params in fl_metrics[epoch]])

print(f"Final aggregated parameters after {FL_EPOCHS} epochs: ", fl_model_params)
'''




'''
# Configuración del servidor
server = sy.orchestra.launch(
    name="my-datasite",
    port=8080,
    create_producer=True,
    n_consumers=1,
    dev_mode=False,
    reset=True
)


#client= server.login(email="client@server1.com", password="password")
#client.settings.allow_guest_signup(True)

# Confirmación de inicio del servidor
print("Servidor de PySyft en ejecución en http://10.98.101.115:8080")
'''

'''
import numpy as np
import numpy.typing as npt
from typing import Union, TypeVar, Any

import torch as th

# preprocessing
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from torch.utils.data import TensorDataset

# training (model) using torch
import torch.nn as nn
import torch.optim as optim

# evaluation (metrics)
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import confusion_matrix

DataFrame = TypeVar("pandas.DataFrame")
NDArray = npt.NDArray[Any]
NDArrayInt = npt.NDArray[np.int_]
NDArrayFloat = npt.NDArray[np.float_]

Dataset = TypeVar("torch.utils.data.TensorDataset")  # NEW!
Metric = TypeVar("Metric", bound=dict[str, Union[float, NDArrayInt]])
Metrics = TypeVar("Metrics", bound=tuple[Metric, Metric])  # train and test
ModelParams = TypeVar("ModelParams", bound=dict[str, NDArrayFloat])
Result = TypeVar("Result", bound=tuple[Metrics, ModelParams])


def dl_experiment(data: DataFrame, model_params: ModelParams = None, training_epochs: int = 10) -> Result:
    """DL Experiment using a Multi-layer Perceptron (non-linear) Classifier.
    Steps:
    1. Preprocessing (partitioning; missing values & scaling; convert to Tensor objects)
    2. MLP model definition and init (w/ `model_params`)
    3. Training loop w/ Loss & Optimizer. Gather updated model parameters.
    4. Evaluation: collect metrics on training and test partitions.

    Parameters
    ----------
    data : DataFrame
        Input Heart Study data represented as Pandas DataFrame.
    model_params: ModelParams (dict)
        DL Model Parameters
    training_epochs : int (default = 10)
        Number of training epochs

    Returns
    -------
        metrics : Metrics
            Evaluation metrics (i.e. MCC, Confusion matrix) on both training and test
            data partitions.
        model_params : ModelParams
            Update model params after training (converted as Numpy NDArrays)
    """


    def preprocess(data: DataFrame) -> tuple[Dataset, Dataset]:

        def by_demographics(data: DataFrame) -> NDArray:
            sex = data["sex"].map(lambda v: "0" if v == 0 else "1")
            target = data["num"].map(lambda v: "0" if v == 0 else "1")
            return (sex + target).values

        # Convert all data to float32 arrays for cross torck-kernels compatibility.
        X = data.drop(columns=["age", "sex", "num"], axis=1)
        y = (data["num"].map(lambda v: 0 if v == 0 else 1)).to_numpy(dtype=np.float32)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=12345, stratify=by_demographics(data))

        preprocessor = ColumnTransformer(
            transformers=[("numerical",
                           Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")),
                                           ("scaler", RobustScaler()), ]),
                           ["trestbps", "chol", "thalach", "oldpeak"]),
                          ("categorical",
                           SimpleImputer(strategy="most_frequent", ),
                           ["ca", "cp", "exang", "fbs", "restecg", "slope", "thal"])])

        X_train = preprocessor.fit_transform(X_train).astype(np.float32)
        X_test = preprocessor.transform(X_test).astype(np.float32)
        # Convert to torch tensor
        X_train, X_test = th.from_numpy(X_train), th.from_numpy(X_test)
        y_train, y_test = th.from_numpy(y_train), th.from_numpy(y_test)
        # reshape tensor to add batch dimension
        y_train, y_test = y_train.view(y_train.shape[0], 1), y_test.view(y_test.shape[0], 1)
        # gather all data in TensorDataset
        return TensorDataset(X_train, y_train), TensorDataset(X_test, y_test)

    def train(model: nn.Module, device: th.device, training_data: "TensorDataset") -> ModelParams:
        train_loader = th.utils.data.DataLoader(training_data, shuffle=True,
                                                batch_size=min(200, len(training_data)))
        # Loss and optimizer
        learning_rate = 0.001
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # Train the model
        for epoch in range(training_epochs):
            for data in train_loader:
                X_train, y_train = data[0].to(device), data[1].to(device)
                y_pred = model(X_train)
                loss = criterion(y_pred, y_train)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        # return model parameters as NDArray
        return {k: t.cpu().numpy() for k, t in model.state_dict().items()}

    def evaluate(model: nn.Module, device: th.device, dataset: "TensorDataset") -> Metric:
        data_loader = th.utils.data.DataLoader(dataset, batch_size=min(200, len(dataset)))
        with th.no_grad():
            y_true, y_pred = [], []
            for data in data_loader:
                X_val, y_val = data[0].to(device), data[1].to(device)
                y_pred.append(model(X_val).round().cpu().detach().numpy())
                y_true.append(y_val.cpu().numpy())
            y_pred = np.vstack(y_pred)
            y_true = np.vstack(y_true)
        return {"mcc": mcc(y_true, y_pred), "cm": confusion_matrix(y_true, y_pred)}

    # -- DL Experiment --
    # 1. preprocessing
    training_data, test_data = preprocess(data)

    # 2. model setup
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(11, 22)
            self.fc2 = nn.Linear(22, 11)
            self.classifier = nn.Linear(11, 1)

        def forward(self, x):
            x = th.sigmoid(self.fc1(x))
            x = th.sigmoid(self.fc2(x))
            return th.sigmoid(self.classifier(x))

    clf = MLP()
    if model_params:  # convert to torch tensor and load
        clf.load_state_dict({k: th.from_numpy(v) for k, v in model_params.items()})

    device = th.device("cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu")
    clf.to(device)
    # 3. training
    model_params = train(clf, device, training_data)
    # 4. evaluation
    training_metrics = evaluate(clf, device, training_data)
    test_metrics = evaluate(clf, device, test_data)
    return (training_metrics, test_metrics), model_params
'''