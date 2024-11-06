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
        "DataSite1": "/home/ubuntu/pysyft/alto.csv",
    }
    file_path = dataset_paths.get(name)
    if not file_path or not os.path.exists(file_path):
        print(f"No se encontró el archivo de datos para {name}")
        return None
    return pd.read_csv(file_path)

# Aqui habria que poner la direccion de cada cliente
DATASITE_URLS = {
    "DataSite1": "http://10.98.101.120:54879",
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
    return data_site

spawn_server(0)
