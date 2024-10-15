import flwr as fl
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
from typing import List

# Cargar el dataset
def load_dataset():
    # Cambia el nombre del archivo por el del dataset federado que mencionas
    df = pd.read_csv("/home/ubuntu/aprendizaje_federado/federated_working_IBARMIA.csv")

    # Selecciona las características (X) y la variable a predecir (y)
    X = df[['load_X', 'load_Z', 'power_X', 'power_Z', 'load_SPINDLE', 'speed_SPINDLE', 'override_SPINDLE', 'powerDrive_SPINDLE']]
    y = df['potenciaKW']

    # Dividir en entrenamiento y validación (80% - 20%)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cargar el escalador desde archivo para que sea el mismo que se usó en el entrenamiento inicial
    with open("/home/ubuntu/aprendizaje_federado/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Escalar los datos
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, X_val, y_train, y_val

    '''
    # Escalar los datos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, X_val, y_train, y_val, scaler
    '''

# Definir el cliente de Flower
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_val, y_val):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def get_parameters(self, config):
        # Los modelos de scikit-learn no tienen un método directo para obtener parámetros
        # Por lo tanto, no hacemos nada en este caso
        return []

    def fit(self, parameters, config):
        # Entrenamos el modelo
        self.model.fit(self.X_train, self.y_train)
        # Devolver los parámetros y el tamaño del dataset de entrenamiento
        return [], len(self.X_train), {}

    def evaluate(self, parameters, config):
        # Hacer predicciones en el conjunto de validación
        predictions = self.model.predict(self.X_val)
        # Calcular el error cuadrático medio (MSE)
        mse = mean_squared_error(self.y_val, predictions)
        # Devolver el error y el tamaño del dataset de validación
        return float(mse), len(self.X_val), {"mse": float(mse)}

if __name__ == "__main__":
    '''
    # Cargar el dataset
    X_train, X_val, y_train, y_val, scaler = load_dataset()
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=4,
        min_samples_split=10,
        bootstrap=True,
        random_state=42
    )
    # Crear el cliente federado
    client = FederatedClient(model, X_train, y_train, X_val, y_val)
    '''

    # Cargar el dataset
    X_train, X_val, y_train, y_val = load_dataset()

    # Cargar el modelo preentrenado desde el archivo .pkl
    with open("/home/ubuntu/aprendizaje_federado/modelo_preentrenado_random_forest.pkl", "rb") as f:
        model = pickle.load(f)

    # Crear el cliente federado
    client = FederatedClient(model, X_train, y_train, X_val, y_val)

    # Iniciar el cliente y conectarse al servidor
    fl.client.start_client(server_address="10.98.101.104:8080", client=client.to_client())
