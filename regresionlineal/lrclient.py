import flwr as fl
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
from typing import List

# Cargar el dataset
def load_dataset():
    df = pd.read_csv("/home/ubuntu/aprendizaje_federado/federated_working_IBARMIA.csv")

    # Selecciona las características (X) y la variable a predecir (y)
    X = df[['load_X', 'load_Z', 'power_X', 'power_Z', 'load_SPINDLE', 'speed_SPINDLE', 'override_SPINDLE', 'powerDrive_SPINDLE']]
    y = df['potenciaKW']

    # Dividir en entrenamiento y validación (80% - 20%)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val



'''
# Funciones para obtener y establecer parámetros
def set_parameters(model: LinearRegression, parameters: List[np.ndarray]):
    #model.coef_ = parameters[:-1]
    #model.intercept_ = parameters[-1]
    model.coef_ = np.array(parameters[:-1])
    model.intercept_ = np.array(parameters[-1])

def get_parameters(model: LinearRegression) -> List[np.ndarray]:
    return np.concatenate((model.coef_, [model.intercept_])).flatten()
'''

def set_parameters(model: LinearRegression, parameters: List[np.ndarray]):
    # Establecemos el coeficiente (X parámetros) y el intercepto (último parámetro)
    model.coef_ = np.array(parameters[:-1])  # Todo menos el último parámetro son los coeficientes
    model.intercept_ = np.array(parameters[-1])  # El último parámetro es el intercepto
    #model.coef_ = np.array(parameters[:-1]).flatten()
    #model.intercept_ = np.array(parameters[-1]).item()


def get_parameters(model: LinearRegression) -> List[np.ndarray]:
    # Devolvemos los coeficientes y el intercepto como listas separadas
    return [model.coef_, model.intercept_]


# Definir el cliente de Flower
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_val, y_val, scaler):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.scaler = scaler

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        print("Received parameters:", parameters)
        # Establecer los parámetros en el modelo
        set_parameters(self.model, parameters)
        # Escalar los datos de entrenamiento
        #X_train_scaled = self.scaler.transform(self.X_train)
        X_train_scaled = self.scaler.transform(self.X_train.values)
        # Entrenar el modelo
        self.model.fit(X_train_scaled, self.y_train)
        # Obtener los parámetros actualizados
        params = get_parameters(self.model)
        print(params)
        # Calcular MSE en el conjunto de entrenamiento
        mse_train = mean_squared_error(self.y_train, self.model.predict(X_train_scaled))
        print(f"MSE on training data: {mse_train}")
        # Asegúrate de devolver los parámetros como listas, el tamaño de los datos y el MSE
        return params, len(self.X_train), {"mse": mse_train}

    def evaluate(self, parameters, config):
        print("Evaluating with parameters:", parameters)
        # Establecer parámetros en el modelo
        set_parameters(self.model, parameters)
        # Escalar los datos de validación
        #X_val_scaled = self.scaler.transform(self.X_val)
        X_val_scaled = self.scaler.transform(self.X_val.values)
        # Hacer predicciones en el conjunto de validación
        predictions = self.model.predict(X_val_scaled)
        # Calcular el MSE
        mse = mean_squared_error(self.y_val, predictions)
        print(f"MSE on validation data: {mse}")
        # Devolver el MSE y el tamaño del dataset de validación
        return float(mse), len(self.X_val), {"mse": mse}

    '''
    def fit(self, parameters, config):
        print("Received parameters:", parameters)
        # Establecer los parámetros en el modelo
        set_parameters(self.model, parameters)
        # Escalar los datos de entrenamiento
        X_train_scaled = self.scaler.transform(self.X_train)
        # Entrenar el modelo
        self.model.fit(X_train_scaled, self.y_train)
        # Obtener parámetros después de entrenar
        params = get_parameters(self.model).tolist()
        print('AAAAAAAAAAAAAA')
        print(type(params))
        #params = np.array(params)
        #print('BBBBBBBBBBBBB')
        #print(type(params))
        # Calcular MSE en el conjunto de entrenamiento
        mse_train = mean_squared_error(self.y_train, self.model.predict(X_train_scaled))
        #print("Returning:", params, len(self.X_train), {"mse": mse_train})
        # Asegurarse de que los parámetros son un ndarray y el mse es un flotante
        return params, len(self.X_train), {"mse": float(mse_train)}
        #return [], len(self.X_train), {}

    def evaluate(self, parameters, config):
        print('CCCCCCCCCCCC')
        # Establecer parámetros en el modelo
        set_parameters(self.model, parameters)
        # Escalar los datos de validación
        X_val_scaled = self.scaler.transform(self.X_val)
        # Hacer predicciones en el conjunto de validación
        predictions = self.model.predict(X_val_scaled)
        # Calcular el error cuadrático medio (MSE)
        mse = mean_squared_error(self.y_val, predictions)
        # Devolver el error y el tamaño del dataset de validación
        return float(mse), len(self.X_val), {"mse": float(mse)}
    '''



'''
# Definir el cliente de Flower
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_val, y_val, scaler):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.scaler = scaler

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        # Establecer los parámetros en el modelo
        set_parameters(self.model, parameters)
        # Escalar los datos de entrenamiento
        X_train_scaled = self.scaler.transform(self.X_train)
        # Entrenar el modelo
        self.model.fit(X_train_scaled, self.y_train)
        # Obtener parámetros después de entrenar
        params = get_parameters(self.model)
        # Calcular MSE en el conjunto de entrenamiento
        mse_train = mean_squared_error(self.y_train, self.model.predict(X_train_scaled))
        return params, len(self.X_train), {"mse": mse_train}




    def evaluate(self, parameters, config):
        # Escalar los datos de validación
        X_val_scaled = self.scaler.transform(self.X_val)
        # Hacer predicciones en el conjunto de validación
        predictions = self.model.predict(X_val_scaled)
        # Calcular el error cuadrático medio (MSE)
        mse = mean_squared_error(self.y_val, predictions)
        # Devolver el error y el tamaño del dataset de validación
        return float(mse), len(self.X_val), {"mse": float(mse)}
'''

if __name__ == "__main__":
    # Cargar el dataset
    X_train, X_val, y_train, y_val = load_dataset()

    # Cargar el modelo preentrenado desde el archivo .pkl
    model = joblib.load("/home/ubuntu/aprendizaje_federado/modelo_preentrenado_regresion_lineal.joblib")

    # Cargar el escalador desde el archivo .pkl
    scaler = joblib.load("/home/ubuntu/aprendizaje_federado/scaler_regresion_lineal.joblib")

    # Crear el cliente federado
    client = FederatedClient(model, X_train, y_train, X_val, y_val, scaler)
    # Iniciar el cliente y conectarse al servidor
    fl.client.start_client(server_address="10.98.101.104:8080", client=client.to_client())
