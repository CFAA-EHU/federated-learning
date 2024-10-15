import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
import sklearn

print(sklearn.__version__)

# Cargar datasets
data_machine1 = pd.read_csv('/home/ubuntu/aprendizaje_federado/federated_working_IBARMIA.csv')
data_machine2 = pd.read_csv('/home/ubuntu/aprendizaje_federado/federated_working_GMTK.csv')

# Unir ambos datasets
data_combined = pd.concat([data_machine1, data_machine2])

# Definir características y variable objetivo
features = ['load_X', 'load_Z', 'power_X', 'power_Z', 'load_SPINDLE', 'speed_SPINDLE', 'override_SPINDLE', 'powerDrive_SPINDLE']
target = 'potenciaKW'

X = data_combined[features].values
y = data_combined[target].values

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Crear el modelo con los mejores hiperparámetros
best_params = {'bootstrap': True, 'max_depth': 20, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 200}
model = RandomForestRegressor(**best_params)

# Entrenar el modelo
model.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f"Error cuadrático medio (MSE): {mse}")

# Guardar el modelo y el escalador
#joblib.dump(model, 'modelo_preentrenado_random_forest.pkl')
#joblib.dump(scaler, 'scaler.pkl')

# Guardar el modelo con pickle
with open("modelo_preentrenado_random_forest.pkl", "wb") as f:
    pickle.dump(model, f)

# Guardar el escalador con pickle
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Modelo y escalador guardados correctamente.")
