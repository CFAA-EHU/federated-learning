# 1. Importar las librerías necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler



'''
# 2. Cargar el dataset
# Asegúrate de modificar el path según tu archivo
data = pd.read_csv('C:\\Users\\836582\\Downloads\\federated_working_IBARMIA.csv')



# 3. Verifica el dataset
print(data.head())

# 4. Seleccionar las columnas de entrada (features) y la columna de salida (target)
# Features: Las variables que usarás para predecir
X = data[['load_X', 'load_Z', 'power_X', 'power_Z', 'load_SPINDLE',
          'speed_SPINDLE', 'override_SPINDLE', 'powerDrive_SPINDLE']]

# Target: La variable que queremos predecir
y = data['potenciaKW']

# 5. Escalar los datos (opcional pero recomendado para modelos basados en distancias como RandomForest)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Separar los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. Crear y entrenar el modelo de Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# 9. Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R2 Score: {r2}")

# 10. Mostrar algunas predicciones vs valores reales
resultados = pd.DataFrame({'Real': y_test, 'Predicción': y_pred})
print(resultados.head())
'''


## para encontrar los mejores hiperparmetros usar gridsearch




'''
# Cargar el dataset
data = pd.read_csv('C:\\Users\\836582\\Downloads\\federated_working_IBARMIA.csv')

# Seleccionar las features y el target
X = data[['load_X', 'load_Z', 'power_X', 'power_Z', 'load_SPINDLE',
          'speed_SPINDLE', 'override_SPINDLE', 'powerDrive_SPINDLE']]
y = data['potenciaKW']

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Definir el modelo
rf = RandomForestRegressor(random_state=42)

# Definir los hiperparámetros a probar en GridSearchCV
param_grid = {
    'n_estimators': [100, 200],           # Número de árboles en el bosque
    'max_depth': [10, 20, 30],          # Profundidad máxima de los árboles
    'min_samples_split': [2, 5, 10],          # Mínimo de muestras para dividir un nodo
    'min_samples_leaf': [2, 4],            # Mínimo de muestras para ser hoja
    'bootstrap': [True, False]                # Si se usa el bootstrapping en los árboles
}

# Configurar el GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           scoring='neg_mean_squared_error',
                           cv=3, verbose=2, n_jobs=-1)

# Entrenar el modelo con todas las combinaciones de hiperparámetros
grid_search.fit(X_train, y_train)

# Mostrar los mejores hiperparámetros
print("Mejores hiperparámetros encontrados:")
print(grid_search.best_params_)

# Usar el mejor modelo para predecir en el conjunto de test
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# Evaluar el rendimiento
mse = mean_squared_error(y_test, y_pred)
print(f"Error cuadrático medio (MSE) en test: {mse}")
'''


## crear modelo inicial para empezar a aplicar el aprendizaje federado
# en este caso se unen los 2 datasets para aplicar el entrenamiento

import joblib

# Cargar datasets
data_machine1 = pd.read_csv('C:\\Users\\836582\\Downloads\\federated_working_IBARMIA.csv')
data_machine2 = pd.read_csv('C:\\Users\\836582\\Downloads\\federated_working_GMTK.csv')

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
joblib.dump(model, 'modelo_preentrenado_random_forest.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Modelo y escalador guardados correctamente.")